import numpy as np
import pokers as pkrs
import pytest
import torch

from src.core.action_space import NUM_ACTIONS, legal_action_mask
from src.core.deep_cfr import DeepCFRAgent
from src.core.model import CARD_CONTEXT_ARCHITECTURE, CARD_FEATURE_SIZE
from src.evaluation.blueprint_policy import FrozenBlueprintPolicy
from src.evaluation.paired_harness import evaluate_paired
from src.utils import config as config_mod


def test_frozen_policy_exposes_six_legal_probabilities(tmp_path):
    checkpoint = tmp_path / "six_fixed.pt"
    DeepCFRAgent(player_id=0, num_players=6).save_model(str(checkpoint))
    policy = FrozenBlueprintPolicy.from_checkpoint(checkpoint)
    state = pkrs.State.from_seed(n_players=6, button=0, sb=1.0, bb=2.0, stake=200.0, seed=107)
    probabilities = policy.probabilities(state)
    assert probabilities.shape == (NUM_ACTIONS,)
    assert np.isclose(probabilities.sum(), 1.0)
    assert np.all(probabilities[policy.agent.get_legal_action_mask(state) == 0.0] == 0.0)


def test_frozen_policy_loads_strategy_only_checkpoint_without_advantage_network(tmp_path):
    checkpoint = tmp_path / "strategy_only.pt"
    agent = DeepCFRAgent(player_id=0, num_players=6)
    torch.save(agent.build_light_checkpoint(), checkpoint)

    policy = FrozenBlueprintPolicy.from_checkpoint(checkpoint)
    state = pkrs.State.from_seed(n_players=6, button=0, sb=1.0, bb=2.0, stake=200.0, seed=107)
    probabilities = policy.probabilities(state)

    assert probabilities.shape == (NUM_ACTIONS,)
    assert np.isclose(probabilities.sum(), 1.0)
    assert np.all(np.isfinite(probabilities))
    assert np.all(probabilities >= 0.0)
    assert np.all(probabilities[legal_action_mask(state) == 0.0] == 0.0)


def test_frozen_policy_batch_probabilities_match_single_state_inference(tmp_path):
    checkpoint = tmp_path / "six_fixed.pt"
    DeepCFRAgent(player_id=0, num_players=6).save_model(str(checkpoint))
    policy = FrozenBlueprintPolicy.from_checkpoint(checkpoint)
    states = [
        pkrs.State.from_seed(n_players=6, button=0, sb=1.0, bb=2.0, stake=200.0, seed=107),
        pkrs.State.from_seed(n_players=6, button=1, sb=1.0, bb=2.0, stake=200.0, seed=211),
    ]

    probabilities = policy.probabilities_batch(states)

    assert probabilities.shape == (2, NUM_ACTIONS)
    assert np.all(np.isfinite(probabilities))
    assert np.allclose(probabilities.sum(axis=1), np.ones(2))
    assert np.allclose(probabilities, np.stack([policy.probabilities(state) for state in states]))
    for row, state in zip(probabilities, states):
        assert np.all(row[legal_action_mask(state) == 0.0] == 0.0)


def test_identical_frozen_policy_has_zero_paired_difference(tmp_path):
    checkpoint = tmp_path / "six_fixed.pt"
    DeepCFRAgent(player_id=0, num_players=2).save_model(str(checkpoint))
    baseline = FrozenBlueprintPolicy.from_checkpoint(checkpoint, num_players=2)
    candidate = FrozenBlueprintPolicy.from_checkpoint(checkpoint, num_players=2)
    result = evaluate_paired(baseline, candidate, num_deals=2, seed=107, num_players=2)
    assert np.array_equal(result.differences, np.zeros(result.samples))


def test_light_checkpoint_rejects_layer_width_inconsistent_with_encoder_metadata(tmp_path):
    checkpoint = tmp_path / "inconsistent_light.pt"
    sixmax = DeepCFRAgent(player_id=0, num_players=6)
    heads_up = DeepCFRAgent(player_id=0, num_players=2)
    payload = sixmax.build_light_checkpoint()
    payload["strategy_net"] = heads_up.strategy_net.state_dict()
    torch.save(payload, checkpoint)

    with pytest.raises(ValueError, match="размер входа encoder"):
        FrozenBlueprintPolicy.from_checkpoint(checkpoint)


def test_frozen_policy_loads_card_context_checkpoint_with_legal_policy(tmp_path):
    checkpoint = tmp_path / "card-context.pt"
    DeepCFRAgent(
        player_id=0,
        num_players=2,
        network_architecture=CARD_CONTEXT_ARCHITECTURE,
    ).save_model(str(checkpoint))

    policy = FrozenBlueprintPolicy.from_checkpoint(checkpoint, num_players=2)
    state = pkrs.State.from_seed(n_players=2, button=0, sb=1.0, bb=2.0, stake=200.0, seed=107)
    probabilities = policy.probabilities(state)

    assert policy.strategy_net.architecture == CARD_CONTEXT_ARCHITECTURE
    assert probabilities.shape == (NUM_ACTIONS,)
    assert np.isclose(probabilities.sum(), 1.0)
    assert np.all(probabilities[legal_action_mask(state) == 0.0] == 0.0)


def test_frozen_policy_uses_declared_hidden_size_for_full_card_context_checkpoint(tmp_path):
    checkpoint = tmp_path / "card-context-full.pt"
    small_config = tmp_path / "small-config.yaml"
    large_config = tmp_path / "large-config.yaml"
    small_config.write_text("num_actions: 6\nhidden_size: 8\n", encoding="utf-8")
    large_config.write_text("num_actions: 6\nhidden_size: 16\n", encoding="utf-8")

    try:
        config_mod.load_config(small_config)
        DeepCFRAgent(
            player_id=0,
            num_players=2,
            network_architecture=CARD_CONTEXT_ARCHITECTURE,
        ).save_model(str(checkpoint))
        config_mod.load_config(large_config)

        policy = FrozenBlueprintPolicy.from_checkpoint(checkpoint, num_players=2)

        assert policy.strategy_net.card_encoder[0].out_features == 8
    finally:
        config_mod.load_config("config.yaml")


def test_frozen_policy_rejects_card_context_weights_without_architecture_metadata(tmp_path):
    checkpoint = tmp_path / "card-context-without-metadata.pt"
    payload = DeepCFRAgent(
        player_id=0,
        num_players=2,
        network_architecture=CARD_CONTEXT_ARCHITECTURE,
    ).build_light_checkpoint()
    payload.pop("network_architecture")
    payload.pop("card_feature_size")
    payload["config"].pop("network_architecture")
    payload["config"].pop("card_feature_size")
    torch.save(payload, checkpoint)

    with pytest.raises(ValueError, match="метаданные архитектуры"):
        FrozenBlueprintPolicy.from_checkpoint(checkpoint, num_players=2)


def test_frozen_policy_rejects_wrong_card_feature_metadata(tmp_path):
    checkpoint = tmp_path / "card-context-wrong-card-features.pt"
    payload = DeepCFRAgent(
        player_id=0,
        num_players=2,
        network_architecture=CARD_CONTEXT_ARCHITECTURE,
    ).build_light_checkpoint()
    payload["card_feature_size"] = CARD_FEATURE_SIZE + 1
    payload["config"]["card_feature_size"] = CARD_FEATURE_SIZE + 1
    torch.save(payload, checkpoint)

    with pytest.raises(ValueError, match="размер card-признаков"):
        FrozenBlueprintPolicy.from_checkpoint(checkpoint, num_players=2)


def test_frozen_policy_rejects_explicit_null_architecture_metadata(tmp_path):
    checkpoint = tmp_path / "null-architecture.pt"
    payload = DeepCFRAgent(player_id=0, num_players=2).build_light_checkpoint()
    payload["network_architecture"] = None
    torch.save(payload, checkpoint)

    with pytest.raises(ValueError, match="некорректное значение архитектуры"):
        FrozenBlueprintPolicy.from_checkpoint(checkpoint, num_players=2)
