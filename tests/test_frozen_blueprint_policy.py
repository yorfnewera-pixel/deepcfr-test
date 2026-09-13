import numpy as np
import pokers as pkrs
import pytest
import torch

from src.core.action_space import NUM_ACTIONS, legal_action_mask
from src.core import deep_cfr as deep_cfr_mod
from src.core.deep_cfr import DeepCFRAgent
from src.core.model import CARD_CONTEXT_ARCHITECTURE, CARD_CONTEXT_V2_ARCHITECTURE, CARD_FEATURE_SIZE
from src.core.model import encode_state_for_version
from src.evaluation.blueprint_policy import FrozenBlueprintPolicy
from src.evaluation.paired_harness import evaluate_paired
from src.utils import config as config_mod


@pytest.fixture(autouse=True)
def _isolate_action_only_blueprint_checkpoints(monkeypatch):
    """Blueprint tests создают action-only checkpoint и не тестируют D2CFR."""
    original_cfg_get = deep_cfr_mod.cfg_get

    def cfg_get_for_action_only_tests(key, default=None):
        if key == "d2cfr_enabled":
            return False
        return original_cfg_get(key, default)

    monkeypatch.setattr(deep_cfr_mod, "cfg_get", cfg_get_for_action_only_tests)


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


def test_frozen_policy_rejects_wrong_v2_card_feature_metadata_before_loading(tmp_path):
    checkpoint = tmp_path / "card-context-v2-wrong-card-features.pt"
    payload = DeepCFRAgent(
        player_id=0,
        num_players=2,
        network_architecture=CARD_CONTEXT_V2_ARCHITECTURE,
    ).build_light_checkpoint()
    payload["card_feature_size"] = CARD_FEATURE_SIZE + 1
    payload["config"]["card_feature_size"] = CARD_FEATURE_SIZE + 1
    torch.save(payload, checkpoint)

    with pytest.raises(ValueError, match="размер card-признаков"):
        FrozenBlueprintPolicy.from_checkpoint(checkpoint, num_players=2)


def test_frozen_policy_rejects_non_mapping_light_config_before_loading(tmp_path):
    checkpoint = tmp_path / "malformed-config-light.pt"
    payload = DeepCFRAgent(player_id=0, num_players=2).build_light_checkpoint()
    payload["config"] = []
    torch.save(payload, checkpoint)

    with pytest.raises(ValueError, match="config"):
        FrozenBlueprintPolicy.from_checkpoint(checkpoint, num_players=2)


def test_frozen_policy_loads_actor_conditioned_hu_light_checkpoint_for_each_player(tmp_path):
    config_path = tmp_path / "hu.yaml"
    config_path.write_text(
        "\n".join(
            (
                "num_actions: 6",
                "num_players: 2",
                "num_trainable_players: 2",
                "hu_current_policy_self_play: true",
                "hidden_size: 8",
                "network_architecture: monolithic_v1",
            )
        ) + "\n",
        encoding="utf-8",
    )
    config_mod.load_config(config_path)
    try:
        agent = DeepCFRAgent(player_id=0, num_players=2, device="cpu")
        from src.training import train as train_mod

        train_mod._create_hu_current_policy_coordinator(agent)
        checkpoint_path = tmp_path / "hu-light.pt"
        torch.save(agent.build_light_checkpoint(seed=19), checkpoint_path)

        policy = FrozenBlueprintPolicy.from_checkpoint(checkpoint_path, num_players=2)
        first = pkrs.State.from_seed(
            n_players=2, button=0, sb=1.0, bb=2.0, stake=100.0, seed=19
        )
        second = first.apply_action(pkrs.Action(pkrs.ActionEnum.Call))

        def expected_probabilities(state, player_id):
            base = encode_state_for_version(state, player_id, agent.encoding_version)
            actor = np.zeros(2, dtype=np.float32)
            actor[player_id] = 1.0
            with torch.inference_mode():
                logits = agent.strategy_net(
                    torch.from_numpy(np.concatenate((base, actor))).unsqueeze(0)
                )
            logits = logits.clone()
            mask = legal_action_mask(state)
            logits[0, mask == 0.0] = -1e20
            return torch.softmax(logits, dim=1)[0].numpy()

        assert np.allclose(policy.probabilities(first, player_id=0), expected_probabilities(first, 0))
        assert np.allclose(policy.probabilities(second, player_id=1), expected_probabilities(second, 1))
        assert np.allclose(
            policy.probabilities_batch((first, second)),
            np.stack((expected_probabilities(first, 0), expected_probabilities(second, 1))),
        )
    finally:
        config_mod.load_config("config.yaml")


def _save_actor_conditioned_hu_full_pair(tmp_path, *, architecture="monolithic_v1"):
    config_path = tmp_path / "hu-full.yaml"
    config_path.write_text(
        "\n".join(
            (
                "num_actions: 6",
                "num_players: 2",
                "num_trainable_players: 2",
                "hu_current_policy_self_play: true",
                "hidden_size: 8",
                f"network_architecture: {architecture}",
            )
        ) + "\n",
        encoding="utf-8",
    )
    config_mod.load_config(config_path)
    agent = DeepCFRAgent(player_id=0, num_players=2, device="cpu")
    from src.training import train as train_mod

    train_mod._create_hu_current_policy_coordinator(agent)
    full_path = tmp_path / "hu-full.pt"
    light_path = tmp_path / "hu-light.pt"
    torch.save(train_mod._build_hu_checkpoint(agent, seed=19), full_path)
    torch.save(agent.build_light_checkpoint(seed=19), light_path)
    return agent, full_path, light_path


def test_frozen_policy_loads_hu_full_without_training_restore_and_preserves_actor_conditioning(
    tmp_path, monkeypatch
):
    agent, full_path, light_path = _save_actor_conditioned_hu_full_pair(tmp_path)
    try:
        def unexpected_training_agent(*_args, **_kwargs):
            pytest.fail("Frozen HU evaluation не должна создавать DeepCFRAgent")

        monkeypatch.setattr(
            "src.evaluation.blueprint_policy.DeepCFRAgent",
            unexpected_training_agent,
        )
        monkeypatch.setattr(
            torch.optim,
            "AdamW",
            lambda *_args, **_kwargs: pytest.fail(
                "Frozen HU evaluation не должна создавать optimizer"
            ),
        )

        inferred = FrozenBlueprintPolicy.from_checkpoint(full_path)
        explicit = FrozenBlueprintPolicy.from_checkpoint(full_path, num_players=2)
        light = FrozenBlueprintPolicy.from_checkpoint(light_path, num_players=2)
        first = pkrs.State.from_seed(
            n_players=2, button=0, sb=1.0, bb=2.0, stake=100.0, seed=19
        )
        second = first.apply_action(pkrs.Action(pkrs.ActionEnum.Call))

        assert inferred.agent is None
        assert inferred.num_players == 2
        assert not inferred.strategy_net.training
        assert all(not parameter.requires_grad for parameter in inferred.strategy_net.parameters())
        for state, player_id in ((first, 0), (first, 1), (second, 1)):
            assert np.allclose(
                inferred.probabilities(state, player_id=player_id),
                light.probabilities(state, player_id=player_id),
            )
        expected_batch = np.stack(
            (
                inferred.probabilities(first, player_id=0),
                inferred.probabilities(second, player_id=1),
            )
        )
        actual_batch = explicit.probabilities_batch((first, second))
        assert np.allclose(actual_batch, expected_batch)
        assert np.array_equal(
            actual_batch == 0.0,
            np.stack((legal_action_mask(first), legal_action_mask(second))) == 0.0,
        )
        assert np.allclose(
            inferred.strategy_net(torch.ones((1, agent.input_size + 2))),
            light.strategy_net(torch.ones((1, agent.input_size + 2))),
        )
    finally:
        config_mod.load_config("config.yaml")


def test_frozen_policy_rejects_hu_full_with_mismatched_requested_player_count(tmp_path):
    _, full_path, _ = _save_actor_conditioned_hu_full_pair(tmp_path)
    try:
        with pytest.raises(ValueError, match="двух игроков"):
            FrozenBlueprintPolicy.from_checkpoint(full_path, num_players=6)
    finally:
        config_mod.load_config("config.yaml")


@pytest.mark.parametrize(
    ("mutate", "message"),
    (
        (
            lambda payload: payload["architecture"]["strategy"].__setitem__(
                "network_architecture", "unknown"
            ),
            "архитектур",
        ),
        (
            lambda payload: payload["mode"].__setitem__(
                "encoder_input_size", payload["mode"]["encoder_input_size"] + 1
            ),
            "base encoder",
        ),
        (
            lambda payload: payload["architecture"]["strategy"].__setitem__(
                "input_size", payload["mode"]["encoder_input_size"]
            ),
            "actor",
        ),
    ),
)
def test_frozen_policy_rejects_malformed_hu_full_strategy_contract(
    tmp_path, mutate, message
):
    _, full_path, _ = _save_actor_conditioned_hu_full_pair(tmp_path)
    try:
        payload = torch.load(full_path, weights_only=False)
        mutate(payload)
        malformed_path = tmp_path / "malformed-hu-full.pt"
        torch.save(payload, malformed_path)

        with pytest.raises(ValueError, match=message):
            FrozenBlueprintPolicy.from_checkpoint(malformed_path)
    finally:
        config_mod.load_config("config.yaml")


def test_frozen_policy_validates_hu_full_card_context_v2_tensor_shapes(tmp_path):
    _, full_path, _ = _save_actor_conditioned_hu_full_pair(
        tmp_path, architecture=CARD_CONTEXT_V2_ARCHITECTURE
    )
    try:
        payload = torch.load(full_path, weights_only=False)
        payload["strategy"]["network"]["fusion.0.weight"] = torch.zeros((8, 15))
        malformed_path = tmp_path / "malformed-hu-full-v2.pt"
        torch.save(payload, malformed_path)

        with pytest.raises(ValueError, match="fusion"):
            FrozenBlueprintPolicy.from_checkpoint(malformed_path)
    finally:
        config_mod.load_config("config.yaml")
