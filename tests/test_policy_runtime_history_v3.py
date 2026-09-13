import numpy as np
import pokers as pkrs
import pytest
import torch

import policy_runtime.core as runtime_core
from policy_runtime.adapters.pokers import wrap_state
from policy_runtime.core import (
    PolicyRuntimeAgent,
    encode_state_history_summary_v3 as runtime_encode_state,
)
from src.core.action_space import ActionSlot, resolve_action
from src.core.deep_cfr import DeepCFRAgent
from src.core.model import CARD_CONTEXT_V2_ARCHITECTURE
from src.core.model import encode_state_history_summary_v3 as training_encode_state
from src.training import train as train_mod
from src.utils import config as config_mod


def test_runtime_history_v3_encoder_matches_training_for_small_blind() -> None:
    state = pkrs.State.from_seed(
        n_players=2,
        button=0,
        sb=0.0025,
        bb=0.005,
        stake=2.0,
        seed=107,
    )

    training_vector = training_encode_state(state, player_id=int(state.current_player))
    runtime_vector = runtime_encode_state(
        wrap_state(state),
        player_id=int(state.current_player),
    )

    assert np.allclose(runtime_vector, training_vector)


def test_runtime_loads_actor_conditioned_hu_light_checkpoint(tmp_path) -> None:
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
        train_mod._create_hu_current_policy_coordinator(agent)
        checkpoint_path = tmp_path / "hu-light.pt"
        torch.save(agent.build_light_checkpoint(seed=19), checkpoint_path)

        runtime = PolicyRuntimeAgent(str(checkpoint_path))
        initial_state = pkrs.State.from_seed(
            n_players=2, button=0, sb=1.0, bb=2.0, stake=100.0, seed=19
        )
        button_action = runtime.choose_action(initial_state, player_id=0, deterministic=True)
        big_blind_state = initial_state.apply_action(
            resolve_action(ActionSlot.CALL, initial_state).action
        )
        big_blind_action = runtime.choose_action(big_blind_state, player_id=1, deterministic=True)

        assert button_action in {int(action) for action in initial_state.legal_actions}
        assert big_blind_action in {int(action) for action in big_blind_state.legal_actions}
    finally:
        config_mod.load_config("config.yaml")


def test_runtime_reconstructs_v2_light_checkpoint_with_identical_logits(tmp_path) -> None:
    agent = DeepCFRAgent(
        player_id=0,
        num_players=2,
        hidden_size=8,
        network_architecture=CARD_CONTEXT_V2_ARCHITECTURE,
    )
    checkpoint_path = tmp_path / "card-context-v2-light.pt"
    torch.save(agent.build_light_checkpoint(seed=19), checkpoint_path)

    runtime = PolicyRuntimeAgent(str(checkpoint_path))
    inputs = torch.randn(3, agent.input_size)

    assert runtime.strategy_net.architecture == CARD_CONTEXT_V2_ARCHITECTURE
    assert torch.equal(runtime.strategy_net(inputs), agent.strategy_net(inputs))


@pytest.mark.parametrize(
    "mutation",
    (
        "missing_metadata",
        "missing_fusion_weight",
        "malformed_card_rank",
        "malformed_action_bias",
    ),
)
def test_runtime_rejects_incomplete_v2_fusion_contract_before_loading(tmp_path, mutation) -> None:
    agent = DeepCFRAgent(
        player_id=0,
        num_players=2,
        hidden_size=8,
        network_architecture=CARD_CONTEXT_V2_ARCHITECTURE,
    )
    payload = agent.build_light_checkpoint(seed=19)
    if mutation == "missing_metadata":
        payload.pop("fusion_input_size")
        payload["config"].pop("fusion_input_size")
    elif mutation == "missing_fusion_weight":
        payload["strategy_net"].pop("fusion.0.weight")
    elif mutation == "malformed_card_rank":
        payload["strategy_net"]["card_encoder.0.weight"] = torch.zeros(8)
    else:
        payload["strategy_net"]["action_head.bias"] = torch.zeros(5)
    checkpoint_path = tmp_path / f"card-context-v2-{mutation}.pt"
    torch.save(payload, checkpoint_path)

    with pytest.raises(ValueError, match="fusion"):
        PolicyRuntimeAgent(str(checkpoint_path))


def test_runtime_rejects_v2_weights_forged_as_v1_before_load_state_dict(tmp_path, monkeypatch) -> None:
    agent = DeepCFRAgent(
        player_id=0,
        num_players=2,
        hidden_size=8,
        network_architecture=CARD_CONTEXT_V2_ARCHITECTURE,
    )
    payload = agent.build_light_checkpoint(seed=19)
    payload["network_architecture"] = "card_context_v1"
    payload["config"]["network_architecture"] = "card_context_v1"
    checkpoint_path = tmp_path / "forged-v1-runtime.pt"
    torch.save(payload, checkpoint_path)
    load_calls = []

    def unexpected_load(*args, **kwargs):
        load_calls.append((args, kwargs))
        raise AssertionError("load_state_dict не должен вызываться для forged architecture")

    monkeypatch.setattr(runtime_core.PokerNetwork, "load_state_dict", unexpected_load)

    with pytest.raises(ValueError, match="card_context_v1.*веса"):
        PolicyRuntimeAgent(str(checkpoint_path))

    assert load_calls == []
