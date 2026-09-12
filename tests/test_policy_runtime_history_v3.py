import numpy as np
import pokers as pkrs
import torch

from policy_runtime.adapters.pokers import wrap_state
from policy_runtime.core import (
    PolicyRuntimeAgent,
    encode_state_history_summary_v3 as runtime_encode_state,
)
from src.core.action_space import ActionSlot, resolve_action
from src.core.deep_cfr import DeepCFRAgent
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
