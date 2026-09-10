import numpy as np
import pytest

from src.core.action_space import NUM_ACTIONS
from src.core.deep_cfr import DeepCFRAgent
from src.training import train as train_mod
from src.utils import config as config_mod


@pytest.fixture
def d2_hu_agent(tmp_path):
    config_path = tmp_path / "hu-d2cfr.yaml"
    config_path.write_text(
        "\n".join(
            (
                "num_actions: 6",
                "num_players: 2",
                "num_trainable_players: 2",
                "hu_current_policy_self_play: true",
                "hidden_size: 8",
                "d2cfr_enabled: true",
                "d2cfr_mc_correction_enabled: false",
                "advantage_memory_size: 8",
                "strategy_memory_size: 8",
            )
        )
        + "\n",
        encoding="utf-8",
    )
    config_mod.load_config(config_path)
    try:
        agent = DeepCFRAgent(player_id=0, num_players=2, device="cpu")
        train_mod._create_hu_current_policy_coordinator(agent)
        agent.iteration_count = 4
        mask = np.array([1, 1, 0, 0, 0, 0], dtype=np.float32)
        for player_id, buffer in enumerate(agent.hu_advantage_buffers):
            buffer.add(
                np.full(agent.input_size, player_id, dtype=np.float32),
                np.array([0.05, 0.02, 0, 0, 0, 0], dtype=np.float32),
                0.03,
                np.array([0.02, -0.01, 0, 0, 0, 0], dtype=np.float32),
                mask,
                iteration=4,
            )
        yield agent
    finally:
        config_mod.load_config("config.yaml")


def test_hu_d2_checkpoint_serializes_two_dueling_legs_without_target_networks(d2_hu_agent):
    checkpoint = train_mod._build_hu_checkpoint(d2_hu_agent, seed=17)

    assert checkpoint["algorithm_variant"] == "d2cfr_dueling_v1"
    assert "advantage_target" not in checkpoint["architecture"]
    assert all("target_network" not in leg for leg in checkpoint["advantage_legs"])
    assert all({"action_values", "state_values", "regrets"} <= set(leg["buffer"]) for leg in checkpoint["advantage_legs"])


def test_hu_d2_checkpoint_round_trip_restores_two_legs(tmp_path, d2_hu_agent):
    path = train_mod._save_hu_checkpoint(d2_hu_agent, tmp_path / "hu-d2.pt", seed=17)

    config_path = tmp_path / "restore.yaml"
    config_path.write_text(
        "\n".join(
            (
                "num_actions: 6", "num_players: 2", "num_trainable_players: 2",
                "hu_current_policy_self_play: true", "hidden_size: 8", "d2cfr_enabled: true",
                "d2cfr_mc_correction_enabled: false", "advantage_memory_size: 8", "strategy_memory_size: 8",
            )
        ) + "\n",
        encoding="utf-8",
    )
    config_mod.load_config(config_path)
    try:
        restored = DeepCFRAgent(player_id=0, num_players=2, device="cpu")
        train_mod._create_hu_current_policy_coordinator(restored)
        train_mod._load_hu_checkpoint(restored, path)
    finally:
        config_mod.load_config("config.yaml")

    assert restored.iteration_count == 4
    assert [len(buffer) for buffer in restored.hu_advantage_buffers] == [1, 1]
    assert restored.hu_advantage_target_nets is None
