import numpy as np
import pytest
import torch

from src.core import deep_cfr as deep_cfr_mod
from src.core.deep_cfr import DeepCFRAgent
from src.core.model import CARD_CONTEXT_ARCHITECTURE
from src.utils import config as config_mod
from tools import checkpoint_tools


@pytest.fixture(autouse=True)
def _isolate_action_only_checkpoint_tools(monkeypatch):
    """Эти проверки строят legacy action-only full checkpoint намеренно."""
    original_cfg_get = deep_cfr_mod.cfg_get

    def cfg_get_for_action_only_tests(key, default=None):
        if key == "d2cfr_enabled":
            return False
        return original_cfg_get(key, default)

    monkeypatch.setattr(deep_cfr_mod, "cfg_get", cfg_get_for_action_only_tests)


def _light_checkpoint(path):
    agent = DeepCFRAgent(player_id=0, num_players=6)
    torch.save(agent.build_light_checkpoint(), path)


def test_diagnose_accepts_strategy_only_light_checkpoint(tmp_path):
    checkpoint = tmp_path / "light.pt"
    _light_checkpoint(checkpoint)

    result = checkpoint_tools.run_weight_sanity(checkpoint)

    assert result["ok"] is True
    assert result["errors"] == []


def test_paired_evaluation_of_identical_checkpoints_has_zero_difference(tmp_path):
    checkpoint = tmp_path / "light.pt"
    _light_checkpoint(checkpoint)

    result = checkpoint_tools.evaluate_paired_checkpoints(
        checkpoint,
        checkpoint,
        games=1,
        seed=7,
    )

    assert result["samples"] == 1
    assert result["seat_samples"] == 6
    assert result["mean_difference"] == 0.0
    assert result["bb_per_100"] == 0.0
    assert result["ci95_low"] == 0.0
    assert result["ci95_high"] == 0.0


def test_profile_reports_action_distribution_and_flop_outcomes(tmp_path):
    checkpoint = tmp_path / "light.pt"
    _light_checkpoint(checkpoint)

    result = checkpoint_tools.profile_checkpoint(checkpoint, games=2, seed=7)

    assert result["games"] == 2
    assert set(result["actions_by_street"]) == {"preflop", "flop", "turn", "river"}
    assert sum(item["count"] for item in result["actions_by_street"]["preflop"].values()) > 0
    assert set(result["flop_paths"]) == {"raise", "three_bet", "all_in"}
    assert all(np.isfinite(item["mean_reward"]) for item in result["flop_paths"].values())


def test_profile_supports_heads_up_checkpoint(tmp_path):
    checkpoint = tmp_path / "hu_light.pt"
    agent = DeepCFRAgent(player_id=0, num_players=2)
    torch.save(agent.build_light_checkpoint(), checkpoint)

    result = checkpoint_tools.profile_checkpoint(checkpoint, games=1, seed=7)

    assert result["games"] == 1


def test_flop_probe_of_identical_checkpoints_has_zero_policy_difference(tmp_path):
    checkpoint = tmp_path / "light.pt"
    _light_checkpoint(checkpoint)

    result = checkpoint_tools.probe_flop_probabilities(
        checkpoint,
        checkpoint,
        games=200,
        seed=7,
    )

    assert result["samples"] > 0
    assert result["mean_l1_distance"] == 0.0
    assert result["argmax_changes"] == 0
    assert all(item["difference"] == 0.0 for item in result["actions"].values())


def test_full_probe_of_identical_checkpoints_has_zero_network_differences(tmp_path):
    checkpoint = tmp_path / "full.pt"
    agent = DeepCFRAgent(player_id=0, num_players=6)
    torch.save(agent._build_checkpoint(), checkpoint)

    result = checkpoint_tools.probe_full_checkpoints(
        checkpoint,
        checkpoint,
        games=200,
        seed=7,
    )

    assert result["samples"] > 0
    assert result["strategy"]["mean_l1_distance"] == 0.0
    assert all(item["difference"] == 0.0 for item in result["advantages"].values())
    assert result["buffers"]["advantage"]["present"] is True
    assert result["buffers"]["advantage"]["baseline"]["count"] == 0


def test_full_probe_loads_card_context_checkpoint(tmp_path):
    checkpoint = tmp_path / "card-context-full.pt"
    agent = DeepCFRAgent(
        player_id=0,
        num_players=6,
        network_architecture=CARD_CONTEXT_ARCHITECTURE,
    )
    torch.save(agent._build_checkpoint(), checkpoint)

    result = checkpoint_tools.probe_full_checkpoints(
        checkpoint,
        checkpoint,
        games=200,
        seed=7,
    )

    assert result["samples"] > 0
    assert result["strategy"]["mean_l1_distance"] == 0.0


def test_full_probe_uses_declared_hidden_size_for_card_context_checkpoint(tmp_path):
    checkpoint = tmp_path / "card-context-full.pt"
    small_config = tmp_path / "small-config.yaml"
    large_config = tmp_path / "large-config.yaml"
    small_config.write_text("num_actions: 6\nhidden_size: 8\n", encoding="utf-8")
    large_config.write_text("num_actions: 6\nhidden_size: 16\n", encoding="utf-8")

    try:
        config_mod.load_config(small_config)
        agent = DeepCFRAgent(
            player_id=0,
            num_players=6,
            network_architecture=CARD_CONTEXT_ARCHITECTURE,
        )
        torch.save(agent._build_checkpoint(), checkpoint)
        config_mod.load_config(large_config)

        result = checkpoint_tools.probe_full_checkpoints(
            checkpoint,
            checkpoint,
            games=200,
            seed=7,
        )

        assert result["samples"] > 0
        assert result["strategy"]["mean_l1_distance"] == 0.0
    finally:
        config_mod.load_config("config.yaml")
