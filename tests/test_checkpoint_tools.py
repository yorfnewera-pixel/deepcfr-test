import numpy as np
import torch

from src.core.deep_cfr import DeepCFRAgent
from src.core.model import CARD_CONTEXT_ARCHITECTURE
from tools import checkpoint_tools


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
