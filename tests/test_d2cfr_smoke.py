"""Процессные smoke-проверки opt-in D2CFR на настоящем pkrs.State."""
from pathlib import Path

import numpy as np
import pytest
import torch

from src.training import train as train_mod
from src.utils import config as config_mod


_SMOKE_CONFIG = Path(__file__).resolve().parents[1] / "configs" / "hu_d2cfr_smoke.yaml"


def _finite_dueling_outputs(agent):
    state = torch.zeros((1, agent.input_size), dtype=torch.float32)
    for network in agent.hu_advantage_nets:
        output = network.forward_components(state)
        assert torch.isfinite(output.state_values).all()
        assert torch.isfinite(output.action_values).all()
        assert torch.isfinite(output.regrets).all()


def test_hu_d2_smoke_creates_full_and_light_checkpoint_and_resumes(tmp_path, capsys):
    config_mod.load_config(_SMOKE_CONFIG)
    try:
        agent = train_mod.train_self_play_multi(
            num_iterations=1,
            traversals_per_iteration=1,
            evaluate_every=0,
            save_dir=tmp_path,
            num_players=2,
            trainable_players=2,
            hu_current_policy_self_play=True,
            seed=20260910,
        )
        full_checkpoint = tmp_path / "hu_checkpoint_final.pt"
        light_checkpoint = tmp_path / "hu_light_checkpoint_final.pt"
        assert full_checkpoint.is_file()
        assert light_checkpoint.is_file()
        assert (tmp_path / "hu_checkpoint_iter_1.pt").is_file()
        assert (tmp_path / "hu_light_checkpoint_iter_1.pt").is_file()
        assert agent.iteration_count == 1
        assert agent.hu_advantage_target_nets is None
        _finite_dueling_outputs(agent)

        resumed = train_mod.train_self_play_multi(
            num_iterations=1,
            traversals_per_iteration=1,
            evaluate_every=0,
            save_dir=tmp_path,
            num_players=2,
            trainable_players=2,
            hu_current_policy_self_play=True,
            initial_checkpoint=full_checkpoint,
            seed=20260910,
        )
        assert resumed.iteration_count == 2
        _finite_dueling_outputs(resumed)
        policy = resumed.get_policy_distribution(train_mod._new_hand(2, 41), player_id=0)
        assert np.isfinite(policy).all()
        assert np.isclose(policy.sum(), 1.0)
        assert "D2CFR" in capsys.readouterr().out
    finally:
        config_mod.load_config("config.yaml")


def test_six_max_d2cfr_is_rejected_until_stage_two(tmp_path):
    config_mod.load_config(_SMOKE_CONFIG)
    try:
        with pytest.raises(ValueError, match="только HU"):
            train_mod.train_self_play_multi(
                num_iterations=1,
                traversals_per_iteration=1,
                evaluate_every=0,
                save_dir=tmp_path,
                num_players=6,
                trainable_players=1,
                hu_current_policy_self_play=False,
                seed=20260910,
            )
    finally:
        config_mod.load_config("config.yaml")


def test_heads_up_d2cfr_requires_current_policy_self_play(tmp_path):
    config_mod.load_config(_SMOKE_CONFIG)
    try:
        with pytest.raises(ValueError, match="current-policy self-play"):
            train_mod.train_self_play_multi(
                num_iterations=1,
                traversals_per_iteration=1,
                evaluate_every=0,
                save_dir=tmp_path,
                num_players=2,
                trainable_players=1,
                hu_current_policy_self_play=False,
                seed=20260910,
            )
    finally:
        config_mod.load_config("config.yaml")
