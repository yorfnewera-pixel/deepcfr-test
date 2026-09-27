import importlib
import numpy as np
import pytest
import subprocess
import sys
import torch
from pathlib import Path
from types import SimpleNamespace

from src.core.buffers import DuelingAdvantageBuffer
from src.core.deep_cfr import DeepCFRAgent
from src.core.model import DuelingRegretNetwork
from src.training import train as train_mod
from src.utils import config as config_mod
from tools import d2cfr_signal_probe as signal_probe

accumulation_probe = importlib.import_module("tools.d2cfr_accumulation_probe")


def _sample(state, state_value, regrets, mask):
    action_values = np.asarray(regrets, dtype=np.float32) + np.float32(state_value)
    return (
        np.asarray(state, dtype=np.float32),
        action_values,
        np.float32(state_value),
        np.asarray(regrets, dtype=np.float32),
        np.asarray(mask, dtype=np.float32),
        np.float32(1.0),
    )


def test_regret_matching_policy_uses_positive_regrets_or_uniform():
    mask = torch.tensor([1, 1, 0, 1, 0, 0], dtype=torch.float32)

    policy = accumulation_probe.regret_matching_policy(
        torch.tensor([-1.0, 2.0, 9.0, 1.0, 0.0, 0.0]),
        mask,
    )

    assert policy.tolist() == pytest.approx([0.0, 2.0 / 3.0, 0.0, 1.0 / 3.0, 0.0, 0.0])

    uniform = accumulation_probe.regret_matching_policy(
        torch.tensor([-1.0, 0.0, 9.0, -2.0, 0.0, 0.0]),
        mask,
    )

    assert uniform.tolist() == pytest.approx([1.0 / 3.0, 1.0 / 3.0, 0.0, 1.0 / 3.0, 0.0, 0.0])


def test_evaluate_prediction_snapshot_reports_regret_and_rm_errors():
    class StaticNetwork(torch.nn.Module):
        def forward_components(self, states):
            batch_size = states.shape[0]
            return SimpleNamespace(
                action_values=torch.tensor(
                    [[0.1, 0.2, 0, 0, 0, 0], [0.7, -0.1, 0, 0, 0, 0]],
                    dtype=torch.float32,
                )[:batch_size],
                state_values=torch.tensor([[0.0], [0.2]], dtype=torch.float32)[:batch_size],
                regrets=torch.tensor(
                    [[0.1, -0.2, 0, 0, 0, 0], [0.5, -0.3, 0, 0, 0, 0]],
                    dtype=torch.float32,
                )[:batch_size],
            )

    mask = np.array([1, 1, 0, 0, 0, 0], dtype=np.float32)
    batch = signal_probe.build_probe_batch(
        [
            _sample([1.0, 0.0], 0.0, [0.3, -0.1, 0, 0, 0, 0], mask),
            _sample([0.0, 1.0], 0.1, [0.2, -0.2, 0, 0, 0, 0], mask),
        ]
    )

    report = accumulation_probe.evaluate_prediction_snapshot(StaticNetwork(), batch)

    assert report["samples"] == 2
    assert report["legal_slots"] == 4
    assert report["regret_abs_error_mean"] == pytest.approx(0.175)
    assert report["state_value_abs_error_mean"] == pytest.approx(0.05)
    assert report["rm_policy_l1_mean"] == pytest.approx(0.0)
    assert report["target_regret_abs_mean"] == pytest.approx(0.2)
    assert report["predicted_regret_abs_mean"] == pytest.approx(0.275)
    assert report["regret_mse"] == pytest.approx(0.0375)
    assert report["state_value_mse"] == pytest.approx(0.005)
    assert report["eval_loss_mse"] == pytest.approx(0.04)


def test_samples_from_dueling_buffer_for_iteration_returns_only_current_targets():
    buffer = DuelingAdvantageBuffer(capacity=8, state_dim=2)
    mask = np.array([1, 1, 0, 0, 0, 0], dtype=np.float32)
    for iteration, state_value in [(1, 0.1), (2, 0.2), (2, 0.3)]:
        buffer.add(
            np.array([float(iteration), state_value], dtype=np.float32),
            np.array([state_value + 0.1, state_value - 0.1, 0, 0, 0, 0], dtype=np.float32),
            state_value,
            np.array([0.1, -0.1, 0, 0, 0, 0], dtype=np.float32),
            mask,
            iteration,
        )

    samples = accumulation_probe.samples_from_dueling_buffer_for_iteration(
        buffer,
        iteration=2,
        sample_count=8,
        seed=3,
    )

    assert len(samples) == 2
    assert {float(sample[5]) for sample in samples} == {2.0}


def test_evaluate_prediction_snapshot_reports_sign_margin_buckets():
    class StaticNetwork(torch.nn.Module):
        def forward_components(self, states):
            batch_size = states.shape[0]
            return SimpleNamespace(
                action_values=torch.zeros((batch_size, 6), dtype=torch.float32),
                state_values=torch.zeros((batch_size, 1), dtype=torch.float32),
                regrets=torch.tensor([[0.02, -0.02, -0.2, 0.2, 0.0, 0.0]], dtype=torch.float32)[
                    :batch_size
                ],
            )

    mask = np.array([1, 1, 1, 1, 0, 0], dtype=np.float32)
    batch = signal_probe.build_probe_batch(
        [_sample([1.0, 0.0], 0.0, [-0.005, 0.005, 0.2, -0.2, 0, 0], mask)]
    )

    report = accumulation_probe.evaluate_prediction_snapshot(StaticNetwork(), batch)
    buckets = report["sign_margin"]

    assert buckets["abs_le_0.01"]["legal_slots"] == 2
    assert buckets["abs_le_0.01"]["sign_flip_rate"] == pytest.approx(1.0)
    assert buckets["abs_gt_0.1"]["legal_slots"] == 2
    assert buckets["abs_gt_0.1"]["sign_flip_rate"] == pytest.approx(1.0)
    assert buckets["overall"]["sign_flips"] == 4


def test_run_weight_mode_fit_ab_keeps_initial_network_equal_and_changes_only_sample_weights():
    mask = np.array([1, 1, 0, 0, 0, 1], dtype=np.float32)
    batch = signal_probe.build_probe_batch([
        (*_sample([1.0, 0.0], 0.0, [0.2, -0.1, 0, 0, 0, 0], mask)[:5], np.float32(1.0)),
        (*_sample([0.0, 1.0], 0.1, [0.1, -0.2, 0, 0, 0, 0], mask)[:5], np.float32(1.0)),
        (*_sample([1.0, 1.0], -0.1, [-0.3, 0.3, 0, 0, 0, 0], mask)[:5], np.float32(10.0)),
        (*_sample([0.5, 1.0], 0.2, [0.4, -0.2, 0, 0, 0, 0], mask)[:5], np.float32(10.0)),
    ])

    report = accumulation_probe.run_weight_mode_fit_ab(
        network_factory=lambda: DuelingRegretNetwork(input_size=2, hidden_size=8),
        train_batch=batch,
        evaluation_batch=batch,
        steps=2,
        batch_size=2,
        learning_rate=0.01,
        state_value_loss_weight=0.5,
        seed=13,
    )

    assert report["arms"]["uniform"]["final_train"]["samples"] == 4
    assert "all_in" in report["arms"]["uniform"]["final_train_by_action"]

    weighted = report["arms"]["iteration_weighted"]
    uniform = report["arms"]["uniform"]
    assert weighted["initial"]["eval_loss_mse"] == pytest.approx(uniform["initial"]["eval_loss_mse"])
    assert weighted["weight_summary"]["normalization"] == "per_minibatch_mean_1"
    assert weighted["weight_summary"]["batch_mean_minimum"] == pytest.approx(1.0)
    assert weighted["weight_summary"]["batch_mean_maximum"] == pytest.approx(1.0)
    assert uniform["weight_summary"]["normalization"] == "uniform"
    assert weighted["final"]["samples"] == 4
    assert uniform["final"]["samples"] == 4
    assert "all_in" in weighted["initial_by_action"]
    assert weighted["initial_by_action"]["all_in"]["q_abs_error_mean"] == pytest.approx(
        uniform["initial_by_action"]["all_in"]["q_abs_error_mean"]
    )


def test_replay_freshness_fit_ab_uses_only_recent_labels_in_recent_arm():
    mask = np.array([1, 1, 0, 0, 0, 1], dtype=np.float32)
    full_batch = signal_probe.build_probe_batch([
        (*_sample([1.0, 0.0], 0.0, [0.2, -0.1, 0, 0, 0, 0], mask)[:5], np.float32(1.0)),
        (*_sample([0.0, 1.0], 0.1, [0.1, -0.2, 0, 0, 0, 0], mask)[:5], np.float32(2.0)),
        (*_sample([1.0, 1.0], -0.1, [-0.3, 0.3, 0, 0, 0, 0], mask)[:5], np.float32(9.0)),
        (*_sample([0.5, 1.0], 0.2, [0.4, -0.2, 0, 0, 0, 0], mask)[:5], np.float32(10.0)),
    ])
    recent_batch = accumulation_probe._slice_probe_batch(full_batch, torch.tensor([2, 3]))

    report = accumulation_probe.run_replay_freshness_fit_ab(
        network_factory=lambda: DuelingRegretNetwork(input_size=2, hidden_size=8),
        full_batch=full_batch,
        recent_batch=recent_batch,
        evaluation_batch=full_batch,
        steps=2,
        batch_size=2,
        learning_rate=0.01,
        state_value_loss_weight=0.5,
        seed=13,
    )

    full = report["arms"]["full_replay"]
    recent = report["arms"]["recent_window"]
    assert full["train_samples"] == 4
    assert recent["train_samples"] == 2
    assert full["source_iteration_minimum"] == pytest.approx(1.0)
    assert recent["source_iteration_minimum"] == pytest.approx(9.0)
    assert full["initial"]["eval_loss_mse"] == pytest.approx(recent["initial"]["eval_loss_mse"])


def test_action_prediction_metrics_groups_all_in_sign_flips_by_target_margin():
    class StaticNetwork(torch.nn.Module):
        def forward_components(self, states):
            count = states.shape[0]
            return SimpleNamespace(
                action_values=torch.zeros((count, 6), dtype=torch.float32),
                state_values=torch.zeros((count, 1), dtype=torch.float32),
                regrets=torch.tensor(
                    [[0, 0, 0, 0, 0, -0.01], [0, 0, 0, 0, 0, -0.20]], dtype=torch.float32
                )[:count],
            )

    mask = np.array([0, 0, 0, 0, 0, 1], dtype=np.float32)
    batch = signal_probe.build_probe_batch([
        _sample([1.0, 0.0], 0.0, [0, 0, 0, 0, 0, 0.005], mask),
        _sample([0.0, 1.0], 0.0, [0, 0, 0, 0, 0, 0.20], mask),
    ])

    report = accumulation_probe._action_prediction_metrics(StaticNetwork(), batch)["all_in"]["sign_margin"]

    assert report["abs_le_0.01"]["legal_slots"] == 1
    assert report["abs_le_0.01"]["sign_flips"] == 1
    assert report["abs_gt_0.1"]["legal_slots"] == 1
    assert report["abs_gt_0.1"]["sign_flips"] == 1


def test_checkpoint_weight_mode_ab_uses_frozen_postflop_targets_and_writes_report(tmp_path):
    config_path = tmp_path / "probe.yaml"
    config_path.write_text(
        "\n".join((
            "num_actions: 6", "num_players: 2", "num_trainable_players: 2",
            "hu_current_policy_self_play: true", "hidden_size: 8", "d2cfr_enabled: true",
            "d2cfr_mc_correction_enabled: false", "advantage_memory_size: 32", "strategy_memory_size: 32",
        )) + "\n",
        encoding="utf-8",
    )
    config_mod.load_config(config_path)
    try:
        source = DeepCFRAgent(player_id=0, num_players=2, device="cpu")
        train_mod._create_hu_current_policy_coordinator(source)
        mask = np.array([1, 1, 0, 0, 0, 0], dtype=np.float32)
        for iteration in (1, 4):
            source.hu_advantage_buffers[0].add(
                np.zeros(source.input_size, dtype=np.float32),
                np.zeros(6, dtype=np.float32), 0.0, np.zeros(6, dtype=np.float32), mask, iteration,
            )
        source.iteration_count = 4
        checkpoint_path = train_mod._save_hu_checkpoint(source, tmp_path / "hu.pt")
        output_path = tmp_path / "weight-ab.json"
        report = accumulation_probe.run_checkpoint_weight_mode_ab_probe(
            checkpoint_path,
            config_path=config_path,
            output_path=output_path,
            fit_samples=2,
            fit_steps=1,
            fit_batch_size=2,
            frozen_states=1,
            frozen_repeats=2,
            frozen_roots=64,
            seed=17,
        )
    finally:
        config_mod.load_config("config.yaml")

    assert output_path.is_file()
    assert report["frozen_postflop"]["states_collected"] == 1
    comparison = report["fit"]["comparison"]
    assert comparison["arms"]["uniform"]["final"]["samples"] == 1
    assert "all_in" in comparison["arms"]["uniform"]["final_by_action"]


def test_checkpoint_replay_freshness_probe_limits_recent_arm_to_window(tmp_path):
    config_path = tmp_path / "probe.yaml"
    config_path.write_text(
        "\n".join((
            "num_actions: 6", "num_players: 2", "num_trainable_players: 2",
            "hu_current_policy_self_play: true", "hidden_size: 8", "d2cfr_enabled: true",
            "d2cfr_mc_correction_enabled: false", "advantage_memory_size: 32", "strategy_memory_size: 32",
        )) + "\n",
        encoding="utf-8",
    )
    config_mod.load_config(config_path)
    try:
        source = DeepCFRAgent(player_id=0, num_players=2, device="cpu")
        train_mod._create_hu_current_policy_coordinator(source)
        mask = np.array([1, 1, 0, 0, 0, 0], dtype=np.float32)
        for iteration in (1, 4):
            for _ in range(2):
                source.hu_advantage_buffers[0].add(
                    np.zeros(source.input_size, dtype=np.float32),
                    np.zeros(6, dtype=np.float32), 0.0, np.zeros(6, dtype=np.float32), mask, iteration,
                )
        source.iteration_count = 4
        checkpoint_path = train_mod._save_hu_checkpoint(source, tmp_path / "hu.pt")
        report = accumulation_probe.run_checkpoint_replay_freshness_probe(
            checkpoint_path,
            config_path=config_path,
            output_path=tmp_path / "freshness.json",
            recent_iterations=1,
            fit_samples=2,
            fit_steps=1,
            fit_batch_size=2,
            frozen_states=1,
            frozen_repeats=2,
            frozen_roots=64,
            seed=17,
        )
    finally:
        config_mod.load_config("config.yaml")

    recent = report["fit"]["comparison"]["arms"]["recent_window"]
    assert recent["train_samples"] == 2
    assert recent["source_iteration_minimum"] == pytest.approx(4.0)


def test_accumulation_probe_script_can_run_from_tools_directory():
    script = Path(__file__).resolve().parents[1] / "tools" / "d2cfr_accumulation_probe.py"

    result = subprocess.run(
        [sys.executable, str(script), "--help"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    assert "--iterations" in result.stdout
    assert "--weight-mode-ab" in result.stdout
    assert "--replay-freshness-ab" in result.stdout


def test_heldout_fit_probe_trains_clone_and_reports_train_validation_split():
    torch.manual_seed(3)
    mask = np.array([1, 1, 0, 0, 0, 0], dtype=np.float32)
    samples = [
        _sample([1.0, 0.0], 0.1, [0.2, -0.1, 0, 0, 0, 0], mask),
        _sample([0.0, 1.0], -0.2, [-0.3, 0.1, 0, 0, 0, 0], mask),
        _sample([1.0, 1.0], 0.3, [0.4, -0.2, 0, 0, 0, 0], mask),
        _sample([0.5, 1.0], -0.1, [-0.2, 0.2, 0, 0, 0, 0], mask),
    ]
    batch = signal_probe.build_probe_batch(samples)
    network = DuelingRegretNetwork(input_size=2, hidden_size=8)
    before = {name: value.detach().clone() for name, value in network.state_dict().items()}

    report = accumulation_probe.run_heldout_fit_probe(
        network,
        batch,
        train_fraction=0.5,
        seed=5,
        steps=5,
        learning_rate=0.05,
    )

    assert report["train"]["samples"] == 2
    assert report["validation"]["samples"] == 2
    assert report["history"][0]["train"]["eval_loss_mse"] >= report["history"][-1]["train"]["eval_loss_mse"]
    assert "generalization_gap_eval_loss_mse" in report
    for name, value in network.state_dict().items():
        assert torch.equal(value, before[name])
