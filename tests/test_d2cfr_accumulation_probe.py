import importlib
import numpy as np
import pytest
import subprocess
import sys
import torch
from pathlib import Path
from types import SimpleNamespace

from src.core.buffers import DuelingAdvantageBuffer
from src.core.model import DuelingRegretNetwork
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
