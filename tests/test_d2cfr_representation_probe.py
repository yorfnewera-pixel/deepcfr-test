import importlib
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

representation_probe = importlib.import_module("tools.d2cfr_representation_probe")


def test_vector_delta_reports_l2_cosine_and_changed_features():
    left = torch.tensor([1.0, 0.0, -2.0, 0.0])
    right = torch.tensor([1.0, 3.0, -1.0, 0.0])

    report = representation_probe.vector_delta(left, right, top_k=2)

    assert report["l2"] == pytest.approx(10.0**0.5)
    assert report["changed_count"] == 2
    assert report["top_abs_deltas"] == [
        {"index": 1, "left": 0.0, "right": 3.0, "delta": 3.0},
        {"index": 2, "left": -2.0, "right": -1.0, "delta": 1.0},
    ]
    assert -1.0 <= report["cosine"] <= 1.0


def test_compare_encoded_pair_reports_hidden_output_and_rm_deltas():
    class TinyNetwork(torch.nn.Module):
        def _encode(self, states):
            return states[:, :2] * 2.0

        def forward_components(self, states):
            batch = states.shape[0]
            return SimpleNamespace(
                action_values=torch.stack(
                    (
                        states[:, 0],
                        states[:, 1],
                        torch.zeros(batch),
                        torch.zeros(batch),
                        torch.zeros(batch),
                        torch.zeros(batch),
                    ),
                    dim=1,
                ),
                state_values=states[:, :1] * 0.5,
                regrets=torch.stack(
                    (
                        states[:, 0] - states[:, 0] * 0.5,
                        states[:, 1] - states[:, 0] * 0.5,
                        torch.zeros(batch),
                        torch.zeros(batch),
                        torch.zeros(batch),
                        torch.zeros(batch),
                    ),
                    dim=1,
                ),
            )

    report = representation_probe.compare_encoded_pair(
        TinyNetwork(),
        torch.tensor([1.0, 0.0, 0.0]),
        torch.tensor([0.0, 1.0, 0.0]),
        label="swap",
        legal_mask=torch.tensor([1, 1, 0, 0, 0, 0], dtype=torch.float32),
        top_k=2,
    )

    assert report["label"] == "swap"
    assert report["encoded"]["changed_count"] == 2
    assert report["hidden"]["l2"] > 0.0
    assert report["outputs"]["q"]["l2"] > 0.0
    assert report["outputs"]["rm_policy"]["l2"] > 0.0


def test_compare_target_prediction_pair_reports_bottleneck_ratios():
    class StaticNetwork(torch.nn.Module):
        def _encode(self, states):
            return states[:, :2]

        def forward_components(self, states):
            batch = states.shape[0]
            return SimpleNamespace(
                action_values=torch.zeros((batch, 6), dtype=torch.float32),
                state_values=torch.zeros((batch, 1), dtype=torch.float32),
                regrets=torch.zeros((batch, 6), dtype=torch.float32),
            )

    mask = torch.tensor([1, 1, 0, 0, 0, 0], dtype=torch.float32)
    left = representation_probe.TraversalTarget(
        encoded=torch.tensor([1.0, 0.0]),
        action_values=torch.tensor([1.0, 0.0, 0, 0, 0, 0]),
        state_value=torch.tensor(0.5),
        regrets=torch.tensor([0.5, -0.5, 0, 0, 0, 0]),
        mask=mask,
        root_return=0.5,
    )
    right = representation_probe.TraversalTarget(
        encoded=torch.tensor([0.0, 1.0]),
        action_values=torch.tensor([0.0, 1.0, 0, 0, 0, 0]),
        state_value=torch.tensor(0.5),
        regrets=torch.tensor([-0.5, 0.5, 0, 0, 0, 0]),
        mask=mask,
        root_return=0.5,
    )

    report = representation_probe.compare_target_prediction_pair(
        StaticNetwork(),
        left,
        right,
        label="target_changes_but_network_flat",
        top_k=2,
    )

    assert report["target"]["regrets"]["l2"] > 0.0
    assert report["prediction"]["regrets"]["l2"] == pytest.approx(0.0)
    assert report["bottleneck"]["regret_response_ratio"] == pytest.approx(0.0)
    assert report["bottleneck"]["legal_regret_response_ratio"] == pytest.approx(0.0)
    assert report["bottleneck"]["target_regret_l2_minus_prediction_regret_l2"] > 0.0


def test_focused_overfit_probe_teaches_fixed_target_pair():
    torch.manual_seed(7)

    class TrainableNetwork(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.trunk = torch.nn.Sequential(
                torch.nn.Linear(2, 16),
                torch.nn.Tanh(),
            )
            self.q_head = torch.nn.Linear(16, 6)
            self.v_head = torch.nn.Linear(16, 1)

        def _encode(self, states):
            return self.trunk(states)

        def forward_components(self, states):
            hidden = self._encode(states)
            q = self.q_head(hidden)
            v = self.v_head(hidden)
            return SimpleNamespace(
                action_values=q,
                state_values=v,
                regrets=q - v,
            )

    mask = torch.tensor([1, 1, 0, 0, 0, 0], dtype=torch.float32)
    left = representation_probe.TraversalTarget(
        encoded=torch.tensor([1.0, 0.0]),
        action_values=torch.tensor([1.0, -1.0, 0, 0, 0, 0]),
        state_value=torch.tensor(0.0),
        regrets=torch.tensor([1.0, -1.0, 0, 0, 0, 0]),
        mask=mask,
        root_return=1.0,
    )
    right = representation_probe.TraversalTarget(
        encoded=torch.tensor([0.0, 1.0]),
        action_values=torch.tensor([-1.0, 1.0, 0, 0, 0, 0]),
        state_value=torch.tensor(0.0),
        regrets=torch.tensor([-1.0, 1.0, 0, 0, 0, 0]),
        mask=mask,
        root_return=-1.0,
    )

    report = representation_probe.run_focused_overfit_probe(
        TrainableNetwork(),
        [{"label": "synthetic_pair", "left": left, "right": right}],
        steps=250,
        learning_rate=0.05,
        top_k=2,
    )

    pair = report["pairs"][0]
    assert report["mode"] == "focused_overfit_probe"
    assert report["steps"] == 250
    assert pair["status"] == "fits_fixed_targets"
    assert pair["after"]["loss"]["total"] < pair["before"]["loss"]["total"] * 0.05
    assert pair["after"]["comparison"]["bottleneck"]["legal_regret_response_ratio"] > 0.8


def test_representation_probe_script_can_run_from_tools_directory():
    script = Path(__file__).resolve().parents[1] / "tools" / "d2cfr_representation_probe.py"

    result = subprocess.run(
        [sys.executable, str(script), "--help"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    assert "--collect-traversals" in result.stdout
    assert "--focused-overfit-steps" in result.stdout
