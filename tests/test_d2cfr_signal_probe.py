import numpy as np
import pytest
import subprocess
import sys
import torch
from pathlib import Path

from src.core.action_space import NUM_ACTIONS
from src.core.buffers import DuelingAdvantageBuffer
from src.core.model import DuelingRegretNetwork
from tools import d2cfr_signal_probe as signal_probe


def _sample(state, state_value, regrets, mask):
    action_values = np.asarray(regrets, dtype=np.float32) + np.float32(state_value)
    return (
        np.asarray(state, dtype=np.float32),
        action_values,
        np.float32(state_value),
        np.asarray(regrets, dtype=np.float32),
        np.asarray(mask, dtype=np.float32),
        np.float32(3.0),
    )


def test_analyse_targets_reports_zero_fraction_and_q_minus_v_identity():
    mask = np.array([1, 1, 0, 0, 0, 0], dtype=np.float32)
    samples = [
        _sample([1.0, 0.0, 0.5], 0.25, [0.0, 0.5, 0, 0, 0, 0], mask),
        _sample([0.0, 1.0, 0.5], -0.25, [0.0, -0.5, 0, 0, 0, 0], mask),
    ]
    batch = signal_probe.build_probe_batch(samples)

    report = signal_probe.analyse_targets(batch)

    assert report["samples"] == 2
    assert report["legal_slots"] == 4
    assert report["q_minus_v_ok"] is True
    assert report["q_minus_v_max_abs_error"] == pytest.approx(0.0)
    assert report["regret_legal"]["zero_fraction"] == pytest.approx(0.5)
    assert report["regret_legal"]["std"] > 0.0


def test_signal_probe_trains_clone_and_reports_gradients_without_mutating_source():
    torch.manual_seed(7)
    mask = np.array([1, 1, 0, 0, 0, 0], dtype=np.float32)
    samples = [
        _sample([1.0, 0.0, 0.5], 0.2, [0.4, -0.2, 0, 0, 0, 0], mask),
        _sample([0.0, 1.0, 0.5], -0.1, [-0.3, 0.1, 0, 0, 0, 0], mask),
    ]
    batch = signal_probe.build_probe_batch(samples)
    network = DuelingRegretNetwork(input_size=3, hidden_size=8)
    before = {name: value.detach().clone() for name, value in network.state_dict().items()}

    report = signal_probe.run_one_step_signal_probe(
        network,
        batch,
        learning_rate=0.05,
        state_value_loss_weight=0.5,
    )

    assert report["loss_after"] < report["loss_before"]
    assert report["grad_norm_total"] > 0.0
    assert report["update_norm_total"] > 0.0
    assert report["modules"]["action_value_head"]["grad_norm"] > 0.0
    assert report["modules"]["state_value_head"]["grad_norm"] > 0.0
    for name, value in network.state_dict().items():
        assert torch.equal(value, before[name])


def test_microfit_probe_shows_trunk_gradients_after_heads_become_nonzero():
    torch.manual_seed(7)
    mask = np.array([1, 1, 0, 0, 0, 0], dtype=np.float32)
    samples = [
        _sample([1.0, 0.0, 0.5], 0.2, [0.4, -0.2, 0, 0, 0, 0], mask),
        _sample([0.0, 1.0, 0.5], -0.1, [-0.3, 0.1, 0, 0, 0, 0], mask),
    ]
    batch = signal_probe.build_probe_batch(samples)
    network = DuelingRegretNetwork(input_size=3, hidden_size=8)

    report = signal_probe.run_microfit_signal_probe(
        network,
        batch,
        steps=4,
        learning_rate=0.05,
        state_value_loss_weight=0.5,
    )

    assert report["initial_loss"] > report["final_loss"]
    assert report["history"][0]["modules"]["base"]["grad_norm"] == pytest.approx(0.0)
    assert max(item["modules"]["base"]["grad_norm"] for item in report["history"][1:]) > 0.0
    assert report["max_module_grad_norms"]["base"] > 0.0


def test_describe_sample_tensors_exposes_input_indices_targets_and_predictions():
    mask = np.array([1, 0, 1, 0, 0, 0], dtype=np.float32)
    samples = [
        _sample([0.0, 2.0, -3.0], 0.25, [0.5, 0, -0.25, 0, 0, 0], mask),
    ]
    batch = signal_probe.build_probe_batch(samples)
    network = DuelingRegretNetwork(input_size=3, hidden_size=8)

    report = signal_probe.describe_sample_tensors(network, batch, sample_limit=1, max_features=2)

    assert report["input_size"] == 3
    sample = report["samples"][0]
    assert sample["legal_slots"] == [0, 2]
    assert sample["input"]["nonzero_count"] == 2
    assert sample["input"]["top_abs_features"] == [
        {"index": 2, "value": -3.0},
        {"index": 1, "value": 2.0},
    ]
    assert sample["targets"]["state_value"] == pytest.approx(0.25)
    assert sample["targets"]["regrets"] == pytest.approx([0.5, 0, -0.25, 0, 0, 0])
    assert sample["targets"]["legal"] == [
        {"slot": 0, "q": pytest.approx(0.75), "regret": pytest.approx(0.5)},
        {"slot": 2, "q": pytest.approx(0.0), "regret": pytest.approx(-0.25)},
    ]
    assert sample["network_before"]["regrets"] == pytest.approx([0, 0, 0, 0, 0, 0])


def test_build_probe_batch_rejects_empty_samples():
    with pytest.raises(ValueError, match="samples"):
        signal_probe.build_probe_batch([])


def test_samples_from_dueling_buffer_uses_live_buffer_without_checkpoint():
    buffer = DuelingAdvantageBuffer(capacity=4, state_dim=3)
    mask = np.array([1, 1, 0, 0, 0, 0], dtype=np.float32)
    buffer.add(
        np.array([1.0, 0.0, 0.0], dtype=np.float32),
        np.array([0.4, 0.1, 0, 0, 0, 0], dtype=np.float32),
        0.2,
        np.array([0.2, -0.1, 0, 0, 0, 0], dtype=np.float32),
        mask,
        1,
    )

    samples = signal_probe.samples_from_dueling_buffer(buffer, sample_count=1, seed=7)

    batch = signal_probe.build_probe_batch(samples)
    assert batch.states.shape == (1, 3)
    assert signal_probe.analyse_targets(batch)["q_minus_v_ok"] is True


def test_collect_live_probe_runs_traversal_without_training():
    class FakeAgent:
        def __init__(self):
            self.d2cfr_enabled = True
            self.advantage_net = DuelingRegretNetwork(input_size=3, hidden_size=8)
            self.d2cfr_buffer = DuelingAdvantageBuffer(capacity=8, state_dim=3)
            self.prepared = []
            self.training_calls = 0
            self.traversed_states = []

        def prepare_iteration(self, iteration, traversing_player):
            self.prepared.append((iteration, traversing_player))

        def reset_traversal_stats(self):
            self.traversed_states.clear()

        def cfr_traverse_multi(self, state, iteration, traversing_player):
            self.traversed_states.append((state, iteration, traversing_player))
            mask = np.array([1, 1, 0, 0, 0, 0], dtype=np.float32)
            self.d2cfr_buffer.add(
                np.array([float(state), 0.0, 1.0], dtype=np.float32),
                np.array([0.4, 0.1, 0, 0, 0, 0], dtype=np.float32),
                0.2,
                np.array([0.2, -0.1, 0, 0, 0, 0], dtype=np.float32),
                mask,
                iteration,
            )

        def train_d2cfr_advantage_network_multi(self):
            self.training_calls += 1

    agent = FakeAgent()

    network, batch, metadata = signal_probe.collect_live_probe(
        agent,
        roots=[1, 2, 3],
        iteration=5,
        traversing_player=0,
        sample_count=2,
        seed=11,
        device="cpu",
        state_factory=lambda root: root,
    )

    assert network is agent.advantage_net
    assert batch.states.shape == (2, 3)
    assert metadata["traversals"] == 3
    assert metadata["buffer_size"] == 3
    assert agent.prepared == [(5, 0)]
    assert agent.training_calls == 0


def test_signal_probe_script_can_run_from_tools_directory():
    script = Path(__file__).resolve().parents[1] / "tools" / "d2cfr_signal_probe.py"

    result = subprocess.run(
        [sys.executable, str(script), "--help"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    assert "--checkpoint" in result.stdout
