import importlib

import numpy as np
import pytest
import torch

from src.core.model import DuelingRegretNetwork
from src.core.deep_cfr import DeepCFRAgent
from src.training import train as train_mod
from src.utils import config as config_mod
from tools import d2cfr_signal_probe as signal_probe


probe = importlib.import_module("tools.d2cfr_frozen_target_fit_probe")


def _sample(state, state_value, regrets, mask, iteration=1):
    regrets = np.asarray(regrets, dtype=np.float32)
    return (
        np.asarray(state, dtype=np.float32),
        regrets + np.float32(state_value),
        np.float32(state_value),
        regrets,
        np.asarray(mask, dtype=np.float32),
        np.float32(iteration),
    )


def test_noisy_vs_mean_fit_evaluates_both_arms_on_same_mean_reference():
    mask = np.array([0, 0, 0, 1, 1, 1], dtype=np.float32)
    mean_batch = signal_probe.build_probe_batch([
        _sample([1.0, 0.0], 0.1, [0, 0, 0, -0.1, 0.1, 0.3], mask),
        _sample([0.0, 1.0], -0.1, [0, 0, 0, 0.1, -0.1, -0.3], mask),
    ])
    noisy_batch = signal_probe.build_probe_batch([
        _sample([1.0, 0.0], 0.1, [0, 0, 0, -0.2, 0.2, 0.5], mask),
        _sample([1.0, 0.0], 0.1, [0, 0, 0, 0.0, 0.0, 0.1], mask),
        _sample([0.0, 1.0], -0.1, [0, 0, 0, 0.2, -0.2, -0.5], mask),
        _sample([0.0, 1.0], -0.1, [0, 0, 0, 0.0, 0.0, -0.1], mask),
    ])

    report = probe.run_noisy_vs_mean_fit(
        network_factory=lambda: DuelingRegretNetwork(input_size=2, hidden_size=8),
        mean_batch=mean_batch,
        noisy_batch=noisy_batch,
        steps=2,
        batch_size=2,
        learning_rate=0.01,
        state_value_loss_weight=0.5,
        seed=13,
    )

    assert report["reference_samples"] == 2
    assert report["noisy_training_samples"] == 4
    assert set(report["arms"]) == {"mean_targets", "noisy_targets"}
    assert report["arms"]["mean_targets"]["final"]["samples"] == 2
    assert "all_in" in report["arms"]["mean_targets"]["final_by_action"]
    ordering = report["arms"]["mean_targets"]["final_pairwise_ordering"]
    assert ordering["all_in_vs_raise_1pot"]["legal_pairs"] == 2
    assert 0.0 <= ordering["all_in_vs_raise_1pot"]["q_ordering_accuracy"] <= 1.0


def test_train_validation_fit_evaluates_unseen_mean_targets_separately():
    mask = np.array([0, 0, 0, 1, 1, 1], dtype=np.float32)
    train_batch = signal_probe.build_probe_batch([
        _sample([1.0, 0.0], 0.0, [0, 0, 0, -0.2, 0.1, 0.4], mask),
        _sample([0.0, 1.0], 0.0, [0, 0, 0, 0.2, -0.1, -0.4], mask),
    ])
    validation_batch = signal_probe.build_probe_batch([
        _sample([1.0, 1.0], 0.0, [0, 0, 0, -0.4, 0.3, 0.8], mask),
    ])

    report = probe.run_mean_target_train_validation_fit(
        network_factory=lambda: DuelingRegretNetwork(input_size=2, hidden_size=8),
        training_batch=train_batch,
        validation_batch=validation_batch,
        steps=0,
        batch_size=2,
        learning_rate=0.01,
        state_value_loss_weight=0.5,
        seed=13,
    )

    assert report["training_samples"] == 2
    assert report["validation_samples"] == 1
    assert report["initial"]["train"]["samples"] == 2
    assert report["initial"]["validation"]["samples"] == 1
    assert report["final"]["train"]["regret_abs_error_mean"] == pytest.approx(0.23333333)
    assert report["final"]["validation"]["regret_abs_error_mean"] == pytest.approx(0.5)


def test_checkpoint_probe_rejects_validation_that_consumes_all_frozen_states(tmp_path):
    with pytest.raises(ValueError, match="validation_states"):
        probe.run_checkpoint_probe(
            tmp_path / "missing.pt",
            config_path=tmp_path / "missing.yaml",
            output_path=tmp_path / "unused.json",
            states=4,
            validation_states=4,
        )


def test_pairwise_ordering_ignores_equal_target_values():
    targets = torch.tensor([[0, 0, 0, 0.2, 0.2, 0.4]], dtype=torch.float32)
    predictions = torch.tensor([[0, 0, 0, 0.3, 0.1, 0.2]], dtype=torch.float32)
    masks = torch.tensor([[0, 0, 0, 1, 1, 1]], dtype=torch.float32)

    report = probe.pairwise_ordering_metrics(targets, predictions, masks)

    assert report["raise_1pot_vs_raise_0.5pot"]["comparable_pairs"] == 0
    assert report["all_in_vs_raise_1pot"]["q_ordering_accuracy"] == pytest.approx(1.0)


def test_checkpoint_probe_writes_mean_and_noisy_fit_report(tmp_path):
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
        checkpoint_path = train_mod._save_hu_checkpoint(source, tmp_path / "hu.pt")
        output_path = tmp_path / "frozen-fit.json"
        report = probe.run_checkpoint_probe(
            checkpoint_path,
            config_path=config_path,
            output_path=output_path,
            states=1,
            repeats=2,
            roots=64,
            fit_steps=1,
            fit_batch_size=2,
            seed=17,
        )
    finally:
        config_mod.load_config("config.yaml")

    assert output_path.is_file()
    assert report["frozen_targets"]["states"] == 1
    assert report["fit"]["comparison"]["noisy_training_samples"] == 2
    assert report["fit"]["comparison"]["arms"]["mean_targets"]["final"]["samples"] == 1
