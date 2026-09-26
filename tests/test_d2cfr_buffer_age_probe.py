import importlib
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from src.core.buffers import DuelingAdvantageBuffer
from src.core.deep_cfr import DeepCFRAgent
from src.training import train as train_mod
from src.utils import config as config_mod


age_probe = importlib.import_module("tools.d2cfr_accumulation_probe")


class _ZeroDuelingNetwork(torch.nn.Module):
    def forward_components(self, states):
        batch_size = states.shape[0]
        return SimpleNamespace(
            action_values=torch.zeros((batch_size, 6), dtype=torch.float32),
            state_values=torch.zeros((batch_size, 1), dtype=torch.float32),
            regrets=torch.zeros((batch_size, 6), dtype=torch.float32),
        )


def test_describe_buffer_age_separates_current_and_old_labels_by_source_iteration():
    """Ловит смешивание свежих labels со старыми при поиске replay drift."""
    buffer = DuelingAdvantageBuffer(capacity=4, state_dim=2)
    mask = np.array([0, 0, 0, 1, 1, 1], dtype=np.float32)
    buffer.add(
        np.array([1.0, 0.0], dtype=np.float32),
        np.array([0, 0, 0, 0.1, 0.2, 0.6], dtype=np.float32),
        np.float32(0.1),
        np.array([0, 0, 0, 0.0, 0.1, 0.5], dtype=np.float32),
        mask,
        iteration=100,
    )
    buffer.add(
        np.array([0.0, 1.0], dtype=np.float32),
        np.array([0, 0, 0, -0.2, 0.3, 0.7], dtype=np.float32),
        np.float32(-0.1),
        np.array([0, 0, 0, -0.1, 0.4, 0.8], dtype=np.float32),
        mask,
        iteration=70,
    )

    report = age_probe.describe_buffer_age(
        _ZeroDuelingNetwork(),
        buffer,
        checkpoint_iteration=100,
        age_bucket_width=25,
    )

    fresh = report["by_age"]["age_0_24"]
    old = report["by_age"]["age_25_49"]
    assert fresh["samples"] == 1
    assert fresh["source_iteration_min"] == 100
    assert fresh["source_iteration_max"] == 100
    assert fresh["age_min"] == 0
    assert old["samples"] == 1
    assert old["source_iteration_min"] == 70
    assert old["age_max"] == 30
    assert fresh["by_action"]["all_in"]["q_abs_error_mean"] == pytest.approx(0.6)
    assert old["by_action"]["raise_1pot"]["regret_sign_flip_rate"] == pytest.approx(1.0)


def test_run_checkpoint_buffer_age_probe_reads_hu_checkpoint_without_training(tmp_path):
    config_path = tmp_path / "probe.yaml"
    config_path.write_text(
        "\n".join((
            "num_actions: 6",
            "num_players: 2",
            "num_trainable_players: 2",
            "hu_current_policy_self_play: true",
            "hidden_size: 8",
            "d2cfr_enabled: true",
            "d2cfr_mc_correction_enabled: false",
            "advantage_memory_size: 8",
            "strategy_memory_size: 8",
        )) + "\n",
        encoding="utf-8",
    )
    config_mod.load_config(config_path)
    try:
        source = DeepCFRAgent(player_id=0, num_players=2, device="cpu")
        train_mod._create_hu_current_policy_coordinator(source)
        source.iteration_count = 10
        mask = np.array([0, 0, 0, 1, 1, 1], dtype=np.float32)
        source.hu_advantage_buffers[0].add(
            np.zeros(source.input_size, dtype=np.float32),
            np.array([0, 0, 0, 0.1, 0.2, 0.4], dtype=np.float32),
            np.float32(0.1),
            np.array([0, 0, 0, 0.0, 0.1, 0.3], dtype=np.float32),
            mask,
            iteration=10,
        )
        checkpoint_path = train_mod._save_hu_checkpoint(source, tmp_path / "hu.pt")
        output_path = tmp_path / "age-report.json"
        report = age_probe.run_checkpoint_buffer_age_probe(
            checkpoint_path,
            config_path=config_path,
            output_path=output_path,
            age_bucket_width=5,
        )
    finally:
        config_mod.load_config("config.yaml")

    assert output_path.is_file()
    assert report["checkpoint_iteration"] == 10
    assert report["by_age"]["age_0_4"]["samples"] == 1


def test_parse_args_accepts_checkpoint_age_report_mode():
    args = age_probe.parse_args([
        "--checkpoint", "models/hu.pt",
        "--output", "diagnostics/age.json",
        "--age-bucket-width", "10",
        "--age-samples", "128",
    ])

    assert str(args.checkpoint) == "models\\hu.pt"
    assert str(args.output) == "diagnostics\\age.json"
    assert args.age_bucket_width == 10
    assert args.age_samples == 128
