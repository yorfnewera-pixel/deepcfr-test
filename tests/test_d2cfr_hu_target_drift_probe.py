import importlib
import random

import numpy as np
import torch

from src.core.deep_cfr import DeepCFRAgent
from src.training import train as train_mod
from src.utils import config as config_mod


probe = importlib.import_module("tools.d2cfr_hu_target_drift_probe")


def test_rng_preservation_keeps_training_random_stream_unchanged():
    random.seed(31)
    np.random.seed(31)
    torch.manual_seed(31)
    expected = (random.random(), float(np.random.random()), float(torch.rand(1).item()))

    random.seed(31)
    np.random.seed(31)
    torch.manual_seed(31)
    with probe._preserve_rng_state():
        random.random()
        np.random.random()
        torch.rand(1)
    actual = (random.random(), float(np.random.random()), float(torch.rand(1).item()))

    assert actual == expected


def test_target_drift_probe_tracks_fixed_anchor_without_recording_probe_samples(tmp_path):
    config_path = tmp_path / "probe.yaml"
    config_path.write_text(
        "\n".join((
            "num_actions: 6", "num_players: 2", "num_trainable_players: 2",
            "hu_current_policy_self_play: true", "hidden_size: 8", "d2cfr_enabled: true",
            "d2cfr_mc_correction_enabled: false", "advantage_memory_size: 64", "strategy_memory_size: 64",
            "advantage_epochs: 1", "strategy_train_steps: 1",
        )) + "\n",
        encoding="utf-8",
    )
    config_mod.load_config(config_path)
    try:
        agent = DeepCFRAgent(player_id=0, num_players=2, device="cpu")
        coordinator = train_mod._create_hu_current_policy_coordinator(agent)
        report = probe.run_target_drift_probe(
            agent=agent,
            coordinator=coordinator,
            iterations=1,
            traversals_per_player=1,
            anchor_states=1,
            anchor_repeats=2,
            anchor_roots=64,
            anchor_every=1,
            seed=17,
        )
    finally:
        config_mod.load_config("config.yaml")

    assert report["anchors"]["states_collected"] == 1
    assert [item["iteration"] for item in report["snapshots"]] == [0, 1]
    assert report["snapshots"][0]["drift_from_initial"]["q_mae"] == 0.0
    assert report["anchors"]["probe_recorded_samples"] == 0
