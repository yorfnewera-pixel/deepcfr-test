"""Регрессии подключения HU current-policy self-play к training loop."""
from types import SimpleNamespace

import torch

from src.training import train as train_mod


class _HuAgent:
    """Минимальный агент: координатор в этих тестах заменён mock-объектом."""

    def __init__(self):
        self.iteration_count = 0
        self.num_players = 2
        self.num_trainable_players = 2
        self.device = "cpu"
        self.optimizer = SimpleNamespace(param_groups=[{"lr": 1e-4}])

    def reset_traversal_stats(self):
        pass

    def get_traversal_stats(self):
        return {
            "nodes": 0,
            "terminal_nodes": 0,
            "max_depth": 0,
            "recorded_nodes": 0,
            "buffer_skip_ratio": 0.0,
            "attempted": 0,
            "successful": 0,
            "failed": 0,
            "cancelled_samples": 0,
            "depth_limit_hits": 0,
        }


class _RecordingCoordinator:
    def __init__(self, events):
        self.events = events
        self.training_losses = ([0.0, 0.0], [0.0])

    def run_iteration(self, *, iteration, traversals_per_player, new_initial_state, **_kwargs):
        self.events.append(("run", iteration, traversals_per_player))
        assert new_initial_state(0, 0) == (0, 6)


def _cfg(key, default=None):
    values = {
        "checkpoint_save_every": 1000,
        "checkpoint_keep_every": 50000,
        "traversal_single_thread": True,
        "training_torch_threads": None,
        "advantage_buffer_reservoir": False,
        "clear_strategy_buffer_each_iteration": False,
    }
    return values.get(key, default)


def test_hu_training_uses_coordinator_without_opponent_pool(monkeypatch, tmp_path, capsys):
    events = []
    agent = _HuAgent()
    monkeypatch.setattr(train_mod, "DeepCFRAgent", lambda **_kwargs: agent)
    monkeypatch.setattr(train_mod, "cfg_get", _cfg)
    monkeypatch.setattr(train_mod, "_create_writer", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        train_mod,
        "_prepare_hu_current_policy_iteration",
        lambda _agent: events.append("prepare"),
    )
    monkeypatch.setattr(
        train_mod,
        "_create_hu_current_policy_coordinator",
        lambda _agent: _RecordingCoordinator(events),
    )
    monkeypatch.setattr(train_mod, "_new_hand", lambda _players, seed: (seed % 2, seed))
    monkeypatch.setattr(
        train_mod,
        "OpponentPoolSchedule",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("opponent pool запрещён в HU")),
    )
    monkeypatch.setattr(
        train_mod,
        "_configure_strategy_opponent_pool",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("opponent pool запрещён в HU")),
    )

    result = train_mod.train_self_play_multi(
        num_iterations=1,
        traversals_per_iteration=3,
        evaluate_every=0,
        save_dir=tmp_path,
        num_players=2,
        trainable_players=2,
        hu_current_policy_self_play=True,
    )

    assert result is agent
    assert events == ["prepare", ("run", 1, 3)]
    assert "HU current-policy self-play" in capsys.readouterr().out


def test_legacy_training_does_not_use_hu_coordinator(monkeypatch, tmp_path):
    calls = []
    agent = _HuAgent()
    agent.num_trainable_players = 1
    agent.prepare_iteration = lambda *_args, **_kwargs: calls.append("legacy_prepare")
    agent.record_traversal_attempt = lambda: None
    agent.record_traversal_success = lambda: None
    agent.record_traversal_failure = lambda _error: None
    agent.cfr_traverse_multi = lambda *_args, **_kwargs: calls.append("legacy_traverse")
    agent.train_advantage_network_multi = lambda: 0.0
    agent.train_strategy_network = lambda: 0.0
    agent.advantage_buffer = []
    agent.strategy_buffer = []
    agent.strategy_net = SimpleNamespace(state_dict=lambda: {"weight": torch.tensor([1.0])})
    monkeypatch.setattr(train_mod, "DeepCFRAgent", lambda **_kwargs: agent)
    monkeypatch.setattr(train_mod, "cfg_get", _cfg)
    monkeypatch.setattr(train_mod, "_create_writer", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(train_mod, "_new_hand", lambda *_args: None)
    monkeypatch.setattr(train_mod, "_log_multi_cfr_diagnostics", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(
        train_mod,
        "_create_hu_current_policy_coordinator",
        lambda _agent: (_ for _ in ()).throw(AssertionError("HU coordinator вызван в legacy")),
    )
    monkeypatch.setattr(
        train_mod,
        "_configure_strategy_opponent_pool",
        lambda *_args, **_kwargs: calls.append("opponent_pool") or [],
    )

    train_mod.train_self_play_multi(
        num_iterations=1,
        traversals_per_iteration=1,
        evaluate_every=0,
        save_dir=tmp_path,
        num_players=2,
        hu_current_policy_self_play=False,
    )

    assert calls == ["opponent_pool", "legacy_prepare", "legacy_traverse"]
