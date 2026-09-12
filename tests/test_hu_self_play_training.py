"""Регрессии подключения HU current-policy self-play к training loop."""
from types import SimpleNamespace

import numpy as np
import pokers as pkrs
import torch
import pytest

from src.core.action_space import ActionSlot, resolve_action
from src.core.deep_cfr import DeepCFRAgent
from src.core.model import MONOLITHIC_ARCHITECTURE
from src.core.traversal_errors import TraversalFailure, TraversalFailureContext
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

    def record_traversal_attempt(self):
        pass

    def record_traversal_success(self):
        pass

    def record_traversal_failure(self, _error):
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


class _ResumeCoordinator:
    def __init__(self, events):
        self.events = events
        self.training_losses = ([0.0, 0.0], [0.0])

    def run_iteration(self, *, iteration, traversals_per_player, **_kwargs):
        self.events.append(("run", iteration, traversals_per_player))


class _SchedulingCoordinator:
    def __init__(self, events):
        self.events = events
        self.training_losses = ([0.0, 0.0], [])
        self.training_timings = ([0.0, 0.0], [0.0])
        self.d2cfr_component_losses = [{}, {}]

    def run_iteration(self, *, iteration, traversals_per_player, train_strategy_due, **_kwargs):
        self.events.append(("run", iteration, traversals_per_player, train_strategy_due))


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
    monkeypatch.setattr(train_mod, "_save_hu_checkpoint", lambda _agent, path, seed=None: path)
    monkeypatch.setattr(train_mod, "_save_light_checkpoint", lambda _agent, path, seed=None: path)
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
    output = capsys.readouterr().out
    assert "HU current-policy self-play" in output
    assert "обе фазы обходов P0/P1 на frozen snapshots -> обучение advantage P0/P1 -> обучение shared strategy" in output


def test_hu_strategy_schedule_runs_periodically_and_once_at_finish(monkeypatch, tmp_path):
    events = []
    final_strategy_steps = []
    agent = _HuAgent()
    agent.strategy_train_every = 2
    agent.strategy_train_steps = 50
    agent.strategy_final_train_steps = 1000
    agent.train_strategy_network = lambda: final_strategy_steps.append(
        agent.strategy_train_steps
    ) or 0.25
    monkeypatch.setattr(train_mod, "DeepCFRAgent", lambda **_kwargs: agent)
    monkeypatch.setattr(train_mod, "cfg_get", _cfg)
    monkeypatch.setattr(train_mod, "_create_writer", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(train_mod, "_prepare_hu_current_policy_iteration", lambda _agent: None)
    monkeypatch.setattr(
        train_mod,
        "_create_hu_current_policy_coordinator",
        lambda _agent: _SchedulingCoordinator(events),
    )
    monkeypatch.setattr(train_mod, "_save_hu_checkpoint", lambda _agent, path, seed=None: path)
    monkeypatch.setattr(train_mod, "_save_light_checkpoint", lambda _agent, path, seed=None: path)

    train_mod.train_self_play_multi(
        num_iterations=3,
        traversals_per_iteration=1,
        evaluate_every=0,
        save_dir=tmp_path,
        num_players=2,
        trainable_players=2,
        hu_current_policy_self_play=True,
    )

    assert events == [
        ("run", 1, 1, False),
        ("run", 2, 1, True),
        ("run", 3, 1, False),
    ]
    assert final_strategy_steps == [1000]
    assert agent.strategy_train_steps == 50


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


def test_generic_multiplayer_prepares_replay_once_before_all_traversers(monkeypatch, tmp_path):
    prepared = []
    traversed = []
    agent = _HuAgent()
    agent.num_players = 3
    agent.num_trainable_players = 2
    agent.prepare_iteration = lambda iteration, traversing_player: prepared.append(
        (iteration, traversing_player)
    )
    agent.record_traversal_attempt = lambda: None
    agent.record_traversal_success = lambda: None
    agent.record_traversal_failure = lambda _error: None
    agent.cfr_traverse_multi = lambda _state, _iteration, traversing_player, **_kwargs: traversed.append(
        traversing_player
    )
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
    monkeypatch.setattr(train_mod, "_configure_strategy_opponent_pool", lambda *_args, **_kwargs: [])

    train_mod.train_self_play_multi(
        num_iterations=1,
        traversals_per_iteration=1,
        evaluate_every=0,
        save_dir=tmp_path,
        num_players=3,
        trainable_players=2,
        hu_current_policy_self_play=False,
    )

    assert prepared == [(1, None)]
    assert traversed == [0, 1]


def test_hu_failure_handler_ispolzuet_skip_limit_i_diagnostics(monkeypatch):
    diagnostics = []
    agent = _HuAgent()
    agent.record_traversal_failure = lambda error: diagnostics.append(error.context)
    monkeypatch.setattr(
        train_mod,
        "cfg_training_error_mode",
        lambda: "skip_traversal",
    )
    monkeypatch.setattr(
        train_mod,
        "cfg_training_max_failed_traversals_per_iteration",
        lambda: 2,
    )
    handler = train_mod._create_hu_traversal_failure_handler(agent)
    error = TraversalFailure(
        TraversalFailureContext(1, 0, 0, 1, 3, "ошибка HU", ("slot 1",))
    )

    assert handler(error) is True
    with pytest.raises(RuntimeError, match="training_max_failed_traversals_per_iteration"):
        handler(error)

    assert [context.reason for context in diagnostics] == ["ошибка HU", "ошибка HU"]


@pytest.mark.parametrize(
    ("regret_norm", "regret_clip"),
    (("none", None), ("pot_stack", 50.0)),
)
def test_hu_adapter_primenyaet_obshchuyu_normalizatsiyu_regretov_k_p0_i_p1(
    regret_norm,
    regret_clip,
):
    agent = DeepCFRAgent(
        player_id=0,
        num_players=2,
        device="cpu",
        hidden_size=8,
        network_architecture=MONOLITHIC_ARCHITECTURE,
    )
    agent.advantage_regret_norm = regret_norm
    agent.advantage_regret_clip = regret_clip
    agent.advantage_reward_scale = 200.0
    adapter = train_mod._create_hu_current_policy_coordinator(agent).adapter
    root = pkrs.State.from_seed(
        n_players=2,
        button=0,
        sb=1.0,
        bb=2.0,
        stake=200.0,
        seed=17,
    )
    p0_state = root.apply_action(resolve_action(ActionSlot.CALL, root).action)
    regrets = np.array([200.0, -100.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
    mask = np.array([1.0, 1.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)

    assert {int(root.current_player), int(p0_state.current_player)} == {0, 1}
    for state in (root, p0_state):
        expected = regrets.copy()
        if regret_norm == "pot_stack":
            player = state.players_state[int(state.current_player)]
            expected /= max(float(state.pot) + float(player.stake), 1.0)
        if regret_clip is not None:
            expected = np.clip(expected, -regret_clip, regret_clip)
        expected /= 200.0

        assert np.allclose(adapter.normalize_regrets(state, regrets, mask), expected)


def test_hu_resume_continues_from_next_iteration_and_writes_periodic_and_final_checkpoint(monkeypatch, tmp_path):
    events = []
    saved_paths = []
    agent = _HuAgent()
    monkeypatch.setattr(train_mod, "DeepCFRAgent", lambda **_kwargs: agent)
    monkeypatch.setattr(
        train_mod,
        "cfg_get",
        lambda key, default=None: 1 if key == "hu_checkpoint_save_every" else _cfg(key, default),
    )
    monkeypatch.setattr(train_mod, "_create_writer", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        train_mod,
        "_create_hu_current_policy_coordinator",
        lambda _agent: _ResumeCoordinator(events),
    )
    monkeypatch.setattr(
        train_mod,
        "_prepare_hu_current_policy_iteration",
        lambda _agent: events.append("prepare"),
    )

    def load_checkpoint(loaded_agent, path):
        assert str(path).endswith("resume.pt")
        events.append("load")
        loaded_agent.iteration_count = 4
        return {"seed": 9}

    monkeypatch.setattr(train_mod, "_load_hu_checkpoint", load_checkpoint)
    monkeypatch.setattr(
        train_mod,
        "_save_hu_checkpoint",
        lambda _agent, path, seed=None: saved_paths.append((path.name, seed)) or path,
    )
    monkeypatch.setattr(
        train_mod,
        "_save_light_checkpoint",
        lambda _agent, path, seed=None: saved_paths.append((path.name, seed)) or path,
    )

    train_mod.train_self_play_multi(
        num_iterations=1,
        traversals_per_iteration=3,
        evaluate_every=0,
        save_dir=tmp_path,
        num_players=2,
        trainable_players=2,
        initial_checkpoint=tmp_path / "resume.pt",
        hu_current_policy_self_play=True,
    )

    assert events == ["load", "prepare", ("run", 5, 3)]
    assert saved_paths == [
        ("hu_checkpoint_iter_5.pt", 9),
        ("light_checkpoint_iter_5.pt", 9),
        ("hu_checkpoint_final.pt", 9),
        ("light_checkpoint_final.pt", 9),
    ]


def test_hu_writes_full_and_light_checkpoints_on_hu_schedule_and_at_finish(monkeypatch, tmp_path):
    """Ломается, если HU не сохраняет light checkpoint вместе с full по своему интервалу."""
    saved_paths = []
    agent = _HuAgent()
    monkeypatch.setattr(train_mod, "DeepCFRAgent", lambda **_kwargs: agent)
    monkeypatch.setattr(
        train_mod,
        "cfg_get",
        lambda key, default=None: {
            "hu_checkpoint_save_every": 1,
            "checkpoint_save_every": 2,
        }.get(key, _cfg(key, default)),
    )
    monkeypatch.setattr(train_mod, "_create_writer", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        train_mod,
        "_create_hu_current_policy_coordinator",
        lambda _agent: _ResumeCoordinator([]),
    )
    monkeypatch.setattr(train_mod, "_prepare_hu_current_policy_iteration", lambda _agent: None)
    monkeypatch.setattr(
        train_mod,
        "_save_hu_checkpoint",
        lambda _agent, path, seed=None: saved_paths.append(("full", path.name, seed)) or path,
    )
    monkeypatch.setattr(
        train_mod,
        "_save_light_checkpoint",
        lambda _agent, path, seed=None: saved_paths.append(("light", path.name, seed)) or path,
        raising=False,
    )

    train_mod.train_self_play_multi(
        num_iterations=1,
        traversals_per_iteration=1,
        evaluate_every=0,
        save_dir=tmp_path,
        num_players=2,
        trainable_players=2,
        seed=17,
        hu_current_policy_self_play=True,
    )

    assert saved_paths == [
        ("full", "hu_checkpoint_iter_1.pt", 17),
        ("light", "light_checkpoint_iter_1.pt", 17),
        ("full", "hu_checkpoint_final.pt", 17),
        ("light", "light_checkpoint_final.pt", 17),
    ]
