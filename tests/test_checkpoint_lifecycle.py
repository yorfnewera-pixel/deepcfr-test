from pathlib import Path

import pytest
import torch

from src.training import train as train_mod


class TinyAgent:
    def __init__(self):
        self.iteration_count = 0

    def _build_checkpoint(self, seed=None):
        return {"iteration": self.iteration_count, "seed": seed}


class LightCheckpointAgent(TinyAgent):
    def build_light_checkpoint(self, seed=None):
        return {"iteration": self.iteration_count, "seed": seed, "strategy_net": {}}


@pytest.mark.parametrize(
    ("iteration", "expected"),
    [(1, False), (999, False), (1000, True), (1001, False), (2000, True)],
)
def test_checkpoint_save_due_every_thousand_iterations(iteration, expected):
    assert train_mod._checkpoint_save_due(iteration, every=1000) is expected


def test_iteration_checkpoints_are_unique_and_rotated(tmp_path):
    agent = TinyAgent()
    for iteration in (1, 2, 3):
        agent.iteration_count = iteration
        train_mod._save_iteration_checkpoint(agent, tmp_path, iteration, "checkpoint_iter_", seed=11)

    assert torch.load(tmp_path / "checkpoint_iter_1.pt", weights_only=False)["iteration"] == 1
    assert torch.load(tmp_path / "checkpoint_iter_3.pt", weights_only=False)["iteration"] == 3


def test_light_checkpoint_contains_only_strategy_artifact(tmp_path):
    agent = LightCheckpointAgent()
    agent.iteration_count = 4

    train_mod._save_iteration_checkpoint(agent, tmp_path, 4, seed=11)
    path = train_mod._save_iteration_light_checkpoint(agent, tmp_path, 4, seed=11)

    assert path.name == "light_checkpoint_iter_4.pt"
    assert torch.load(path, weights_only=False) == {"iteration": 4, "seed": 11, "strategy_net": {}}


def test_atomic_save_keeps_previous_file_after_failure(tmp_path, monkeypatch):
    target = tmp_path / "checkpoint.pt"
    torch.save({"old": True}, target)

    def failing_save(_payload, path):
        path.write_bytes(b"partial")
        raise OSError("disk full")

    monkeypatch.setattr(train_mod.torch, "save", failing_save)
    with pytest.raises(OSError, match="disk full"):
        train_mod._atomic_torch_save({"new": True}, target)

    assert torch.load(target, weights_only=False) == {"old": True}


@pytest.mark.parametrize(
    ("iteration", "expected"),
    [
        (1001, [1000, 1000, 1000, 1000, 1000]),
        (2001, [1000, 1000, 1000, 1000, 2000]),
        (3001, [1000, 1000, 1000, 2000, 3000]),
        (6001, [2000, 3000, 4000, 5000, 6000]),
    ],
)
def test_opponent_pool_uses_thousand_step_warmup(tmp_path, iteration, expected):
    for checkpoint_iteration in range(1000, 7001, 1000):
        (tmp_path / f"multi_checkpoint_iter_{checkpoint_iteration}.pt").touch()

    paths = train_mod._opponent_checkpoint_paths(
        iteration, tmp_path, 5, checkpoint_every=1000, historical_every=50000
    )

    assert sorted(train_mod._checkpoint_iteration(path) for path in paths) == expected


def test_opponent_pool_uses_last_slot_and_previous_ten_slots(tmp_path, monkeypatch):
    for checkpoint_iteration in range(1000, 17001, 1000):
        (tmp_path / f"multi_checkpoint_iter_{checkpoint_iteration}.pt").touch()
    monkeypatch.setattr(train_mod.random, "sample", lambda candidates, count: candidates[-count:])

    paths = train_mod._opponent_checkpoint_paths(17001, tmp_path, 5, 1000, 50000)

    assert sorted(train_mod._checkpoint_iteration(path) for path in paths) == [13000, 14000, 15000, 16000, 17000]


def test_opponent_pool_uses_mature_historical_checkpoint(tmp_path, monkeypatch):
    for checkpoint_iteration in range(1000, 100001, 1000):
        (tmp_path / f"multi_checkpoint_iter_{checkpoint_iteration}.pt").touch()
    monkeypatch.setattr(train_mod.random, "sample", lambda candidates, count: candidates[-count:])

    paths = train_mod._opponent_checkpoint_paths(100001, tmp_path, 5, 1000, 50000)

    assert sorted(train_mod._checkpoint_iteration(path) for path in paths) == [50000, 97000, 98000, 99000, 100000]


def test_permanent_legacy_pool_keeps_historical_checkpoint_with_four_own_slots(tmp_path, monkeypatch):
    for checkpoint_iteration in range(1000, 100001, 1000):
        (tmp_path / f"multi_checkpoint_iter_{checkpoint_iteration}.pt").touch()
    monkeypatch.setattr(train_mod.random, "sample", lambda candidates, count: candidates[-count:])

    paths = train_mod._opponent_checkpoint_paths(100001, tmp_path, 4, 1000, 50000)

    assert sorted(train_mod._checkpoint_iteration(path) for path in paths) == [50000, 98000, 99000, 100000]


def test_first_historical_checkpoint_is_fixed_in_pool_before_next_milestone(tmp_path, monkeypatch):
    for checkpoint_iteration in range(1000, 51001, 1000):
        (tmp_path / f"multi_checkpoint_iter_{checkpoint_iteration}.pt").touch()
    monkeypatch.setattr(train_mod.random, "sample", lambda candidates, count: candidates[:count])

    paths = train_mod._opponent_checkpoint_paths(51001, tmp_path, 4, 1000, 50000)

    assert sorted(train_mod._checkpoint_iteration(path) for path in paths) == [41000, 42000, 50000, 51000]


def test_opponent_pool_duplicates_available_full_checkpoint_after_partial_resume(tmp_path):
    full_path = tmp_path / "multi_checkpoint_iter_1000.pt"
    full_path.touch()

    paths = train_mod._opponent_checkpoint_paths(3001, tmp_path, 5, 1000, 50000)

    assert paths == [full_path] * 5


def test_opponent_pool_uses_full_checkpoint_and_ignores_old_advantage_file(tmp_path):
    full_path = tmp_path / "multi_checkpoint_iter_1000.pt"
    old_advantage_path = tmp_path / "opponent_advantage_iter_1000.pt"
    full_path.touch()
    old_advantage_path.touch()

    paths = train_mod._opponent_checkpoint_paths(1001, tmp_path, 5, 1000, 50000)

    assert paths == [full_path] * 5


def test_opponent_pool_schedule_keeps_composition_until_next_checkpoint_slot(monkeypatch, tmp_path):
    selected_iterations = []

    def select(iteration, *_args, **_kwargs):
        selected_iterations.append(iteration)
        return [tmp_path / f"multi_checkpoint_iter_{number}.pt" for number in (7000, 6000, 5000, 4000, 1000)]

    monkeypatch.setattr(train_mod, "_opponent_checkpoint_paths", select)
    schedule = train_mod.OpponentPoolSchedule(tmp_path, checkpoint_every=1000, historical_every=50000)

    first = schedule.paths_for_iteration(7750, num_opponents=5)
    middle = schedule.paths_for_iteration(7751, num_opponents=5)
    next_slot = schedule.paths_for_iteration(8001, num_opponents=5)

    assert selected_iterations == [7750, 8001]
    assert first == middle
    assert next_slot == first


def test_full_checkpoint_retention_keeps_eleven_recent_and_two_historical(tmp_path):
    for iteration in range(1000, 100001, 1000):
        (tmp_path / f"multi_checkpoint_iter_{iteration}.pt").touch()

    train_mod._prune_full_checkpoints(tmp_path, historical_every=50000)

    assert sorted(train_mod._heavy_checkpoints(tmp_path)) == [50000, *range(89000, 101000, 1000)]


def test_checkpoint_pipeline_keeps_full_and_light_history_without_advantage_files(tmp_path):
    agent = LightCheckpointAgent()
    for iteration in (1000, 2000):
        agent.iteration_count = iteration
        train_mod._save_iteration_checkpoint(agent, tmp_path, iteration)
        train_mod._save_iteration_light_checkpoint(agent, tmp_path, iteration)

    assert sorted(train_mod._heavy_checkpoints(tmp_path)) == [1000, 2000]
    assert sorted(train_mod._checkpoint_paths(tmp_path, train_mod._LIGHT_CHECKPOINT_PREFIX)) == [1000, 2000]
    assert not list(tmp_path.glob("opponent_advantage_iter_*.pt"))


def test_opponent_pool_caches_full_checkpoint_strategy_state(tmp_path, monkeypatch):
    full_path = tmp_path / "multi_checkpoint_iter_1000.pt"
    torch.save(
        {
            "action_space_version": "six_fixed_v2",
            "strategy_net": {"weight": torch.tensor([1.0])},
            "advantage_net": {"weight": torch.tensor([2.0])},
            "advantage_buffer": ["large buffer"],
        },
        full_path,
    )
    loaded_paths = []
    original_load = train_mod.torch.load

    def tracked_load(path, *args, **kwargs):
        loaded_paths.append(Path(path))
        return original_load(path, *args, **kwargs)

    class OpponentPoolAgent:
        num_players = 6

        def clear_opponent_advantage_states(self):
            raise AssertionError("Старый advantage API больше не должен использоваться")

        def set_opponent_strategy_states(self, states, traversing_player):
            assert len(states) == 5
            assert traversing_player == 0
            assert all(state["weight"].item() == 1.0 for state in states)

    monkeypatch.setattr(train_mod.torch, "load", tracked_load)
    state_cache = {}

    train_mod._configure_opponent_pool(
        OpponentPoolAgent(), 1001, tmp_path, traversing_player=0, state_cache=state_cache
    )
    train_mod._configure_opponent_pool(
        OpponentPoolAgent(), 1001, tmp_path, traversing_player=0, state_cache=state_cache
    )

    assert loaded_paths == [full_path]


def test_full_checkpoint_strategy_loader_requires_strategy_net(tmp_path):
    full_path = tmp_path / "multi_checkpoint_iter_1000.pt"
    torch.save(
        {
            "action_space_version": "six_fixed_v2",
            "advantage_net": {"weight": torch.tensor([1.0])},
        },
        full_path,
    )

    with pytest.raises(ValueError, match="strategy_net"):
        train_mod._load_full_checkpoint_strategy_state(full_path)
