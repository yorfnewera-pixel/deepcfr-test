import numpy as np
import pytest

from src.core.action_space import NUM_ACTIONS
from src.core.buffers import AdvantageBuffer, DuelingAdvantageBuffer, StrategyBuffer


@pytest.mark.parametrize("buffer_type,values", [
    (AdvantageBuffer, np.arange(NUM_ACTIONS, dtype=np.float32)),
    (StrategyBuffer, np.full(NUM_ACTIONS, 1.0 / NUM_ACTIONS, dtype=np.float32)),
])
def test_action_buffers_store_only_six_slot_vectors(buffer_type, values):
    buffer = buffer_type(capacity=2, state_dim=3)
    status = buffer.add(np.zeros(3, dtype=np.float32), values, np.ones(NUM_ACTIONS), 1)
    assert status == "recorded"
    states, stored_values, masks, iterations = buffer.sample()
    assert states.shape == (1, 3)
    assert stored_values.shape == (1, NUM_ACTIONS)
    assert masks.shape == (1, NUM_ACTIONS)
    assert iterations.tolist() == [1.0]


def test_action_buffers_reject_legacy_action_shape():
    buffer = AdvantageBuffer(capacity=2, state_dim=1)
    with pytest.raises(ValueError):
        buffer.add(np.zeros(1), np.zeros(4), np.ones(4), 1)


def test_strategy_buffer_evicts_oldest_sample_when_full():
    buffer = StrategyBuffer(capacity=2, state_dim=1)
    policy = np.full(NUM_ACTIONS, 1.0 / NUM_ACTIONS, dtype=np.float32)
    mask = np.ones(NUM_ACTIONS, dtype=np.float32)

    buffer.add(np.array([1], dtype=np.float32), policy, mask, 1)
    buffer.add(np.array([2], dtype=np.float32), policy, mask, 2)
    buffer.add(np.array([3], dtype=np.float32), policy, mask, 3)

    states, _, _, iterations = buffer.sample()

    assert sorted(iterations.tolist()) == [2.0, 3.0]
    assert sorted(states.reshape(-1).tolist()) == [2.0, 3.0]


def test_strategy_buffer_reservoir_can_keep_old_sample_when_full(monkeypatch):
    buffer = StrategyBuffer(capacity=2, state_dim=1, reservoir=True)
    policy = np.full(NUM_ACTIONS, 1.0 / NUM_ACTIONS, dtype=np.float32)
    mask = np.ones(NUM_ACTIONS, dtype=np.float32)
    choices = iter([0, 3])
    monkeypatch.setattr(np.random, "randint", lambda *_args: next(choices))

    buffer.add(np.array([1], dtype=np.float32), policy, mask, 1)
    buffer.add(np.array([2], dtype=np.float32), policy, mask, 2)
    first_status = buffer.add(np.array([3], dtype=np.float32), policy, mask, 3)
    second_status = buffer.add(np.array([4], dtype=np.float32), policy, mask, 4)

    states, _, _, iterations = buffer.sample()

    assert first_status == "evicted"
    assert second_status == "skipped"
    assert sorted(iterations.tolist()) == [2.0, 3.0]
    assert sorted(states.reshape(-1).tolist()) == [2.0, 3.0]


@pytest.mark.parametrize("buffer_type", (AdvantageBuffer, DuelingAdvantageBuffer))
def test_advantage_reservoir_tracks_valid_size_separately_from_total_seen(
    buffer_type, monkeypatch
):
    buffer = buffer_type(capacity=2, state_dim=1)
    mask = np.array([1, 1, 0, 0, 0, 0], dtype=np.float32)
    choices = iter([0, 3])
    monkeypatch.setattr(np.random, "randint", lambda *_args: next(choices))

    for iteration in range(1, 5):
        state = np.array([float(iteration)], dtype=np.float32)
        if isinstance(buffer, DuelingAdvantageBuffer):
            status = buffer.add(
                state,
                np.zeros(NUM_ACTIONS, dtype=np.float32),
                0.0,
                np.zeros(NUM_ACTIONS, dtype=np.float32),
                mask,
                iteration,
            )
        else:
            status = buffer.add(state, np.zeros(NUM_ACTIONS, dtype=np.float32), mask, iteration)

    assert status == "skipped"
    assert len(buffer) == 2
    assert buffer._size == 2
    assert buffer._total_seen == 4


def test_strategy_reservoir_tracks_valid_size_separately_from_total_seen(monkeypatch):
    buffer = StrategyBuffer(capacity=2, state_dim=1, reservoir=True)
    policy = np.full(NUM_ACTIONS, 1.0 / NUM_ACTIONS, dtype=np.float32)
    mask = np.ones(NUM_ACTIONS, dtype=np.float32)
    choices = iter([0, 3])
    monkeypatch.setattr(np.random, "randint", lambda *_args: next(choices))

    for iteration in range(1, 5):
        status = buffer.add(np.array([float(iteration)], dtype=np.float32), policy, mask, iteration)

    assert status == "skipped"
    assert len(buffer) == 2
    assert buffer._size == 2
    assert buffer._total_seen == 4
