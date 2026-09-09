import numpy as np
import pokers as pkrs
import pytest

from src.core.action_space import NUM_ACTIONS
from src.core.buffers import StrategyBuffer
from src.core.deep_cfr import DeepCFRAgent
from src.core.traversal_errors import TraversalFailure, TraversalFailureContext


def _state():
    return pkrs.State.from_seed(
        n_players=2,
        button=0,
        sb=1.0,
        bb=2.0,
        stake=100.0,
        seed=7,
    )


def _agent_with_small_reservoir():
    agent = DeepCFRAgent(player_id=0, num_players=2, memory_size=2, device="cpu")
    agent.strategy_buffer = StrategyBuffer(
        2,
        agent.input_size,
        NUM_ACTIONS,
        reservoir=True,
    )
    state = np.zeros(agent.input_size, dtype=np.float32)
    values = np.zeros(NUM_ACTIONS, dtype=np.float32)
    mask = np.array([1, 1, 0, 0, 0, 0], dtype=np.float32)
    policy = np.array([0.5, 0.5, 0, 0, 0, 0], dtype=np.float32)
    for _ in range(2):
        agent.advantage_buffer.add(state, values, mask, 1)
        agent.strategy_buffer.add(state, policy, mask, 1)
    return agent


def _buffer_snapshot(buffer):
    occupied = min(buffer._cur_id, buffer.capacity)
    value_array = getattr(buffer, "_regrets", getattr(buffer, "_policies", None))
    return (
        buffer._cur_id,
        getattr(buffer, "_size", len(buffer)),
        buffer._states[:occupied].copy().tobytes(),
        value_array[:occupied].copy().tobytes(),
        buffer._masks[:occupied].copy().tobytes(),
        buffer._iterations[:occupied].copy().tobytes(),
    )


def _sample(agent):
    state = np.zeros(agent.input_size, dtype=np.float32)
    mask = np.array([1, 1, 0, 0, 0, 0], dtype=np.float32)
    regrets = np.zeros(NUM_ACTIONS, dtype=np.float32)
    strategy = np.array([0.5, 0.5, 0, 0, 0, 0], dtype=np.float32)
    return state, regrets, strategy, mask


def _failure(iteration):
    return TraversalFailure(
        TraversalFailureContext(
            iteration=iteration,
            traversal_index=None,
            traversing_player=0,
            acting_player=None,
            depth=0,
            reason="Тестовый сбой traversal",
        )
    )


def test_failed_root_traversal_does_not_change_full_reservoir(monkeypatch):
    agent = _agent_with_small_reservoir()
    before = _buffer_snapshot(agent.strategy_buffer)
    advantage_before = _buffer_snapshot(agent.advantage_buffer)

    def _record_then_fail(_state, iteration, _traversing_player, _depth):
        encoded, regrets, strategy, mask = _sample(agent)
        agent._record_advantage_sample(encoded, regrets, mask, iteration)
        agent._record_strategy_sample(encoded, strategy, mask, iteration)
        raise _failure(iteration)

    monkeypatch.setattr(agent, "_cfr_traverse_multi", _record_then_fail)

    with pytest.raises(TraversalFailure):
        agent.cfr_traverse_multi(_state(), iteration=1, traversing_player=0)

    assert _buffer_snapshot(agent.strategy_buffer) == before
    assert _buffer_snapshot(agent.advantage_buffer) == advantage_before


def test_successful_root_traversal_commits_collected_samples_once(monkeypatch):
    agent = _agent_with_small_reservoir()
    agent.advantage_buffer.clear()
    agent.strategy_buffer.clear()

    def _record_then_return(_state, iteration, _traversing_player, _depth):
        encoded, regrets, strategy, mask = _sample(agent)
        agent._record_advantage_sample(encoded, regrets, mask, iteration)
        agent._record_strategy_sample(encoded, strategy, mask, iteration)
        return 1.5

    monkeypatch.setattr(agent, "_cfr_traverse_multi", _record_then_return)

    result = agent.cfr_traverse_multi(_state(), iteration=1, traversing_player=0)

    assert result == 1.5
    assert len(agent.advantage_buffer) == 1
    assert len(agent.strategy_buffer) == 1


@pytest.mark.parametrize(
    ("values", "mask"),
    [
        (np.zeros(NUM_ACTIONS - 1, dtype=np.float32), np.array([1, 1, 0, 0, 0, 0], dtype=np.float32)),
        (np.array([np.nan, 0, 0, 0, 0, 0], dtype=np.float32), np.array([1, 1, 0, 0, 0, 0], dtype=np.float32)),
        (np.zeros(NUM_ACTIONS, dtype=np.float32), np.zeros(NUM_ACTIONS, dtype=np.float32)),
    ],
    ids=["неверная_форма", "nan", "нет_допустимого_действия"],
)
def test_invalid_advantage_sample_does_not_change_buffer(monkeypatch, values, mask):
    agent = _agent_with_small_reservoir()
    before = _buffer_snapshot(agent.advantage_buffer)

    def _record_invalid(_state, iteration, _traversing_player, _depth):
        encoded, _regrets, _strategy, _mask = _sample(agent)
        agent._record_advantage_sample(encoded, values, mask, iteration)

    monkeypatch.setattr(agent, "_cfr_traverse_multi", _record_invalid)

    with pytest.raises(TraversalFailure):
        agent.cfr_traverse_multi(_state(), iteration=1, traversing_player=0)

    assert _buffer_snapshot(agent.advantage_buffer) == before
