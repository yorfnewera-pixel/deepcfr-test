import numpy as np
import pokers as pkrs
import pytest
import torch

from src.core.action_space import NUM_ACTIONS
from src.core.buffers import StrategyBuffer
from src.core.deep_cfr import DeepCFRAgent
from src.core.traversal_errors import TraversalFailure, TraversalFailureContext
from src.utils import settings


class _TerminalState:
    final_state = True
    status = pkrs.StateStatus.Ok

    def __init__(self, reward=0.0):
        self.players_state = [type("PlayerState", (), {"reward": reward})() for _ in range(2)]


class _NonterminalState:
    final_state = False
    current_player = 0

    def __init__(self, apply_action):
        self._apply_action = apply_action

    def apply_action(self, action):
        return self._apply_action(action)


@pytest.fixture(autouse=True)
def _restore_strict_checking():
    original = settings.is_strict_checking()
    settings.set_strict_checking(False)
    yield
    settings.set_strict_checking(original)


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


def _agent_and_nonterminal_state(apply_action):
    agent = DeepCFRAgent(player_id=0, num_players=2, memory_size=2, device="cpu")
    return agent, _NonterminalState(apply_action)


def _configure_single_legal_action(monkeypatch, agent):
    monkeypatch.setattr(
        agent,
        "get_legal_action_mask",
        lambda _state: np.array([1, 0, 0, 0, 0, 0], dtype=np.float32),
    )
    monkeypatch.setattr(agent, "action_type_to_pokers_action", lambda _slot, _state: "call")
    monkeypatch.setattr(
        agent,
        "_encode_state",
        lambda _state, _player: np.zeros(agent.input_size, dtype=np.float32),
    )


def test_apply_action_failure_raises_when_strict_mode_changes_after_import(monkeypatch):
    agent, state = _agent_and_nonterminal_state(
        lambda _action: (_ for _ in ()).throw(RuntimeError("engine failure"))
    )
    _configure_single_legal_action(monkeypatch, agent)

    settings.set_strict_checking(True)

    with pytest.raises(TraversalFailure, match="apply_action") as error:
        agent.cfr_traverse_multi(state, iteration=1, traversing_player=0)

    assert isinstance(error.value.__cause__, RuntimeError)


def test_apply_action_failure_is_not_silenced_when_strict_mode_is_disabled(monkeypatch):
    agent, state = _agent_and_nonterminal_state(
        lambda _action: (_ for _ in ()).throw(RuntimeError("engine failure"))
    )
    _configure_single_legal_action(monkeypatch, agent)

    settings.set_strict_checking(False)

    with pytest.raises(TraversalFailure, match="apply_action"):
        agent.cfr_traverse_multi(state, iteration=1, traversing_player=0)


def test_depth_limit_is_failure_not_zero_reward():
    agent, state = _agent_and_nonterminal_state(lambda _action: _TerminalState())

    with pytest.raises(TraversalFailure, match="depth") as error:
        agent.cfr_traverse_multi(state, iteration=1, traversing_player=0, depth=201)

    assert error.value.context.acting_player == 0


def test_real_terminal_zero_reward_is_returned():
    agent = DeepCFRAgent(player_id=0, num_players=2, memory_size=2, device="cpu")
    state = pkrs.State.from_seed(
        n_players=2,
        button=0,
        sb=1.0,
        bb=2.0,
        stake=2.0,
        seed=2,
    ).apply_action(pkrs.Action(pkrs.ActionEnum.Call, 0.0))

    assert state.final_state
    assert state.players_state[0].reward == 0.0
    assert agent.cfr_traverse_multi(state, iteration=1, traversing_player=0) == 0.0


def test_non_ok_state_status_raises_traversal_failure(monkeypatch):
    agent, state = _agent_and_nonterminal_state(
        lambda _action: type("InvalidState", (), {"status": pkrs.StateStatus.IllegalAction})()
    )
    _configure_single_legal_action(monkeypatch, agent)

    with pytest.raises(TraversalFailure, match="status") as error:
        agent.cfr_traverse_multi(
            state,
            iteration=1,
            traversing_player=0,
            traversal_index=4,
        )

    diagnostic = agent.record_traversal_failure(error.value)
    assert diagnostic["traversal_index"] == 4
    assert diagnostic["action"] == "slot 0: 'call'"
    assert diagnostic["status"] == "StateStatus.IllegalAction"


def test_missing_state_status_after_apply_action_raises_traversal_failure(monkeypatch):
    agent, state = _agent_and_nonterminal_state(lambda _action: None)
    _configure_single_legal_action(monkeypatch, agent)

    with pytest.raises(TraversalFailure, match="status") as error:
        agent.cfr_traverse_multi(state, iteration=1, traversing_player=0)

    assert isinstance(error.value.__cause__, AttributeError)


def test_empty_legal_slots_on_nonterminal_state_raise_traversal_failure(monkeypatch):
    agent, state = _agent_and_nonterminal_state(lambda _action: _TerminalState())
    monkeypatch.setattr(
        agent,
        "get_legal_action_mask",
        lambda _state: np.zeros(NUM_ACTIONS, dtype=np.float32),
    )

    with pytest.raises(TraversalFailure, match="допустим"):
        agent.cfr_traverse_multi(state, iteration=1, traversing_player=0)


def test_nan_advantage_policy_raises_traversal_failure_before_sampling(monkeypatch):
    agent, state = _agent_and_nonterminal_state(lambda _action: _TerminalState())
    state.current_player = 1
    _configure_single_legal_action(monkeypatch, agent)
    monkeypatch.setattr(
        agent.advantage_net,
        "forward",
        lambda state_t: torch.full((state_t.shape[0], NUM_ACTIONS), float("nan"), device=state_t.device),
    )

    with pytest.raises(TraversalFailure, match="policy") as error:
        agent.cfr_traverse_multi(
            state,
            iteration=1,
            traversing_player=0,
            traversal_index=2,
        )

    assert error.value.context.depth == 0


def test_nan_strategy_policy_raises_traversal_failure_before_sampling(monkeypatch):
    agent, state = _agent_and_nonterminal_state(lambda _action: _TerminalState())
    state.current_player = 1
    _configure_single_legal_action(monkeypatch, agent)
    agent._opponent_strategy_nets[1] = agent.strategy_net
    monkeypatch.setattr(
        agent.strategy_net,
        "forward",
        lambda state_t: torch.full((state_t.shape[0], NUM_ACTIONS), float("nan"), device=state_t.device),
    )

    with pytest.raises(TraversalFailure, match="policy") as error:
        agent.cfr_traverse_multi(
            state,
            iteration=1,
            traversing_player=0,
            traversal_index=3,
        )

    assert error.value.context.depth == 0
    diagnostic = agent.record_traversal_failure(error.value)
    assert diagnostic["iteration"] == 1
    assert diagnostic["traversal_index"] == 3
    assert diagnostic["mask"] == [1.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    assert all(np.isnan(value) for value in diagnostic["policy"])


def test_nan_runtime_strategy_policy_raises_traversal_failure_before_sampling(monkeypatch):
    agent = DeepCFRAgent(player_id=0, num_players=2, device="cpu")
    state = _state()
    monkeypatch.setattr(
        agent.strategy_net,
        "forward",
        lambda state_t: torch.full((state_t.shape[0], NUM_ACTIONS), float("nan"), device=state_t.device),
    )

    with pytest.raises(TraversalFailure, match="policy"):
        agent.choose_action(state, player_id=int(state.current_player))


def test_late_failed_traverser_action_discards_collector_without_subset_sample(monkeypatch):
    actions = []

    def _apply_action(action):
        actions.append(action)
        if action == "late_failure":
            raise RuntimeError("late engine failure")
        return _TerminalState(0.0)

    agent, state = _agent_and_nonterminal_state(_apply_action)
    monkeypatch.setattr(
        agent,
        "get_legal_action_mask",
        lambda _state: np.array([1, 1, 0, 0, 0, 0], dtype=np.float32),
    )
    monkeypatch.setattr(
        agent,
        "action_type_to_pokers_action",
        lambda slot, _state: "valid" if slot == 0 else "late_failure",
    )
    monkeypatch.setattr(
        agent,
        "_encode_state",
        lambda _state, _player: np.zeros(agent.input_size, dtype=np.float32),
    )

    with pytest.raises(TraversalFailure, match="slot 1"):
        agent.cfr_traverse_multi(state, iteration=1, traversing_player=0)

    assert actions == ["valid", "late_failure"]
    assert len(agent.advantage_buffer) == 0
    assert len(agent.strategy_buffer) == 0


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


def test_oversized_iteration_does_not_mutate_full_reservoir(monkeypatch):
    agent = _agent_with_small_reservoir()
    advantage_before = _buffer_snapshot(agent.advantage_buffer)
    strategy_before = _buffer_snapshot(agent.strategy_buffer)

    def _record_oversized_iteration(_state, iteration, _traversing_player, _depth):
        encoded, regrets, strategy, mask = _sample(agent)
        agent._record_advantage_sample(encoded, regrets, mask, iteration)
        agent._record_strategy_sample(encoded, strategy, mask, iteration)
        return 1.5

    monkeypatch.setattr(agent, "_cfr_traverse_multi", _record_oversized_iteration)
    monkeypatch.setattr(np.random, "randint", lambda *_args, **_kwargs: 0)

    with pytest.raises(TraversalFailure):
        agent.cfr_traverse_multi(_state(), iteration=10**400, traversing_player=0)

    assert _buffer_snapshot(agent.advantage_buffer) == advantage_before
    assert _buffer_snapshot(agent.strategy_buffer) == strategy_before


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


@pytest.mark.parametrize(
    ("strategy", "reason"),
    [
        (
            np.array([-0.1, 1.1, 0, 0, 0, 0], dtype=np.float32),
            "отрицательные вероятности",
        ),
        (
            np.array([0.5, 0.4, 0.1, 0, 0, 0], dtype=np.float32),
            "недопустимого действия",
        ),
        (
            np.array([0.4, 0.4, 0, 0, 0, 0], dtype=np.float32),
            "Сумма strategy",
        ),
    ],
    ids=["отрицательная_вероятность", "масса_на_недопустимом", "ненормированная_масса"],
)
def test_invalid_strategy_sample_does_not_change_buffer(strategy, reason):
    agent = _agent_with_small_reservoir()
    state, _regrets, _strategy, mask = _sample(agent)
    before = _buffer_snapshot(agent.strategy_buffer)

    with pytest.raises(TraversalFailure, match=reason):
        agent._record_strategy_sample(state, strategy, mask, 1)

    assert _buffer_snapshot(agent.strategy_buffer) == before
