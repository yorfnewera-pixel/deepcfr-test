import numpy as np
import pokers as pkrs
import pytest

from src.core.action_space import ActionSlot, legal_action_mask, resolve_action
from src.core.model import encode_state


def _initial_hu_state() -> pkrs.State:
    return pkrs.State.from_seed(
        n_players=2,
        button=0,
        sb=1.0,
        bb=2.0,
        stake=30.0,
        seed=17,
    )


def _apply_slot(state: pkrs.State, slot: ActionSlot) -> pkrs.State:
    resolved_action = resolve_action(slot, state)
    next_state = state.apply_action(resolved_action.action)
    assert next_state.status == pkrs.StateStatus.Ok
    return next_state


def _flop_raise_vs_turn_raise() -> tuple[pkrs.State, pkrs.State]:
    turn_raise = _initial_hu_state()
    for slot in (
        ActionSlot.CALL,
        ActionSlot.CALL,
        ActionSlot.CHECK,
        ActionSlot.CHECK,
        ActionSlot.CHECK,
        ActionSlot.RAISE_HALF_POT,
        ActionSlot.CALL,
        ActionSlot.CHECK,
    ):
        turn_raise = _apply_slot(turn_raise, slot)

    flop_raise = _initial_hu_state()
    for slot in (
        ActionSlot.CALL,
        ActionSlot.CALL,
        ActionSlot.CHECK,
        ActionSlot.RAISE_HALF_POT,
        ActionSlot.CALL,
        ActionSlot.CHECK,
        ActionSlot.CHECK,
        ActionSlot.CHECK,
    ):
        flop_raise = _apply_slot(flop_raise, slot)

    return flop_raise, turn_raise


def _turn_aggressor_player_zero_vs_one() -> tuple[pkrs.State, pkrs.State]:
    player_zero_aggressor = _initial_hu_state()
    for slot in (
        ActionSlot.CALL,
        ActionSlot.CALL,
        ActionSlot.CHECK,
        ActionSlot.CHECK,
        ActionSlot.CHECK,
        ActionSlot.RAISE_HALF_POT,
        ActionSlot.CALL,
    ):
        player_zero_aggressor = _apply_slot(player_zero_aggressor, slot)

    player_one_aggressor = _initial_hu_state()
    for slot in (
        ActionSlot.CALL,
        ActionSlot.CALL,
        ActionSlot.CHECK,
        ActionSlot.CHECK,
        ActionSlot.RAISE_HALF_POT,
        ActionSlot.CALL,
    ):
        player_one_aggressor = _apply_slot(player_one_aggressor, slot)

    return player_zero_aggressor, player_one_aggressor


def _assert_current_alias(first: pkrs.State, second: pkrs.State) -> None:
    assert first.stage == pkrs.Stage.River
    assert second.stage == pkrs.Stage.River
    assert np.array_equal(legal_action_mask(first), legal_action_mask(second))

    first_encoding = encode_state(first, player_id=0)
    second_encoding = encode_state(second, player_id=0)
    assert first_encoding.shape == (133,)
    assert second_encoding.shape == (133,)
    assert np.array_equal(first_encoding, second_encoding)


@pytest.mark.parametrize(
    "state_factory",
    (_flop_raise_vs_turn_raise, _turn_aggressor_player_zero_vs_one),
    ids=("street-of-aggression", "turn-aggressor"),
)
def test_current_encoder_reproduces_documented_hu_aliases(state_factory) -> None:
    first, second = state_factory()

    _assert_current_alias(first, second)


@pytest.mark.xfail(
    strict=True,
    reason="history_summary_v3 ещё не реализован: разные observable betting lines сливаются в один tensor",
)
@pytest.mark.parametrize(
    "state_factory",
    (_flop_raise_vs_turn_raise, _turn_aggressor_player_zero_vs_one),
    ids=("street-of-aggression", "turn-aggressor"),
)
def test_history_summary_must_separate_documented_hu_aliases(state_factory) -> None:
    first, second = state_factory()

    assert not np.array_equal(encode_state(first, player_id=0), encode_state(second, player_id=0))
