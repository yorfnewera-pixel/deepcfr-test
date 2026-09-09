import numpy as np
import pokers as pkrs
import pytest

from src.core.action_space import legal_action_mask
from src.core.model import encode_state


def _initial_hu_state() -> pkrs.State:
    return pkrs.State.from_seed(
        n_players=2,
        button=0,
        sb=1.0,
        bb=2.0,
        stake=200.0,
        seed=17,
    )


def _apply(state: pkrs.State, action: pkrs.ActionEnum, amount: float = 0.0) -> pkrs.State:
    next_state = state.apply_action(pkrs.Action(action, amount))
    assert next_state.status == pkrs.StateStatus.Ok
    return next_state


def _preflop_raise_vs_flop_bet() -> tuple[pkrs.State, pkrs.State]:
    preflop_raise = _initial_hu_state()
    for action, amount in (
        (pkrs.ActionEnum.Raise, 8.0),
        (pkrs.ActionEnum.Call, 0.0),
        (pkrs.ActionEnum.Check, 0.0),
        (pkrs.ActionEnum.Check, 0.0),
        (pkrs.ActionEnum.Check, 0.0),
        (pkrs.ActionEnum.Check, 0.0),
    ):
        preflop_raise = _apply(preflop_raise, action, amount)

    flop_bet = _initial_hu_state()
    for action, amount in (
        (pkrs.ActionEnum.Call, 0.0),
        (pkrs.ActionEnum.Call, 0.0),
        (pkrs.ActionEnum.Raise, 8.0),
        (pkrs.ActionEnum.Call, 0.0),
        (pkrs.ActionEnum.Check, 0.0),
        (pkrs.ActionEnum.Check, 0.0),
    ):
        flop_bet = _apply(flop_bet, action, amount)

    return preflop_raise, flop_bet


def _turn_aggressor_player_zero_vs_one() -> tuple[pkrs.State, pkrs.State]:
    player_zero_aggressor = _initial_hu_state()
    for action, amount in (
        (pkrs.ActionEnum.Call, 0.0),
        (pkrs.ActionEnum.Call, 0.0),
        (pkrs.ActionEnum.Check, 0.0),
        (pkrs.ActionEnum.Check, 0.0),
        (pkrs.ActionEnum.Check, 0.0),
        (pkrs.ActionEnum.Raise, 2.0),
        (pkrs.ActionEnum.Call, 0.0),
    ):
        player_zero_aggressor = _apply(player_zero_aggressor, action, amount)

    player_one_aggressor = _initial_hu_state()
    for action, amount in (
        (pkrs.ActionEnum.Call, 0.0),
        (pkrs.ActionEnum.Call, 0.0),
        (pkrs.ActionEnum.Check, 0.0),
        (pkrs.ActionEnum.Check, 0.0),
        (pkrs.ActionEnum.Raise, 2.0),
        (pkrs.ActionEnum.Call, 0.0),
    ):
        player_one_aggressor = _apply(player_one_aggressor, action, amount)

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
    (_preflop_raise_vs_flop_bet, _turn_aggressor_player_zero_vs_one),
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
    (_preflop_raise_vs_flop_bet, _turn_aggressor_player_zero_vs_one),
    ids=("street-of-aggression", "turn-aggressor"),
)
def test_history_summary_must_separate_documented_hu_aliases(state_factory) -> None:
    first, second = state_factory()

    assert not np.array_equal(encode_state(first, player_id=0), encode_state(second, player_id=0))
