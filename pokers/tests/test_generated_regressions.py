"""Deterministic regression scenarios for the controlled self-play game engine."""

import pytest

import pokers as pkrs


def card(value):
    parsed = pkrs.Card.from_string(value[::-1])
    assert parsed is not None
    return parsed


def mid_hand(**overrides):
    defaults = dict(
        n_players=2,
        button=0,
        sb=1.0,
        bb=2.0,
        stake=100.0,
        deck=[card(value) for value in ("7c", "8d", "9h", "Ts", "Jc")],
        hole_cards=[(card("Ac"), card("Kd")), (card("Qh"), card("Js"))],
        public_cards=[card("2c"), card("3d"), card("4h")],
        stage=pkrs.Stage.Flop,
        pot=20.0,
        bet_chips=[0.0, 0.0],
        pot_chips=[10.0, 10.0],
        active=[True, True],
        last_stage_action=[None, None],
        current_player=1,
        last_raise_increment=2.0,
    )
    defaults.update(overrides)
    return pkrs.State.from_mid_hand(**defaults)


def assert_conservation(before, after):
    before_total = before.pot + sum(player.stake for player in before.players_state)
    after_total = after.pot + sum(player.stake for player in after.players_state)
    assert after_total == pytest.approx(before_total)
    assert all(player.stake >= 0.0 for player in after.players_state)
    assert after.pot >= 0.0


def test_flop_bet_call_advances_to_turn_without_mutating_input():
    state = mid_hand()
    original = (state.stage, state.pot, state.current_player, [p.stake for p in state.players_state])

    bet = state.apply_action(pkrs.Action(pkrs.ActionEnum.Raise, amount=10.0))
    assert state.stage == original[0]
    assert state.pot == original[1]
    assert state.current_player == original[2]
    assert [p.stake for p in state.players_state] == original[3]
    assert bet.stage == pkrs.Stage.Flop
    assert bet.min_bet == 10.0
    assert bet.current_player == 0
    assert_conservation(state, bet)

    called = bet.apply_action(pkrs.Action(pkrs.ActionEnum.Call))
    assert called.status == pkrs.StateStatus.Ok
    assert called.stage == pkrs.Stage.Turn
    assert len(called.public_cards) == 4
    assert called.pot == pytest.approx(40.0)
    assert [p.bet_chips for p in called.players_state] == [0.0, 0.0]
    assert [p.pot_chips for p in called.players_state] == [20.0, 20.0]
    assert_conservation(bet, called)


def test_bet_raise_fold_awards_pot_and_keeps_rewards_zero_sum():
    state = mid_hand()
    bet = state.apply_action(pkrs.Action(pkrs.ActionEnum.Raise, amount=10.0))
    raise_state = bet.apply_action(pkrs.Action(pkrs.ActionEnum.Raise, amount=20.0))
    folded = raise_state.apply_action(pkrs.Action(pkrs.ActionEnum.Fold))

    assert folded.status == pkrs.StateStatus.Ok
    assert folded.final_state
    assert folded.legal_actions == []
    assert folded.players_state[0].reward == pytest.approx(20.0)
    assert folded.players_state[1].reward == pytest.approx(-20.0)
    assert sum(player.reward for player in folded.players_state) == pytest.approx(0.0)


def test_full_reraise_updates_minimum_raise_increment():
    state = mid_hand()
    bet = state.apply_action(pkrs.Action(pkrs.ActionEnum.Raise, amount=10.0))
    reraised = bet.apply_action(pkrs.Action(pkrs.ActionEnum.Raise, amount=20.0))

    assert reraised.status == pkrs.StateStatus.Ok
    assert reraised.min_bet == pytest.approx(30.0)
    assert reraised.last_raise_increment == pytest.approx(20.0)
    assert reraised.players_state[0].bet_chips == pytest.approx(30.0)
    assert_conservation(bet, reraised)


def test_short_all_in_raise_is_accepted_but_does_not_reopen_minimum_raise():
    state = mid_hand(
        stake=35.0,
        pot=40.0,
        bet_chips=[10.0, 0.0],
        pot_chips=[10.0, 20.0],
        last_stage_action=[pkrs.ActionEnum.Raise, None],
        current_player=1,
        last_raise_increment=10.0,
    )
    assert state.players_state[1].stake == pytest.approx(15.0)

    all_in = state.apply_action(pkrs.Action(pkrs.ActionEnum.Raise, amount=10.0))
    assert all_in.status == pkrs.StateStatus.Ok
    assert all_in.players_state[1].stake == pytest.approx(0.0)
    assert all_in.players_state[1].bet_chips == pytest.approx(15.0)
    assert all_in.min_bet == pytest.approx(15.0)
    assert all_in.last_raise_increment == pytest.approx(10.0)
    assert_conservation(state, all_in)


def test_all_in_preflop_forces_board_runout_and_showdown():
    state = pkrs.State.from_seed(n_players=2, button=0, sb=1.0, bb=2.0, stake=4.0, seed=7)

    raised = state.apply_action(pkrs.Action(pkrs.ActionEnum.Raise, amount=2.0))
    showdown = raised.apply_action(pkrs.Action(pkrs.ActionEnum.Call))

    assert showdown.status == pkrs.StateStatus.Ok
    assert showdown.final_state
    assert showdown.stage == pkrs.Stage.Showdown
    assert len(showdown.public_cards) == 5
    assert showdown.legal_actions == []
    assert sum(player.reward for player in showdown.players_state) == pytest.approx(0.0)


def test_river_check_check_reaches_showdown_without_dealing_more_cards():
    state = mid_hand(
        deck=[],
        public_cards=[card("2c"), card("3d"), card("4h"), card("5s"), card("6c")],
        stage=pkrs.Stage.River,
        current_player=1,
    )

    first_check = state.apply_action(pkrs.Action(pkrs.ActionEnum.Check))
    showdown = first_check.apply_action(pkrs.Action(pkrs.ActionEnum.Check))

    assert showdown.status == pkrs.StateStatus.Ok
    assert showdown.final_state
    assert showdown.stage == pkrs.Stage.Showdown
    assert len(showdown.public_cards) == 5
    assert sum(player.reward for player in showdown.players_state) == pytest.approx(0.0)
