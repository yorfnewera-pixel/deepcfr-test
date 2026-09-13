import pokers as pkrs
import pytest


def card(value):
    return pkrs.Card.from_string(value[::-1])


def mid_hand_arguments(stage=pkrs.Stage.Preflop, **overrides):
    if stage == pkrs.Stage.Preflop:
        public_cards = []
    elif stage == pkrs.Stage.Flop:
        public_cards = [card("2c"), card("3d"), card("4h")]
    elif stage == pkrs.Stage.Turn:
        public_cards = [card("2c"), card("3d"), card("4h"), card("5s")]
    else:
        public_cards = [card("2c"), card("3d"), card("4h"), card("5s"), card("6c")]

    arguments = dict(
        n_players=2,
        button=0,
        sb=1.0,
        bb=2.0,
        stake=20.0,
        deck=[],
        hole_cards=[(card("Ac"), card("Kd")), (card("Qh"), card("Js"))],
        public_cards=public_cards,
        stage=stage,
        pot=3.0,
        bet_chips=[1.0, 2.0],
        pot_chips=[0.0, 0.0],
        active=[True, True],
        last_stage_action=[None, None],
        current_player=0,
        last_raise_increment=2.0,
    )
    arguments.update(overrides)
    return arguments


def test_from_mid_hand_requires_explicit_active_actor_and_rejects_showdown():
    with pytest.raises(OSError, match="unresolved Showdown"):
        pkrs.State.from_mid_hand(**mid_hand_arguments(pkrs.Stage.Showdown))

    with pytest.raises(OSError, match="must be provided"):
        pkrs.State.from_mid_hand(**mid_hand_arguments(current_player=None))

    with pytest.raises(OSError, match="must be in range"):
        pkrs.State.from_mid_hand(**mid_hand_arguments(current_player=2))

    with pytest.raises(OSError, match="active player"):
        pkrs.State.from_mid_hand(**mid_hand_arguments(current_player=1, active=[True, False]))


@pytest.mark.parametrize("increment", (None, float("nan"), float("inf"), 0.0, -1.0, 1.0))
def test_from_mid_hand_requires_valid_last_raise_increment(increment):
    with pytest.raises(OSError, match="last_raise_increment"):
        pkrs.State.from_mid_hand(**mid_hand_arguments(last_raise_increment=increment))


def test_from_mid_hand_accepts_explicit_actor_for_non_terminal_streets():
    for stage in (pkrs.Stage.Preflop, pkrs.Stage.Flop, pkrs.Stage.Turn, pkrs.Stage.River):
        state = pkrs.State.from_mid_hand(**mid_hand_arguments(stage, current_player=1))
        assert state.stage == stage
        assert state.current_player == 1


def test_from_mid_hand_preserves_minimum_raise_after_large_prior_raise():
    state = pkrs.State.from_mid_hand(
        **mid_hand_arguments(
            stake=100.0,
            pot=12.0,
            bet_chips=[10.0, 0.0],
            pot_chips=[0.0, 2.0],
            last_stage_action=[pkrs.ActionEnum.Raise, None],
            current_player=1,
            last_raise_increment=8.0,
        )
    )

    too_small = state.apply_action(pkrs.Action(pkrs.ActionEnum.Raise, amount=2.0))
    full_raise = state.apply_action(pkrs.Action(pkrs.ActionEnum.Raise, amount=8.0))

    assert too_small.status == pkrs.StateStatus.IllegalAction
    assert full_raise.status == pkrs.StateStatus.Ok
    assert full_raise.players_state[1].bet_chips == pytest.approx(18.0)


def test_from_mid_hand_preserves_folded_players_and_turn_owner():
    deck = [card(x) for x in ("2c", "3d", "4h", "5s", "6c", "7d", "8h", "9s", "Tc", "Jd")]
    state = pkrs.State.from_mid_hand(
        n_players=3,
        button=0,
        sb=0.5,
        bb=1.0,
        stake=100.0,
        deck=deck,
        hole_cards=[(card("Ac"), card("Kd")), (card("Qh"), card("Js")), (card("Tc"), card("9d"))],
        public_cards=[card("2c"), card("3d"), card("4h")],
        stage=pkrs.Stage.Flop,
        pot=12.0,
        bet_chips=[0.0, 0.0, 0.0],
        pot_chips=[6.0, 6.0, 0.0],
        active=[True, True, False],
        last_stage_action=[None, None, pkrs.ActionEnum.Fold],
        current_player=1,
        last_raise_increment=2.0,
    )
    assert state.current_player == 1
    assert [p.active for p in state.players_state] == [True, True, False]
    assert state.last_raise_increment == 2.0
    assert pkrs.ActionEnum.Fold not in state.legal_actions
    assert state.action_history_complete is False
    assert state.action_history == []


def test_mid_hand_can_copy_complete_public_history_from_live_state():
    source = pkrs.State.from_seed(
        n_players=2, button=0, sb=1.0, bb=2.0, stake=20.0, seed=7
    ).apply_action(pkrs.Action(pkrs.ActionEnum.Call))
    rebuilt = pkrs.State.from_mid_hand(
        n_players=2,
        button=source.button,
        sb=source.sb,
        bb=source.bb,
        stake=20.0,
        deck=list(source.deck),
        hole_cards=[tuple(player.hand) for player in source.players_state],
        public_cards=list(source.public_cards),
        stage=source.stage,
        pot=source.pot,
        bet_chips=[player.bet_chips for player in source.players_state],
        pot_chips=[player.pot_chips for player in source.players_state],
        active=[player.active for player in source.players_state],
        last_stage_action=[player.last_stage_action for player in source.players_state],
        current_player=source.current_player,
        last_raise_increment=source.last_raise_increment,
    )

    rebuilt.copy_public_history_from(source)

    assert rebuilt.action_history_complete is True
    assert len(rebuilt.action_history) == len(source.action_history) == 1
    assert rebuilt.action_history[0].actor_id == source.action_history[0].actor_id


def test_mid_hand_rejects_incomplete_public_history_copy():
    source = pkrs.State.from_mid_hand(
        n_players=2,
        button=0,
        sb=1.0,
        bb=2.0,
        stake=20.0,
        deck=[card(value) for value in ("2c", "3d", "4h", "5s")],
        hole_cards=[(card("Ac"), card("Kd")), (card("Qh"), card("Js"))],
        public_cards=[],
        stage=pkrs.Stage.Preflop,
        pot=3.0,
        bet_chips=[1.0, 2.0],
        pot_chips=[0.0, 0.0],
        active=[True, True],
        last_stage_action=[None, None],
        current_player=0,
        last_raise_increment=2.0,
    )
    target = source.__copy__()

    with pytest.raises(ValueError):
        target.copy_public_history_from(source)


def test_short_all_in_call_uses_remaining_stack_without_reopening_betting():
    # Player 1 has only 1.0 left while facing a 10.0 river bet.
    state = pkrs.State.from_mid_hand(
        n_players=2,
        button=0,
        sb=1.0,
        bb=2.0,
        stake=19.0,
        deck=[],
        hole_cards=[(card("Ac"), card("Kd")), (card("Qh"), card("Js"))],
        public_cards=[card("2c"), card("3d"), card("4h"), card("5s"), card("6c")],
        stage=pkrs.Stage.River,
        pot=27.0,
        # Player 1 has put 18.0 into earlier streets and has 1.0 left.
        # Facing a 10.0 bet, their one-chip action is a short all-in call.
        bet_chips=[10.0, 9.0],
        pot_chips=[0.0, 9.0],
        active=[True, True],
        last_stage_action=[None, pkrs.ActionEnum.Check],
        current_player=1,
        last_raise_increment=4.0,
    )
    result = state.apply_action(pkrs.Action(pkrs.ActionEnum.Call))
    assert result.status == pkrs.StateStatus.Ok
    assert result.players_state[1].stake == 0.0
    assert result.players_state[1].bet_chips == 10.0
    # The public action set remains strict: a player who cannot cover the call
    # may call all-in but is not offered a raise action to the solver/model.
    assert pkrs.ActionEnum.Raise not in state.legal_actions


def test_three_player_river_short_all_in_call_reaches_showdown():
    # Player 1 has only 40.0 remaining after contributing 60.0 earlier.
    # They call Player 0's 100.0 river bet all-in; Player 2 then calls 100.0.
    state = pkrs.State.from_mid_hand(
        n_players=3,
        button=0,
        sb=1.0,
        bb=2.0,
        stake=100.0,
        deck=[],
        hole_cards=[(card("Ac"), card("Kd")), (card("Qh"), card("Js")), (card("Tc"), card("9d"))],
        public_cards=[card("2c"), card("3d"), card("4h"), card("5s"), card("6c")],
        stage=pkrs.Stage.River,
        pot=260.0,
        # stake is the initial stack.  With 60.0 already committed on earlier
        # streets and 0.0 in this round, Player 1 has 40.0 for a short call.
        bet_chips=[100.0, 0.0, 0.0],
        pot_chips=[0.0, 60.0, 0.0],
        active=[True, True, True],
        last_stage_action=[pkrs.ActionEnum.Raise, None, None],
        current_player=1,
        last_raise_increment=100.0,
    )

    short_called = state.apply_action(pkrs.Action(pkrs.ActionEnum.Call))
    assert short_called.status == pkrs.StateStatus.Ok
    assert short_called.players_state[1].stake == 0.0
    assert short_called.players_state[1].bet_chips < short_called.min_bet

    showdown = short_called.apply_action(pkrs.Action(pkrs.ActionEnum.Call))
    assert showdown.status == pkrs.StateStatus.Ok
    assert showdown.final_state
    assert showdown.stage == pkrs.Stage.Showdown
    assert showdown.legal_actions == []
