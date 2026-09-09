import pokers as pkrs


def card(value):
    return pkrs.Card.from_string(value[::-1])


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
