import pokers as pkrs
import json


def test_game_logic_against_pluribus_logs():
    with open("tests/test_files/pluribus_logs.json") as f:
        pluribus_data = json.load(f)

    for i, pb_hand in enumerate(pluribus_data):
        n_players = len(pb_hand["players"])
        button = pb_hand["button"]
        str_deck = [
            hand[round_] for round_ in range(2) for hand in pb_hand["private_cards"]
        ]
        if "public_cards" in pb_hand:
            if "flop" in pb_hand["public_cards"]:
                str_deck += pb_hand["public_cards"]["flop"]
            if "turn" in pb_hand["public_cards"]:
                str_deck += [pb_hand["public_cards"]["turn"]]
            if "river" in pb_hand["public_cards"]:
                str_deck += [pb_hand["public_cards"]["river"]]
        deck = []
        for str_c in str_deck:
            c = pkrs.Card.from_string(str_c[::-1])
            assert c is not None
            deck.append(c)

        deck += [
            c
            for c in pkrs.Card.collect()
            if not any(
                int(c.rank) == int(d.rank) and int(c.suit) == int(d.suit) for d in deck
            )
        ]

        pkrs_state = pkrs.State.from_deck(
            n_players=n_players,
            button=button,
            deck=deck,
            sb=50,
            bb=100,
            stake=10_000,
            chip_unit=0.5,
        )
        prev_state_final = False
        for pb_stage, pb_actions in pb_hand["actions"].items():
            for pb_action in pb_actions:
                assert not prev_state_final
                assert pkrs_state.status == pkrs.StateStatus.Ok, (
                    i,
                    pb_hand["game"],
                    pb_hand["index"],
                    pkrs_state.from_action,
                    pkrs_state.min_bet,
                    pkrs_state.min_raise,
                )
                assert pkrs_state.stage == pkrs.Stage.__dict__[pb_stage.capitalize()]
                assert pkrs_state.current_player == pb_action["player"]
                amount = pb_action.get("amount", 0)
                if (
                    pb_action["action"] == "call"
                    and pkrs.ActionEnum.Check in pkrs_state.legal_actions
                ):
                    action_enum = pkrs.ActionEnum.Check
                else:
                    action_enum = pkrs.ActionEnum.__dict__[
                        pb_action["action"].capitalize()
                    ]
                action = pkrs.Action(action_enum, amount)
                prev_state_final = pkrs_state.final_state
                pkrs_state = pkrs_state.apply_action(action)

        assert pkrs_state.final_state
        for p, r in enumerate(pb_hand["rewards"]):
            assert pkrs_state.players_state[p].reward == r


def test_initial_state():
    for n_players in range(2, 7):
        for button in range(0, n_players):
            state = pkrs.State.from_seed(
                n_players=n_players, button=button, sb=0.5, bb=1.0, stake=100, seed=1234
            )
            assert state.status == pkrs.StateStatus.Ok
            assert state.current_player == (
                button if n_players == 2 else (button + 3) % n_players
            )
            assert state.pot == 1.5
            assert state.min_bet == 1.0

            for ps in state.players_state:
                assert ps.pot_chips == 0
                assert ps.active
                if ps.player == (
                    button if n_players == 2 else (button + 1) % n_players
                ):
                    assert ps.bet_chips == 0.5
                    assert ps.stake == 99.5
                elif ps.player == (
                    (button + 1) % n_players
                    if n_players == 2
                    else (button + 2) % n_players
                ):
                    assert ps.bet_chips == 1.0
                    assert ps.stake == 99
                else:
                    assert ps.bet_chips == 0.0
                    assert ps.stake == 100


def test_illegal_actions():
    state = pkrs.State.from_seed(
        n_players=6, button=0, sb=0.5, bb=1.0, stake=100, seed=1234
    )
    assert state.status == pkrs.StateStatus.Ok

    illegal_action_state = state.apply_action(pkrs.Action(pkrs.ActionEnum.Check))
    assert illegal_action_state.status == pkrs.StateStatus.IllegalAction

    high_bet_state = state.apply_action(pkrs.Action(pkrs.ActionEnum.Raise, amount=101))
    assert high_bet_state.status == pkrs.StateStatus.HighBet


def test_forced_checkdown_runs_out_to_showdown():
    state = pkrs.State.from_seed(
        n_players=3, button=0, sb=1.0, bb=2.0, stake=4.0, seed=0
    )

    state = state.apply_action(pkrs.Action(pkrs.ActionEnum.Raise, amount=2.0))
    assert not state.final_state

    state = state.apply_action(pkrs.Action(pkrs.ActionEnum.Call))
    assert not state.final_state

    state = state.apply_action(pkrs.Action(pkrs.ActionEnum.Call))

    assert state.final_state
    assert state.stage == pkrs.Stage.Showdown
    assert state.status == pkrs.StateStatus.Ok
    assert state.legal_actions == []
    assert abs(sum(ps.reward for ps in state.players_state)) < 1e-9


def test_raise_is_not_legal_when_call_uses_entire_stack():
    state = pkrs.State.from_seed(
        n_players=3, button=0, sb=1.0, bb=2.0, stake=4.0, seed=0
    )

    state = state.apply_action(pkrs.Action(pkrs.ActionEnum.Raise, amount=2.0))
    state = state.apply_action(pkrs.Action(pkrs.ActionEnum.Call))

    current_player = state.players_state[state.current_player]
    call_amount = state.min_bet - current_player.bet_chips

    assert call_amount == current_player.stake
    assert pkrs.ActionEnum.Call in state.legal_actions
    assert pkrs.ActionEnum.Raise not in state.legal_actions
