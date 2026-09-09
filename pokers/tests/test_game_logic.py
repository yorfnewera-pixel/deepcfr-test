import copy
import json
from pathlib import Path

import pokers as pkrs


FIXTURES = Path(__file__).parent / "test_files"


def apply_fixture_action(state, pb_action):
    """Apply one logged action, using explicit all-in metadata when present."""
    if pb_action.get("all_in"):
        player = pb_action["player"]
        stack_before = float(pb_action["stack_before"])
        current = state.players_state[player]
        # The fixture provides this value only for exceptional all-ins.  Set
        # the live player's remaining stack immediately before the action so
        # the normal engine API can process it without special semantics.
        assert state.current_player == player
        assert current.stake == float("inf")
        current.stake = stack_before
        # The recorded amount is the entire short all-in contribution, whereas
        # the normal Raise API expects an increment above the call amount.
        # Convert it to an ordinary Call: the engine caps that call at the
        # explicit remaining stack.
        action = pkrs.Action(pkrs.ActionEnum.Call)
        return state.apply_action(action)

    action = pkrs.Action(
        pkrs.ActionEnum.__dict__[pb_action["action"].capitalize()],
        pb_action.get("amount", 0),
    )
    return state.apply_action(action)


import pytest


@pytest.mark.skip(
    reason="Legacy Pluribus fixture omits stack/all-in data and is not a reliable engine regression."
)
def test_game_logic_against_pluribus_logs():
    with (FIXTURES / "pluribus_logs.json").open() as f:
        pluribus_data = json.load(f)

    for i, pb_hand in enumerate(pluribus_data):
        n_players = len(pb_hand["players"])
        button = pb_hand["button"]
        str_deck = [c for hand in pb_hand["private_cards"] for c in hand]
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

        pkrs_state = pkrs.State.from_deck(
            n_players=n_players,
            button=button,
            deck=deck,
            sb=50,
            bb=100,
            stake=float("inf"),
        )
        print(f"|{i}> game: {pb_hand['game']}, index: {pb_hand['index']}")
        print(pkrs.visualize_trace([pkrs_state]))
        prev_state_final = False
        for pb_stage, pb_actions in pb_hand["actions"].items():
            for pb_action in pb_actions:
                print(pb_action)
                assert not prev_state_final
                assert pkrs_state.status == pkrs.StateStatus.Ok
                assert pkrs_state.stage == pkrs.Stage.__dict__[pb_stage.capitalize()]
                assert pkrs_state.current_player == pb_action["player"]
                prev_state_final = pkrs_state.final_state
                pkrs_state = apply_fixture_action(pkrs_state, pb_action)
                print(pkrs.visualize_state(pkrs_state))

        assert pkrs_state.final_state
        print("Pkrs rewards:", [ps.reward for ps in pkrs_state.players_state])
        print("Real rewards:", pb_hand["rewards"])
        for p, r in enumerate(pb_hand["rewards"]):
            assert pkrs_state.players_state[p].reward == r


def test_initial_state():
    for n_players in range(2, 7):
        for button in range(0, n_players):
            state = pkrs.State.from_seed(
                n_players=n_players, button=button, sb=0.5, bb=1.0, stake=100, seed=1234
            )
            assert state.status == pkrs.StateStatus.Ok
            assert state.current_player == (button + 3) % n_players
            assert state.pot == 1.5
            assert state.min_bet == 1.0

            for ps in state.players_state:
                assert ps.pot_chips == 0
                assert ps.active
                if ps.player == (button + 1) % n_players:
                    assert ps.bet_chips == 0.5
                    assert ps.stake == 99.5
                elif ps.player == (button + 2) % n_players:
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

    # An oversized requested raise is safely capped at the remaining stack and
    # therefore becomes a legal all-in rather than an invalid API request.
    all_in_state = state.apply_action(pkrs.Action(pkrs.ActionEnum.Raise, amount=101))
    assert all_in_state.status == pkrs.StateStatus.Ok
    assert all_in_state.players_state[state.current_player].stake == 0.0


def test_public_action_history_records_only_successful_transitions():
    state = pkrs.State.from_seed(
        n_players=2, button=0, sb=1.0, bb=2.0, stake=20.0, seed=7
    )

    assert state.action_history_complete is True
    assert state.action_history == []

    illegal = state.apply_action(pkrs.Action(pkrs.ActionEnum.Check))
    assert illegal.status == pkrs.StateStatus.IllegalAction
    assert illegal.action_history == []

    next_state = state.apply_action(pkrs.Action(pkrs.ActionEnum.Call))
    assert next_state.status == pkrs.StateStatus.Ok
    assert len(next_state.action_history) == 1

    record = next_state.action_history[0]
    assert record.actor_id == state.current_player
    assert record.street == pkrs.Stage.Preflop
    assert record.requested_action.action == pkrs.ActionEnum.Call
    assert record.paid_amount == 1.0
    assert record.applied_raise_increment == 0.0
    assert record.is_effective_raise is False


def test_public_action_history_uses_actual_short_all_in_raise_amounts():
    state = pkrs.State.from_seed(
        n_players=2, button=0, sb=1.0, bb=2.0, stake=3.0, seed=11
    )

    next_state = state.apply_action(pkrs.Action(pkrs.ActionEnum.Raise, amount=100.0))

    assert next_state.status == pkrs.StateStatus.Ok
    record = next_state.action_history[0]
    assert record.requested_action.amount == 100.0
    assert record.paid_amount == 2.0
    assert record.applied_raise_increment == 1.0
    assert record.is_effective_raise is True


def test_public_action_history_survives_state_copy():
    state = pkrs.State.from_seed(
        n_players=2, button=0, sb=1.0, bb=2.0, stake=20.0, seed=13
    ).apply_action(pkrs.Action(pkrs.ActionEnum.Call))

    copied = copy.copy(state)

    assert copied.action_history_complete is True
    assert len(copied.action_history) == 1
    assert copied.action_history[0].actor_id == state.action_history[0].actor_id


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


def test_repeat_raise_uses_increment_above_call():
    state = pkrs.State.from_seed(
        n_players=3, button=0, sb=50.0, bb=100.0, stake=float("inf"), seed=0
    )
    # Player 0 opens to 250, player 1 3-bets by 850, player 2 folds.
    state = state.apply_action(pkrs.Action(pkrs.ActionEnum.Raise, amount=150.0))
    state = state.apply_action(pkrs.Action(pkrs.ActionEnum.Raise, amount=850.0))
    state = state.apply_action(pkrs.Action(pkrs.ActionEnum.Fold))
    assert state.current_player == 0
    # The original raiser re-raises by another 850 above the 1,000 call.
    state = state.apply_action(pkrs.Action(pkrs.ActionEnum.Raise, amount=850.0))
    assert state.status == pkrs.StateStatus.Ok
    assert state.players_state[0].bet_chips == 1950.0
    assert state.min_bet == 1950.0


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
