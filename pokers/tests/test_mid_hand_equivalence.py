"""S1 regression: reconstructed postflop state must match live engine state."""

import copy

import pytest

import pokers as pkrs


def card(value):
    result = pkrs.Card.from_string(value[::-1])
    assert result is not None
    return result


def card_key(value):
    return int(value.rank), int(value.suit)


def state_snapshot(state):
    return {
        "current_player": state.current_player,
        "stage": state.stage,
        "button": state.button,
        "public_cards": [card_key(value) for value in state.public_cards],
        "deck": [card_key(value) for value in state.deck],
        "pot": state.pot,
        "min_bet": state.min_bet,
        "last_raise_increment": state.last_raise_increment,
        "legal_actions": list(state.legal_actions),
        "players": [
            {
                "player": player.player,
                "hand": (card_key(player.hand[0]), card_key(player.hand[1])),
                "bet_chips": player.bet_chips,
                "pot_chips": player.pot_chips,
                "stake": player.stake,
                "active": player.active,
                "last_stage_action": player.last_stage_action,
            }
            for player in state.players_state
        ],
    }


def assert_equivalent(left, right):
    assert state_snapshot(left) == state_snapshot(right)


def fixed_deck():
    # from_deck consumes the first two cards for player 1, then player 2,
    # then player 0 when button=0. Remaining cards are dealt to the board.
    dealt = [
        "Ah", "Kd",  # player 1
        "Qs", "Jc",  # player 2
        "Tc", "9d",  # player 0
        "2c", "3d", "4h", "5s", "6c",  # flop, turn, river
    ]
    used = set(dealt)
    remaining = [
        rank + suit
        for rank in "23456789TJQKA"
        for suit in "cdhs"
        if rank + suit not in used
    ]
    return [card(value) for value in dealt + remaining]


def test_mid_hand_reconstruction_matches_live_flop_and_next_round():
    live = pkrs.State.from_deck(
        n_players=3,
        button=0,
        sb=1.0,
        bb=2.0,
        stake=100.0,
        deck=fixed_deck(),
    )

    # Preflop order with button=0 is player 0, then SB/player 1, then
    # BB/player 2. All three calls reach a known flop state through the normal
    # engine, with all bookkeeping supplied by it.
    for action in (
        pkrs.Action(pkrs.ActionEnum.Call),
        pkrs.Action(pkrs.ActionEnum.Call),
        pkrs.Action(pkrs.ActionEnum.Call),
    ):
        live = live.apply_action(action)
        assert live.status == pkrs.StateStatus.Ok

    assert live.stage == pkrs.Stage.Flop
    rebuilt = pkrs.State.from_mid_hand(
        n_players=3,
        button=live.button,
        sb=live.sb,
        bb=live.bb,
        stake=100.0,
        deck=list(live.deck),
        hole_cards=[player.hand for player in live.players_state],
        public_cards=list(live.public_cards),
        stage=live.stage,
        pot=live.pot,
        bet_chips=[player.bet_chips for player in live.players_state],
        pot_chips=[player.pot_chips for player in live.players_state],
        active=[player.active for player in live.players_state],
        last_stage_action=[player.last_stage_action for player in live.players_state],
        current_player=live.current_player,
        last_raise_increment=live.last_raise_increment,
    )

    assert_equivalent(live, rebuilt)

    # Same postflop line must retain identical actions and close the betting
    # round at exactly the same node in both representations.
    for action in (
        pkrs.Action(pkrs.ActionEnum.Check),
        pkrs.Action(pkrs.ActionEnum.Raise, amount=6.0),
        pkrs.Action(pkrs.ActionEnum.Call),
        pkrs.Action(pkrs.ActionEnum.Call),
    ):
        live = live.apply_action(action)
        rebuilt = rebuilt.apply_action(action)
        assert live.status == pkrs.StateStatus.Ok
        assert rebuilt.status == pkrs.StateStatus.Ok
        assert_equivalent(live, rebuilt)

    assert live.stage == pkrs.Stage.Turn


def test_copy_and_deepcopy_return_independent_state_clones():
    state = pkrs.State.from_seed(
        n_players=2, button=0, sb=1.0, bb=2.0, stake=20.0, seed=17
    )
    shallow = copy.copy(state)
    deep = copy.deepcopy(state)

    assert_equivalent(state, shallow)
    assert_equivalent(state, deep)

    child = shallow.apply_action(pkrs.Action(pkrs.ActionEnum.Call))
    assert child.current_player != shallow.current_player
    assert_equivalent(state, deep)
    assert state.current_player == shallow.current_player
