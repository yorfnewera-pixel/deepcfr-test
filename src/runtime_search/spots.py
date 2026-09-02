"""Детерминированные constructed spots для runtime search."""
from __future__ import annotations

import pokers as pkrs

from src.runtime_search.cards import parse_card, remaining_deck

_HOLE_CARDS = (
    ("Ac", "Kd"),
    ("Qh", "Js"),
    ("Tc", "9d"),
    ("8s", "7c"),
    ("6d", "5h"),
    ("4s", "3c"),
)
_TURN_BOARD = ("Ah", "Kh", "Qd", "Jc")
_RIVER_CARD = "Td"
_INITIAL_STACK = 200.0
_COMMITTED_PER_PLAYER = 30.0


def _parsed_hole_cards() -> list[tuple[pkrs.Card, pkrs.Card]]:
    """Возвращает типизированные карманные карты для constructed spots."""
    return [
        (parse_card(hand[0]), parse_card(hand[1]))
        for hand in _HOLE_CARDS
    ]


def build_constructed_spot(stage: pkrs.Stage) -> pkrs.State:
    """Строит 6-max flop, turn или river state с согласованными stack и pot."""
    if stage not in (pkrs.Stage.Flop, pkrs.Stage.Turn, pkrs.Stage.River):
        raise ValueError("Constructed spot поддерживает только Flop, Turn или River")

    public_values = _TURN_BOARD[:3]
    if stage in (pkrs.Stage.Turn, pkrs.Stage.River):
        public_values += (_TURN_BOARD[3],)
    if stage == pkrs.Stage.River:
        public_values += (_RIVER_CARD,)
    hole_cards = _parsed_hole_cards()
    public_cards = [parse_card(card) for card in public_values]
    known_cards = [card for hand in hole_cards for card in hand] + public_cards

    return pkrs.State.from_mid_hand(
        n_players=6,
        button=0,
        sb=1.0,
        bb=2.0,
        stake=_INITIAL_STACK,
        deck=list(remaining_deck(known_cards)),
        hole_cards=hole_cards,
        public_cards=public_cards,
        stage=stage,
        pot=_COMMITTED_PER_PLAYER * 6,
        bet_chips=[0.0] * 6,
        pot_chips=[_COMMITTED_PER_PLAYER] * 6,
        active=[True] * 6,
        last_stage_action=[None] * 6,
        current_player=0,
        last_raise_increment=2.0,
        verbose=False,
    )


def build_response_to_bet_spot() -> pkrs.State:
    """Строит turn response-to-bet spot с identity compact action set."""
    hole_cards = _parsed_hole_cards()
    public_cards = [parse_card(card) for card in _TURN_BOARD]
    known_cards = [card for hand in hole_cards for card in hand] + public_cards
    return pkrs.State.from_mid_hand(
        n_players=6,
        button=0,
        sb=1.0,
        bb=2.0,
        stake=_INITIAL_STACK,
        deck=list(remaining_deck(known_cards)),
        hole_cards=hole_cards,
        public_cards=public_cards,
        stage=pkrs.Stage.Turn,
        pot=70.0,
        bet_chips=[10.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        pot_chips=[10.0] * 6,
        active=[True] * 6,
        last_stage_action=[pkrs.ActionEnum.Raise, None, None, None, None, None],
        current_player=1,
        last_raise_increment=10.0,
        verbose=False,
    )
