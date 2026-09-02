"""Независимые операции над картами для runtime search."""
from __future__ import annotations

from collections.abc import Iterable

import pokers as pkrs

_RANKS = "23456789TJQKA"
_SUITS = "cdhs"


def card_key(card: pkrs.Card) -> tuple[int, int]:
    """Возвращает неизменяемый идентификатор карты по rank и suit."""
    return int(card.rank), int(card.suit)


def parse_card(value: str) -> pkrs.Card:
    """Создаёт карту из привычной записи ранга и масти, например ``Ac``."""
    if len(value) != 2 or value[0] not in _RANKS or value[1] not in _SUITS:
        raise ValueError(f"Некорректная запись карты: {value}")
    card = pkrs.Card.from_string(value[::-1])
    if card is None:
        raise ValueError(f"Не удалось создать карту: {value}")
    return card


def full_deck() -> tuple[pkrs.Card, ...]:
    """Возвращает независимую стандартную колоду из 52 карт."""
    return tuple(parse_card(f"{rank}{suit}") for suit in _SUITS for rank in _RANKS)


def remaining_deck(excluded_cards: Iterable[pkrs.Card]) -> tuple[pkrs.Card, ...]:
    """Возвращает карты, не занятые известными private или public cards."""
    excluded = tuple(excluded_cards)
    excluded_keys = {card_key(card) for card in excluded}
    if len(excluded_keys) != len(excluded):
        raise ValueError("Известные карты содержат дубликаты")
    return tuple(card for card in full_deck() if card_key(card) not in excluded_keys)
