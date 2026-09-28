"""Типы и проверки входных карточных ситуаций."""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Sequence

import pokers as pkrs

from src.cards import card_key


class Street(str, Enum):
    """Улица, определяемая исключительно числом открытых карт."""

    PREFLOP = "preflop"
    FLOP = "flop"
    TURN = "turn"
    RIVER = "river"


_STREET_BY_BOARD_SIZE = {
    0: Street.PREFLOP,
    3: Street.FLOP,
    4: Street.TURN,
    5: Street.RIVER,
}


@dataclass(frozen=True)
class CardSituation:
    """Проверенная private/public карточная ситуация одного игрока."""

    hero: tuple[pkrs.Card, pkrs.Card]
    board: tuple[pkrs.Card, ...]
    street: Street

    @classmethod
    def create(
        cls,
        hero: Sequence[pkrs.Card],
        board: Sequence[pkrs.Card],
    ) -> "CardSituation":
        if len(hero) != 2:
            raise ValueError("Hero должен содержать ровно две карты")
        street = _STREET_BY_BOARD_SIZE.get(len(board))
        if street is None:
            raise ValueError("Board должен содержать 0, 3, 4 или 5 карт")

        all_cards = (*hero, *board)
        keys = tuple(card_key(card) for card in all_cards)
        if len(set(keys)) != len(keys):
            raise ValueError("Карточная ситуация содержит дубликат")

        return cls(hero=(hero[0], hero[1]), board=tuple(board), street=street)


@dataclass(frozen=True, order=True)
class CanonicalCardKey:
    """Неизменяемый suit-isomorphic ключ без игровой истории."""

    street: Street
    hero: tuple[tuple[int, int], tuple[int, int]]
    board: tuple[tuple[int, int], ...]

    def as_string(self) -> str:
        """Возвращает стабильную сериализацию для артефактов."""
        cards = (*self.hero, *self.board)
        return f"{self.street.value}:" + ",".join(
            f"{rank}-{suit}" for rank, suit in cards
        )
