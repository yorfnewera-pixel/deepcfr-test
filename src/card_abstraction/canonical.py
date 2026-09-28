"""Suit-isomorphism канонизация перебором всех перестановок мастей."""
from __future__ import annotations

from itertools import permutations

from src.cards import card_key

from .domain import CanonicalCardKey, CardSituation


def _apply_suit_permutation(cards, permutation: tuple[int, int, int, int]):
    return tuple(sorted((rank, permutation[suit]) for rank, suit in cards))


def canonicalize(situation: CardSituation) -> CanonicalCardKey:
    """Сворачивает все общие переименования мастей в один ключ."""
    hero = tuple(card_key(card) for card in situation.hero)
    board = tuple(card_key(card) for card in situation.board)
    candidates = (
        CanonicalCardKey(
            street=situation.street,
            hero=_apply_suit_permutation(hero, permutation),
            board=_apply_suit_permutation(board, permutation),
        )
        for permutation in permutations(range(4))
    )
    return min(candidates)
