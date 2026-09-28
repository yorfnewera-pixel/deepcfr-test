"""Детерминированный расчёт terminal equity для карточной абстракции."""
from __future__ import annotations

import hashlib
from itertools import combinations

import numpy as np
import pokers as pkrs

from src.cards import full_deck

from .canonical import canonicalize
from .domain import CardSituation, Street

_FEATURE_VERSION = "hu_postflop_abstraction_v1"
_RUNOUT_SAMPLES = 128
_OPPONENT_SAMPLES = 128
_RANKS = "23456789TJQKA"
_SUITS = "cdhs"


def _canonical_situation(situation: CardSituation) -> CardSituation:
    key = canonicalize(situation)

    def to_card(value: tuple[int, int]) -> pkrs.Card:
        rank, suit = value
        card = pkrs.Card.from_string(f"{_SUITS[suit]}{_RANKS[rank]}")
        if card is None:
            raise ValueError("Не удалось восстановить каноническую карту")
        return card

    return CardSituation.create(
        tuple(to_card(value) for value in key.hero),
        tuple(to_card(value) for value in key.board),
    )


def _rng_for(situation: CardSituation, master_seed: int) -> np.random.Generator:
    key = canonicalize(situation).as_string()
    payload = f"{_FEATURE_VERSION}|{master_seed}|{key}".encode("utf-8")
    seed = int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")
    return np.random.default_rng(seed)


def _remaining_cards(situation: CardSituation) -> tuple[pkrs.Card, ...]:
    excluded = {(int(card.rank), int(card.suit)) for card in (*situation.hero, *situation.board)}
    return tuple(card for card in full_deck() if (int(card.rank), int(card.suit)) not in excluded)


def _payoff(hero, opponent, board) -> float:
    outcome = pkrs.compare_showdown(hero, opponent, list(board))
    return 1.0 if outcome == 1 else 0.5 if outcome == 0 else 0.0


def _exact_river_equity(situation: CardSituation) -> float:
    outcomes = [
        _payoff(situation.hero, opponent, situation.board)
        for opponent in combinations(_remaining_cards(situation), 2)
    ]
    return float(np.mean(outcomes))


def conditional_equities(
    situation: CardSituation,
    master_seed: int = 20260927,
    runout_samples: int = _RUNOUT_SAMPLES,
    opponent_samples: int = _OPPONENT_SAMPLES,
) -> np.ndarray:
    """Возвращает exact river equity либо 128 future-runout conditional equity."""
    if situation.street is Street.PREFLOP:
        raise ValueError("Preflop не имеет postflop equity feature")

    situation = _canonical_situation(situation)
    if situation.street is Street.RIVER:
        return np.asarray([_exact_river_equity(situation)], dtype=np.float64)
    if runout_samples <= 0 or opponent_samples <= 0:
        raise ValueError("Размеры выборки equity должны быть положительными")

    missing_board_cards = 5 - len(situation.board)
    remaining = _remaining_cards(situation)
    runouts = tuple(combinations(remaining, missing_board_cards))
    rng = _rng_for(situation, master_seed)
    sampled_runouts = rng.integers(0, len(runouts), size=runout_samples)
    result = np.empty(runout_samples, dtype=np.float64)

    for output_index, runout_index in enumerate(sampled_runouts):
        board = (*situation.board, *runouts[int(runout_index)])
        unavailable = {(int(card.rank), int(card.suit)) for card in (*situation.hero, *board)}
        opponents = tuple(
            combinations(
                [card for card in full_deck() if (int(card.rank), int(card.suit)) not in unavailable],
                2,
            )
        )
        sampled_opponents = rng.integers(0, len(opponents), size=opponent_samples)
        result[output_index] = np.mean(
            [_payoff(situation.hero, opponents[int(index)], board) for index in sampled_opponents]
        )
    return result
