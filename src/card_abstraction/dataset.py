"""Воспроизводимая выборка уникальных канонических карточных ситуаций."""
from __future__ import annotations

import numpy as np

from src.runtime_search.cards import full_deck

from .canonical import canonicalize
from .domain import CardSituation, Street

_BOARD_SIZE_BY_STREET = {
    Street.FLOP: 3,
    Street.TURN: 4,
    Street.RIVER: 5,
}


def sample_unique_situations(
    street: Street,
    count: int,
    master_seed: int,
) -> list[CardSituation]:
    """Возвращает ровно `count` уникальных suit-canonical состояний улицы."""
    if street not in _BOARD_SIZE_BY_STREET:
        raise ValueError("Выборка поддерживает только postflop улицы")
    if count <= 0:
        raise ValueError("count должен быть положительным")

    deck = full_deck()
    board_size = _BOARD_SIZE_BY_STREET[street]
    rng = np.random.default_rng(master_seed)
    seen = set()
    result = []
    attempts_limit = count * 100

    for _ in range(attempts_limit):
        selected = rng.choice(len(deck), size=2 + board_size, replace=False)
        situation = CardSituation.create(
            tuple(deck[int(index)] for index in selected[:2]),
            tuple(deck[int(index)] for index in selected[2:]),
        )
        key = canonicalize(situation)
        if key in seen:
            continue
        seen.add(key)
        result.append(situation)
        if len(result) == count:
            return result

    raise RuntimeError(
        f"Не удалось получить {count} уникальных canonical состояний за {attempts_limit} попыток"
    )
