"""Полная lossless preflop абстракция из 169 классов."""
from __future__ import annotations

from itertools import combinations

import pandas as pd

from src.cards import card_key, full_deck

from .canonical import canonicalize
from .domain import CardSituation


def _class_kind(key) -> str:
    first_rank, first_suit = key.hero[0]
    second_rank, second_suit = key.hero[1]
    if first_rank == second_rank:
        return "pair"
    return "suited" if first_suit == second_suit else "offsuit"


def build_lossless_preflop_table() -> pd.DataFrame:
    """Возвращает все 1326 physical hands и их 169 точных классов."""
    rows = []
    for hero in combinations(full_deck(), 2):
        key = canonicalize(CardSituation.create(hero, ()))
        rows.append(
            {
                "hero_first": card_key(hero[0]),
                "hero_second": card_key(hero[1]),
                "canonical_key": key.as_string(),
                "class_kind": _class_kind(key),
            }
        )

    table = pd.DataFrame(rows)
    class_ids = {
        key: index for index, key in enumerate(sorted(table["canonical_key"].unique()))
    }
    table.insert(2, "class_id", table["canonical_key"].map(class_ids).astype("int16"))
    return table
