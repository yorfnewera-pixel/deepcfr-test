from itertools import combinations

import numpy as np
import pytest
import pokers as pkrs

from src.card_abstraction.domain import CardSituation
from src.runtime_search.cards import full_deck, parse_card


def _cards(*values: str):
    return tuple(parse_card(value) for value in values)


def _exact_equity(situation: CardSituation) -> float:
    excluded = {(int(card.rank), int(card.suit)) for card in (*situation.hero, *situation.board)}
    outcomes = []
    for opponent in combinations(
        [card for card in full_deck() if (int(card.rank), int(card.suit)) not in excluded], 2
    ):
        outcome = pkrs.compare_showdown(situation.hero, opponent, list(situation.board))
        outcomes.append(1.0 if outcome == 1 else 0.5 if outcome == 0 else 0.0)
    return sum(outcomes) / len(outcomes)


def test_river_feature_matches_exact_showdown_equity():
    from src.card_abstraction.features import build_feature

    situation = CardSituation.create(
        _cards("As", "Ah"),
        _cards("2c", "3d", "7h", "9s", "Jc"),
    )

    feature = build_feature(situation)

    assert feature.shape == (27,)
    assert feature.dtype == np.float32
    assert feature[0] == pytest.approx(_exact_equity(situation), abs=1e-7)
    assert feature[1] == 0.0
    assert feature[2:7] == pytest.approx(np.repeat(feature[0], 5), abs=1e-7)
    assert feature[7:].sum() == pytest.approx(1.0, abs=1e-6)


def test_flop_feature_is_seed_reproducible_and_suit_invariant():
    from src.card_abstraction.features import build_feature

    source = CardSituation.create(_cards("Ac", "Kc"), _cards("2c", "7d", "Th"))
    permuted = CardSituation.create(_cards("Kh", "Ah"), _cards("Ts", "7c", "2h"))

    first = build_feature(source, master_seed=20260927)

    assert np.array_equal(first, build_feature(source, master_seed=20260927))
    assert np.array_equal(first, build_feature(permuted, master_seed=20260927))
    assert not np.array_equal(first, build_feature(source, master_seed=7))
