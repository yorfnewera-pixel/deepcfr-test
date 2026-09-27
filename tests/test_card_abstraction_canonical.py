import pytest

from src.runtime_search.cards import parse_card


def _cards(*values: str):
    return tuple(parse_card(value) for value in values)


def test_canonicalize_ignores_card_order_and_common_suit_permutation():
    from src.card_abstraction.canonical import canonicalize
    from src.card_abstraction.domain import CardSituation

    source = CardSituation.create(
        _cards("Ac", "Kc"),
        _cards("2c", "7d", "Th"),
    )
    permuted = CardSituation.create(
        _cards("Kh", "Ah"),
        _cards("Ts", "7c", "2h"),
    )

    assert canonicalize(source) == canonicalize(permuted)


@pytest.mark.parametrize("board", (_cards("2c"), _cards("2c", "3d", "4h", "5s", "6c", "7d")))
def test_card_situation_rejects_invalid_board_size(board):
    from src.card_abstraction.domain import CardSituation

    with pytest.raises(ValueError, match="Board"):
        CardSituation.create(_cards("Ac", "Kd"), board)


def test_card_situation_rejects_duplicate_card_between_hero_and_board():
    from src.card_abstraction.domain import CardSituation

    with pytest.raises(ValueError, match="дубликат"):
        CardSituation.create(_cards("Ac", "Kd"), _cards("Ac", "7d", "Th"))


def test_lossless_preflop_table_has_exact_169_classes():
    from src.card_abstraction.preflop import build_lossless_preflop_table

    table = build_lossless_preflop_table()
    classes = table[["class_id", "class_kind"]].drop_duplicates()

    assert len(table) == 1326
    assert len(classes) == 169
    assert classes["class_kind"].value_counts().to_dict() == {
        "pair": 13,
        "suited": 78,
        "offsuit": 78,
    }
