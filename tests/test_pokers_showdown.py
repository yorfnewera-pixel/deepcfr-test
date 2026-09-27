import pytest
import pokers as pkrs


def _card(value: str) -> pkrs.Card:
    card = pkrs.Card.from_string(value[::-1])
    assert card is not None
    return card


def test_compare_showdown_returns_win_for_stronger_hero_hand():
    result = pkrs.compare_showdown(
        (_card("As"), _card("Ah")),
        (_card("Kc"), _card("Kd")),
        [_card("2c"), _card("3d"), _card("7h"), _card("9s"), _card("Jc")],
    )

    assert result == 1


def test_compare_showdown_returns_loss_for_weaker_hero_hand():
    result = pkrs.compare_showdown(
        (_card("Kc"), _card("Kd")),
        (_card("As"), _card("Ah")),
        [_card("2c"), _card("3d"), _card("7h"), _card("9s"), _card("Jc")],
    )

    assert result == -1


def test_compare_showdown_returns_tie_for_playing_board():
    result = pkrs.compare_showdown(
        (_card("2c"), _card("3d")),
        (_card("4h"), _card("5s")),
        [_card("As"), _card("Kd"), _card("Qh"), _card("Jc"), _card("Ts")],
    )

    assert result == 0


def test_compare_showdown_rejects_duplicate_card():
    with pytest.raises(ValueError, match="дубликат"):
        pkrs.compare_showdown(
            (_card("As"), _card("Ah")),
            (_card("Kc"), _card("Kd")),
            [_card("As"), _card("3d"), _card("7h"), _card("9s"), _card("Jc")],
        )
