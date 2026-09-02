import pokers as pkrs
import pytest

from src.core.action_space import ActionSlot, legal_action_mask
from src.runtime_search.cards import card_key
from src.runtime_search.spots import build_constructed_spot, build_response_to_bet_spot


@pytest.mark.parametrize(
    ("stage", "expected_public_cards"),
    [(pkrs.Stage.Flop, 3), (pkrs.Stage.Turn, 4), (pkrs.Stage.River, 5)],
)
def test_constructed_spot_preserves_initial_stack_and_has_unique_cards(stage, expected_public_cards):
    state = build_constructed_spot(stage)
    all_cards = [
        *(card for player in state.players_state for card in player.hand),
        *state.public_cards,
        *state.deck,
    ]

    assert state.stage == stage
    assert len(state.public_cards) == expected_public_cards
    assert state.pot == pytest.approx(180.0)
    assert all(player.active for player in state.players_state)
    assert all(player.stake == pytest.approx(170.0) for player in state.players_state)
    assert len({card_key(card) for card in all_cards}) == len(all_cards)


def test_response_to_bet_spot_has_identity_compact_action_set():
    state = build_response_to_bet_spot()
    mask = legal_action_mask(state).astype(bool)

    assert state.current_player == 1
    assert {ActionSlot(index) for index in mask.nonzero()[0]} == {
        ActionSlot.FOLD,
        ActionSlot.CALL,
        ActionSlot.RAISE_POT,
        ActionSlot.ALL_IN,
    }
    assert not mask[ActionSlot.CHECK]
    assert not mask[ActionSlot.RAISE_HALF_POT]
