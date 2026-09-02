import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pokers as pkrs
from src.core.model import encode_state, _build_suit_canonical_map


def test_first_suit_maps_to_zero():
    s = pkrs.State.from_seed(n_players=6, button=0, sb=1, bb=2, stake=200.0, seed=42)
    hand = list(s.players_state[0].hand)
    public = list(s.public_cards)
    m = _build_suit_canonical_map(hand, public)
    first_suit = int(hand[0].suit)
    assert m[first_suit] == 0, f"First suit in hand should be canonical 0, got {m[first_suit]}"
    print("PASS: first suit maps to canonical 0")


def test_deterministic_encoding():
    s1 = pkrs.State.from_seed(n_players=6, button=0, sb=1, bb=2, stake=200.0, seed=42)
    s2 = pkrs.State.from_seed(n_players=6, button=0, sb=1, bb=2, stake=200.0, seed=42)
    e1 = encode_state(s1, 0)
    e2 = encode_state(s2, 0)
    assert np.array_equal(e1, e2), "Same seed must produce identical encoding"
    print("PASS: deterministic encoding")


def test_new_suits_on_board_get_next_canonical():
    s = pkrs.State.from_seed(n_players=6, button=0, sb=1, bb=2, stake=200.0, seed=42)
    hand = list(s.players_state[0].hand)
    hand_suits = {int(c.suit) for c in hand}
    seen = set(hand_suits)
    canonical_idx = len(hand_suits)
    public = list(s.public_cards)
    m = _build_suit_canonical_map(hand, public)
    for card in public:
        s_val = int(card.suit)
        if s_val not in seen:
            assert m[s_val] == canonical_idx, f"New suit on board should get next canonical index"
            seen.add(s_val)
            canonical_idx += 1
    print("PASS: new suits on board get sequential canonical indices")


def test_hand_encoding_uses_canonical_suits():
    s = pkrs.State.from_seed(n_players=6, button=0, sb=1, bb=2, stake=200.0, seed=42)
    hand = list(s.players_state[0].hand)
    m = _build_suit_canonical_map(hand, list(s.public_cards))
    enc = encode_state(s, 0)
    hand_enc = enc[:52]
    for card in hand:
        canonical_suit = m[int(card.suit)]
        rank = int(card.rank)
        expected_pos = canonical_suit * 13 + rank
        assert hand_enc[expected_pos] == 1.0, f"Card at canonical suit={canonical_suit}, rank={rank} not encoded"
    print("PASS: hand encoding uses canonical suits")


if __name__ == '__main__':
    test_first_suit_maps_to_zero()
    test_deterministic_encoding()
    test_new_suits_on_board_get_next_canonical()
    test_hand_encoding_uses_canonical_suits()
    print("\nAll suit isomorphism tests passed!")
