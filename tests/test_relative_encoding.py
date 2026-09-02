"""Тесты hero-relative encoding: позиционная инвариантность, корректность относительных фич."""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pokers as pkrs
from src.core.model import encode_state


def test_encoding_length():
    state = pkrs.State.from_seed(n_players=6, button=3, sb=1, bb=2, stake=200.0, seed=42)
    assert len(encode_state(state, 0)) == 157
    assert len(encode_state(state, 3)) == 157
    print("PASS: encoding length = 157 for all player_ids")


def test_relative_button():
    state = pkrs.State.from_seed(n_players=6, button=3, sb=1, bb=2, stake=200.0, seed=42)

    enc0 = encode_state(state, player_id=0)
    button_start = 52 + 52 + 5 + 1
    rel_btn0 = np.argmax(enc0[button_start:button_start + 6])
    assert rel_btn0 == (3 - 0) % 6, f"player_id=0: relative_button={rel_btn0}, expected {(3-0)%6}"

    enc3 = encode_state(state, player_id=3)
    rel_btn3 = np.argmax(enc3[button_start:button_start + 6])
    assert rel_btn3 == (3 - 3) % 6, f"player_id=3: relative_button={rel_btn3}, expected {(3-3)%6}"

    print(f"PASS: relative button -- player_id=0 => {rel_btn0}, player_id=3 => {rel_btn3}")


def test_relative_current_player():
    state = pkrs.State.from_seed(n_players=6, button=3, sb=1, bb=2, stake=200.0, seed=42)
    cp = int(state.current_player)

    enc0 = encode_state(state, player_id=0)
    cp_start = 52 + 52 + 5 + 1 + 6
    rel_cp0 = np.argmax(enc0[cp_start:cp_start + 6])
    assert rel_cp0 == (cp - 0) % 6, f"player_id=0: relative_cp={rel_cp0}, expected {(cp-0)%6}"

    enc3 = encode_state(state, player_id=3)
    rel_cp3 = np.argmax(enc3[cp_start:cp_start + 6])
    assert rel_cp3 == (cp - 3) % 6, f"player_id=3: relative_cp={rel_cp3}, expected {(cp-3)%6}"

    print(f"PASS: relative current_player -- player_id=0 => {rel_cp0}, player_id=3 => {rel_cp3}")


def test_hero_at_offset_zero():
    state = pkrs.State.from_seed(n_players=6, button=3, sb=1, bb=2, stake=200.0, seed=42)

    for player_id in [0, 3, 5]:
        enc = encode_state(state, player_id=player_id)
        per_player_start = 52 + 52 + 5 + 1 + 6 + 6
        hero_stake_enc = enc[per_player_start + 3]
        expected = float(state.players_state[player_id].stake) / (100.0 * state.bb)
        assert abs(hero_stake_enc - expected) < 0.01, \
            f"player_id={player_id}: hero offset 0 stake_enc={hero_stake_enc}, expected {expected}"

    print("PASS: hero always at offset 0 in player blocks")


def test_player_blocks_relative_order():
    state = pkrs.State.from_seed(n_players=6, button=3, sb=1, bb=2, stake=200.0, seed=42)
    player_id = 3

    enc = encode_state(state, player_id=player_id)
    per_player_start = 52 + 52 + 5 + 1 + 6 + 6
    per_player_width = 4

    for offset in range(6):
        p = (player_id + offset) % 6
        stake_enc = enc[per_player_start + offset * per_player_width + 3]
        expected = float(state.players_state[p].stake) / (100.0 * state.bb)
        assert abs(stake_enc - expected) < 0.01, \
            f"offset={offset} (player {p}): stake_enc={stake_enc}, expected {expected}"

    print("PASS: player blocks in hero-relative order")


def test_deterministic_with_relative():
    s1 = pkrs.State.from_seed(n_players=6, button=3, sb=1, bb=2, stake=200.0, seed=42)
    s2 = pkrs.State.from_seed(n_players=6, button=3, sb=1, bb=2, stake=200.0, seed=42)
    assert np.array_equal(encode_state(s1, 0), encode_state(s2, 0))
    assert np.array_equal(encode_state(s1, 3), encode_state(s2, 3))
    print("PASS: deterministic encoding with hero-relative")


def test_norm_unit_uses_hero_stake():
    state = pkrs.State.from_seed(n_players=6, button=0, sb=1, bb=2, stake=200.0, seed=42)

    enc = encode_state(state, player_id=0)
    pot_idx = 109
    pot_val = enc[pot_idx]
    norm_unit = 100.0 * state.bb
    assert abs(pot_val - state.pot / norm_unit) < 0.01

    enc3 = encode_state(state, player_id=3)
    pot_val3 = enc3[pot_idx]
    assert abs(pot_val3 - state.pot / norm_unit) < 0.01

    print("PASS: norm_unit consistent (100*bb) regardless of player_id")


if __name__ == '__main__':
    test_encoding_length()
    test_relative_button()
    test_relative_current_player()
    test_hero_at_offset_zero()
    test_player_blocks_relative_order()
    test_deterministic_with_relative()
    test_norm_unit_uses_hero_stake()
    print("\nAll hero-relative encoding tests passed!")
