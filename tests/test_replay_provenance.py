from types import SimpleNamespace

import numpy as np

from src.core.buffers import DuelingAdvantageBuffer
from src.core.replay_provenance import infoset_fingerprint
from src.training import train as train_mod


def _state(*, raise_amount: float = 4.0, suit_offset: int = 0):
    hero_cards = [
        SimpleNamespace(rank=12, suit=(0 + suit_offset) % 4),
        SimpleNamespace(rank=11, suit=(1 + suit_offset) % 4),
    ]
    board = [
        SimpleNamespace(rank=2, suit=(2 + suit_offset) % 4),
        SimpleNamespace(rank=7, suit=(3 + suit_offset) % 4),
        SimpleNamespace(rank=9, suit=(0 + suit_offset) % 4),
    ]
    action = SimpleNamespace(action=3, amount=raise_amount)
    record = SimpleNamespace(
        street=1,
        actor_id=1,
        requested_action=action,
        is_effective_raise=True,
        applied_raise_increment=raise_amount,
    )
    players = [
        SimpleNamespace(hand=hero_cards, active=True, bet_chips=4.0, pot_chips=8.0, stake=92.0),
        SimpleNamespace(hand=[], active=True, bet_chips=4.0, pot_chips=8.0, stake=92.0),
    ]
    return SimpleNamespace(
        stage=1,
        current_player=0,
        button=1,
        pot=16.0,
        min_bet=4.0,
        public_cards=board,
        players_state=players,
        action_history=[record],
        action_history_complete=True,
    )


def test_infoset_fingerprint_ignores_global_suit_renaming_but_keeps_full_action_history():
    first = infoset_fingerprint(_state())
    suit_renamed = infoset_fingerprint(_state(suit_offset=1))
    another_raise = infoset_fingerprint(_state(raise_amount=6.0))

    assert first.dtype == np.uint8
    assert first.shape == (16,)
    assert np.array_equal(first, suit_renamed)
    assert not np.array_equal(first, another_raise)


def test_dueling_buffer_replaces_provenance_in_the_same_reservoir_slot_as_target(monkeypatch):
    buffer = DuelingAdvantageBuffer(capacity=1, state_dim=2, provenance_enabled=True)
    monkeypatch.setattr(np.random, "randint", lambda *_args: 0)
    sample = dict(
        state=np.array([1.0, 2.0], dtype=np.float32),
        action_values=np.zeros(6, dtype=np.float32),
        state_value=np.float32(0.0),
        regrets=np.zeros(6, dtype=np.float32),
        mask=np.ones(6, dtype=np.float32),
    )
    first = np.full(16, 1, dtype=np.uint8)
    second = np.full(16, 2, dtype=np.uint8)

    buffer.add(**sample, iteration=1, provenance=first)
    replacement = {**sample, "state": np.array([3.0, 4.0], dtype=np.float32)}
    buffer.add(**replacement, iteration=2, provenance=second)

    assert np.array_equal(buffer._states[0], np.array([3.0, 4.0], dtype=np.float32))
    assert np.array_equal(buffer.provenances(), second.reshape(1, -1))


def test_hu_checkpoint_payload_round_trip_preserves_replay_provenance():
    source = DuelingAdvantageBuffer(capacity=2, state_dim=2, provenance_enabled=True)
    source.add(
        np.array([1.0, 2.0], dtype=np.float32),
        np.zeros(6, dtype=np.float32),
        0.0,
        np.zeros(6, dtype=np.float32),
        np.ones(6, dtype=np.float32),
        1,
        np.arange(16, dtype=np.uint8),
    )
    restored = DuelingAdvantageBuffer(capacity=2, state_dim=2, provenance_enabled=True)

    train_mod._restore_d2cfr_advantage_buffer(
        restored, train_mod._d2cfr_advantage_buffer_payload(source)
    )

    assert np.array_equal(restored.provenances(), np.arange(16, dtype=np.uint8).reshape(1, -1))
