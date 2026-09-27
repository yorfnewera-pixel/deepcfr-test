"""Компактный неизменяемый отпечаток полного HU infoset для replay-аудита."""
from __future__ import annotations

import hashlib
import json

import numpy as np


_FINGERPRINT_SIZE = 16


def _canonical_cards(cards, suit_map: dict[int, int]) -> list[tuple[int, int]]:
    return sorted((int(card.rank), suit_map[int(card.suit)]) for card in cards)


def infoset_fingerprint(state, player_id: int = 0) -> np.ndarray:
    """Хеширует видимые игроку карты и полную публичную историю без влияния имён мастей."""
    if not bool(getattr(state, "action_history_complete", False)):
        raise ValueError("Replay provenance требует полной публичной истории")
    actor_id = int(player_id)
    players = state.players_state
    if actor_id < 0 or actor_id >= len(players):
        raise ValueError("Replay provenance получил actor вне players_state")

    hero_cards = players[actor_id].hand
    public_cards = state.public_cards
    suit_map: dict[int, int] = {}
    for card in [*hero_cards, *public_cards]:
        suit = int(card.suit)
        if suit not in suit_map:
            suit_map[suit] = len(suit_map)

    history = []
    for record in state.action_history:
        action = record.requested_action
        history.append((
            int(record.street), int(record.actor_id), int(action.action), float(action.amount),
            bool(record.is_effective_raise), float(record.applied_raise_increment),
        ))
    public_players = [
        (bool(player.active), float(player.bet_chips), float(player.pot_chips), float(player.stake))
        for player in players
    ]
    payload = {
        "version": 1,
        "actor": actor_id,
        "hero_cards": _canonical_cards(hero_cards, suit_map),
        "public_cards": _canonical_cards(public_cards, suit_map),
        "stage": int(state.stage),
        "current_player": int(state.current_player),
        "button": int(state.button),
        "pot": float(state.pot),
        "min_bet": float(state.min_bet),
        "players": public_players,
        "history": history,
    }
    serialized = json.dumps(payload, separators=(",", ":"), sort_keys=True).encode("utf-8")
    return np.frombuffer(
        hashlib.blake2b(serialized, digest_size=_FINGERPRINT_SIZE).digest(), dtype=np.uint8
    ).copy()


__all__ = ["infoset_fingerprint"]
