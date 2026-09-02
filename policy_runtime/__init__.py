"""Самодостаточный policy runtime Deep CFR: _light.pt weights → poker actions.

Dependencies: torch, numpy only. No pokers, no config.yaml, no deep_cfr.py.
"""
from policy_runtime.core import (
    PolicyRuntimeAgent, GameState, PlayerState,
    DictGameState, DictPlayerState, DictCard, DictFromAction,
    encode_state, PokerNetwork, INPUT_SIZE, NUM_ACTIONS,
)
