"""Нейросеть и кодирование состояния для action-only Deep CFR."""

import numpy as np
import torch
import torch.nn as nn

from src.core.action_space import NUM_ACTIONS


VERBOSE = False


def set_verbose(verbose_mode):
    global VERBOSE
    VERBOSE = bool(verbose_mode)


class PokerNetwork(nn.Module):
    """Общая сеть, выдающая логиты или преимущества шести слотов действий."""

    def __init__(self, input_size=500, hidden_size=256, num_actions=NUM_ACTIONS):
        super().__init__()
        if int(num_actions) != NUM_ACTIONS:
            raise ValueError(f"PokerNetwork поддерживает только {NUM_ACTIONS} действий")
        self.base = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
        )
        self.action_head = nn.Linear(hidden_size, NUM_ACTIONS)
        nn.init.zeros_(self.action_head.weight)
        nn.init.zeros_(self.action_head.bias)

    def forward(self, x, opponent_features=None):
        del opponent_features
        return self.action_head(self.base(x))


def _build_suit_canonical_map(hand_cards, public_cards):
    suit_to_canonical = {}
    next_canonical = 0
    for card in [*hand_cards, *public_cards]:
        suit = int(card.suit)
        if suit not in suit_to_canonical:
            suit_to_canonical[suit] = next_canonical
            next_canonical += 1
    return suit_to_canonical


def encode_state(state, player_id=0):
    """Кодирует игровое состояние; четыре engine-действия остаются входным признаком."""
    encoded = []
    num_players = len(state.players_state)

    if VERBOSE:
        print(f"Кодирование: текущий={state.current_player}, улица={state.stage}, hero={player_id}")

    hand_cards = state.players_state[player_id].hand
    suit_map = _build_suit_canonical_map(hand_cards, state.public_cards)

    hand_enc = np.zeros(52, dtype=np.float32)
    for card in hand_cards:
        hand_enc[suit_map[int(card.suit)] * 13 + int(card.rank)] = 1.0
    encoded.append(hand_enc)

    community_enc = np.zeros(52, dtype=np.float32)
    for card in state.public_cards:
        community_enc[suit_map[int(card.suit)] * 13 + int(card.rank)] = 1.0
    encoded.append(community_enc)

    stage_enc = np.zeros(5, dtype=np.float32)
    stage_enc[int(state.stage)] = 1.0
    encoded.append(stage_enc)

    player_state = state.players_state[player_id]
    norm_unit = float(getattr(state, "bb", 0.0)) * 100.0
    if norm_unit < 1.0:
        norm_unit = float(player_state.stake + player_state.bet_chips + player_state.pot_chips)
    norm_unit = max(norm_unit, 1.0)

    encoded.append(np.asarray([state.pot / norm_unit], dtype=np.float32))

    button_enc = np.zeros(num_players, dtype=np.float32)
    button_enc[(int(state.button) - player_id) % num_players] = 1.0
    encoded.append(button_enc)

    current_player_enc = np.zeros(num_players, dtype=np.float32)
    current_player_enc[(int(state.current_player) - player_id) % num_players] = 1.0
    encoded.append(current_player_enc)

    for offset in range(num_players):
        player = state.players_state[(player_id + offset) % num_players]
        encoded.append(
            np.asarray(
                [
                    float(bool(player.active)),
                    player.bet_chips / norm_unit,
                    player.pot_chips / norm_unit,
                    player.stake / norm_unit,
                ],
                dtype=np.float32,
            )
        )

    encoded.append(np.asarray([state.min_bet / norm_unit], dtype=np.float32))
    total = float(state.pot) + float(player_state.stake)
    encoded.append(np.asarray([float(state.pot) / total if total > 0.0 else 0.0], dtype=np.float32))

    engine_actions = np.zeros(4, dtype=np.float32)
    for action_enum in state.legal_actions:
        engine_actions[int(action_enum)] = 1.0
    encoded.append(engine_actions)

    previous_action = np.zeros(5, dtype=np.float32)
    if state.from_action is not None:
        previous_action[int(state.from_action.action.action)] = 1.0
        previous_action[4] = state.from_action.action.amount / norm_unit
    encoded.append(previous_action)

    return np.concatenate(encoded, dtype=np.float32)


def encode_state_with_position(state, player_id=0):
    base = encode_state(state, player_id)
    one_hot = np.zeros(len(state.players_state), dtype=np.float32)
    one_hot[int(player_id)] = 1.0
    return np.concatenate([base, one_hot], dtype=np.float32)
