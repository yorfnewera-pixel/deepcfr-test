"""Нейросеть и кодирование состояния для action-only Deep CFR."""

import numpy as np
import torch
import torch.nn as nn

from src.core.action_space import NUM_ACTIONS


VERBOSE = False

LEGACY_ENCODING_VERSION = "legacy_v2"
HISTORY_SUMMARY_V3_ENCODING_VERSION = "history_summary_v3"

MONOLITHIC_ARCHITECTURE = "monolithic_v1"
CARD_CONTEXT_ARCHITECTURE = "card_context_v1"
NETWORK_ARCHITECTURES = (MONOLITHIC_ARCHITECTURE, CARD_CONTEXT_ARCHITECTURE)
CARD_FEATURE_SIZE = 109


def legacy_base_input_size(num_players):
    return 121 + 6 * int(num_players)


def history_summary_size(num_players):
    return 4 * (2 * int(num_players) + 8)


def encoder_input_size(num_players, encoding_version=LEGACY_ENCODING_VERSION, use_multi_agent=False):
    if encoding_version == LEGACY_ENCODING_VERSION:
        size = legacy_base_input_size(num_players)
    elif encoding_version == HISTORY_SUMMARY_V3_ENCODING_VERSION:
        size = legacy_base_input_size(num_players) + history_summary_size(num_players)
    else:
        raise ValueError(f"Неизвестная версия encoder: {encoding_version}")
    return size + (int(num_players) if use_multi_agent else 0)


def set_verbose(verbose_mode):
    global VERBOSE
    VERBOSE = bool(verbose_mode)


class PokerNetwork(nn.Module):
    """Общая сеть, выдающая логиты или преимущества шести слотов действий."""

    def __init__(
        self,
        input_size=500,
        hidden_size=256,
        num_actions=NUM_ACTIONS,
        architecture=MONOLITHIC_ARCHITECTURE,
    ):
        super().__init__()
        if int(num_actions) != NUM_ACTIONS:
            raise ValueError(f"PokerNetwork поддерживает только {NUM_ACTIONS} действий")
        if architecture not in NETWORK_ARCHITECTURES:
            raise ValueError(f"Неизвестная архитектура сети: {architecture}")
        if architecture == CARD_CONTEXT_ARCHITECTURE and int(input_size) < CARD_FEATURE_SIZE:
            raise ValueError(
                f"Архитектура {CARD_CONTEXT_ARCHITECTURE} требует не менее {CARD_FEATURE_SIZE} признаков"
            )

        self.architecture = architecture
        if architecture == MONOLITHIC_ARCHITECTURE:
            self.base = nn.Sequential(
                nn.Linear(input_size, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, hidden_size),
                nn.ReLU(),
            )
            action_input_size = hidden_size
        else:
            context_size = int(input_size) - CARD_FEATURE_SIZE
            self.card_encoder = nn.Sequential(
                nn.Linear(CARD_FEATURE_SIZE, hidden_size),
                nn.ReLU(),
            )
            self.context_encoder = nn.Sequential(
                nn.Linear(context_size, hidden_size),
                nn.ReLU(),
            )
            action_input_size = hidden_size * 2

        self.action_head = nn.Linear(action_input_size, NUM_ACTIONS)
        nn.init.zeros_(self.action_head.weight)
        nn.init.zeros_(self.action_head.bias)

    def forward(self, x, opponent_features=None):
        del opponent_features
        if self.architecture == MONOLITHIC_ARCHITECTURE:
            embedding = self.base(x)
        else:
            embedding = torch.cat((self.encode_cards(x), self.encode_context(x)), dim=-1)
        return self.action_head(embedding)

    def encode_cards(self, x):
        if self.architecture != CARD_CONTEXT_ARCHITECTURE:
            raise ValueError("Кодировщик карт доступен только для card_context_v1")
        return self.card_encoder(x[..., :CARD_FEATURE_SIZE])

    def encode_context(self, x):
        if self.architecture != CARD_CONTEXT_ARCHITECTURE:
            raise ValueError("Контекстный кодировщик доступен только для card_context_v1")
        return self.context_encoder(x[..., CARD_FEATURE_SIZE:])


def _build_suit_canonical_map(hand_cards, public_cards):
    suit_to_canonical = {}
    next_canonical = 0
    for card in [*hand_cards, *public_cards]:
        suit = int(card.suit)
        if suit not in suit_to_canonical:
            suit_to_canonical[suit] = next_canonical
            next_canonical += 1
    return suit_to_canonical


def _normalization_unit(state, player_id):
    player_state = state.players_state[player_id]
    norm_unit = float(getattr(state, "bb", 0.0)) * 100.0
    if norm_unit < 1.0:
        norm_unit = float(player_state.stake + player_state.bet_chips + player_state.pot_chips)
    return max(norm_unit, 1.0)


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
    norm_unit = _normalization_unit(state, player_id)

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


def _history_summary_v3(state, player_id, norm_unit):
    if not bool(getattr(state, "action_history_complete", False)):
        raise ValueError("history_summary_v3 требует полной публичной истории")

    num_players = len(state.players_state)
    streets = 4
    last_actor = np.full(streets, num_players, dtype=np.int64)
    last_action = np.full(streets, 4, dtype=np.int64)
    raise_actors = np.zeros((streets, num_players), dtype=np.float32)
    raise_count = np.zeros(streets, dtype=np.float32)
    raise_amount_total = np.zeros(streets, dtype=np.float32)

    for record in state.action_history:
        street = int(record.street)
        if street >= streets:
            continue
        last_actor[street] = (int(record.actor_id) - int(player_id)) % num_players
        last_action[street] = int(record.requested_action.action)
        if bool(record.is_effective_raise):
            actor = (int(record.actor_id) - int(player_id)) % num_players
            raise_actors[street, actor] = 1.0
            raise_count[street] += 1.0
            raise_amount_total[street] += float(record.applied_raise_increment) / norm_unit

    summary = []
    for street in range(streets):
        actor_enc = np.zeros(num_players + 1, dtype=np.float32)
        actor_enc[last_actor[street]] = 1.0
        action_enc = np.zeros(5, dtype=np.float32)
        action_enc[last_action[street]] = 1.0
        summary.extend((actor_enc, action_enc, raise_actors[street]))
        summary.append(np.asarray([raise_count[street] / max(num_players, 1)], dtype=np.float32))
        summary.append(np.asarray([raise_amount_total[street]], dtype=np.float32))
    return np.concatenate(summary, dtype=np.float32)


def encode_state_history_summary_v3(state, player_id=0):
    """Кодирует legacy infoset с actor-aware summary полной public history."""
    base = encode_state(state, player_id)
    norm_unit = _normalization_unit(state, player_id)
    return np.concatenate([base, _history_summary_v3(state, player_id, norm_unit)], dtype=np.float32)


def encode_state_for_version(state, player_id=0, encoding_version=LEGACY_ENCODING_VERSION):
    if encoding_version == LEGACY_ENCODING_VERSION:
        return encode_state(state, player_id)
    if encoding_version == HISTORY_SUMMARY_V3_ENCODING_VERSION:
        return encode_state_history_summary_v3(state, player_id)
    raise ValueError(f"Неизвестная версия encoder: {encoding_version}")


def encode_state_with_position(
    state,
    player_id=0,
    encoding_version=LEGACY_ENCODING_VERSION,
):
    base = encode_state_for_version(state, player_id, encoding_version)
    one_hot = np.zeros(len(state.players_state), dtype=np.float32)
    one_hot[int(player_id)] = 1.0
    return np.concatenate([base, one_hot], dtype=np.float32)
