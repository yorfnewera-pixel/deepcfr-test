"""Независимый runtime policy — без зависимости от игрового движка.

Dependencies: torch, numpy. No pokers dependency.
Input: dict via choose_action_from_dict() or GameState Protocol via choose_action().
Output: {"action_type": int} — adapter converts the fixed action slot to engine Action.
"""
import json
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
from typing import Protocol, runtime_checkable, Optional


LEGACY_ENCODING_VERSION = "legacy_v2"
HISTORY_SUMMARY_V3_ENCODING_VERSION = "history_summary_v3"
CHECKPOINT_FORMAT_VERSION = 7
GAME_RULES_VERSION = "holdem_standard_hu_v2"
INPUT_SIZE = 157
NUM_ACTIONS = 6
DEFAULT_HIDDEN = 256
MONOLITHIC_ARCHITECTURE = "monolithic_v1"
CARD_CONTEXT_ARCHITECTURE = "card_context_v1"
CARD_CONTEXT_V2_ARCHITECTURE = "card_context_v2"
CARD_CONTEXT_ARCHITECTURES = (CARD_CONTEXT_ARCHITECTURE, CARD_CONTEXT_V2_ARCHITECTURE)
NETWORK_ARCHITECTURES = (MONOLITHIC_ARCHITECTURE, *CARD_CONTEXT_ARCHITECTURES)
CARD_FEATURE_SIZE = 109


def _validate_v2_fusion_contract(checkpoint, config, strategy_state):
    card_weight = strategy_state.get("card_encoder.0.weight")
    context_weight = strategy_state.get("context_encoder.0.weight")
    fusion_input_weight = strategy_state.get("fusion.0.weight")
    fusion_input_bias = strategy_state.get("fusion.0.bias")
    fusion_output_weight = strategy_state.get("fusion.2.weight")
    fusion_output_bias = strategy_state.get("fusion.2.bias")
    action_weight = strategy_state.get("action_head.weight")
    if not all(torch.is_tensor(value) for value in (
        card_weight,
        context_weight,
        fusion_input_weight,
        fusion_input_bias,
        fusion_output_weight,
        fusion_output_bias,
        action_weight,
    )):
        raise ValueError("Checkpoint не содержит полные v2 fusion веса")

    hidden_size = int(card_weight.shape[0])
    input_size = int(card_weight.shape[1]) + int(context_weight.shape[1])
    expected_metadata = {
        "network_architecture": CARD_CONTEXT_V2_ARCHITECTURE,
        "card_feature_size": CARD_FEATURE_SIZE,
        "input_size": input_size,
        "hidden_size": hidden_size,
        "fusion_input_size": hidden_size * 2,
        "fusion_output_size": hidden_size,
        "num_actions": NUM_ACTIONS,
    }
    for key, expected_value in expected_metadata.items():
        if checkpoint.get(key, config.get(key)) != expected_value:
            raise ValueError(f"Checkpoint имеет несовместимые v2 fusion метаданные: {key}")

    expected_shapes = {
        "card_encoder.0.weight": (hidden_size, CARD_FEATURE_SIZE),
        "context_encoder.0.weight": (hidden_size, input_size - CARD_FEATURE_SIZE),
        "fusion.0.weight": (hidden_size, hidden_size * 2),
        "fusion.0.bias": (hidden_size,),
        "fusion.2.weight": (hidden_size, hidden_size),
        "fusion.2.bias": (hidden_size,),
        "action_head.weight": (NUM_ACTIONS, hidden_size),
    }
    for key, expected_shape in expected_shapes.items():
        value = strategy_state[key]
        if tuple(value.shape) != expected_shape:
            raise ValueError(f"Checkpoint имеет несовместимые v2 fusion веса: {key}")
    return input_size, hidden_size
@runtime_checkable
class PlayerState(Protocol):
    hand: list
    active: bool
    bet_chips: float
    pot_chips: float
    stake: float
    last_stage_action: object | None


@runtime_checkable
class GameState(Protocol):
    bb: float
    pot: float
    min_bet: float
    button: int
    current_player: int
    stage: int
    legal_actions: list
    public_cards: list
    from_action: Optional[object]
    players_state: list
    last_raise_increment: float
    action_history_complete: bool
    action_history: list


class DictCard:
    __slots__ = ('suit', 'rank')

    def __init__(self, d):
        if isinstance(d, dict):
            self.suit = int(d.get('suit', 0))
            self.rank = int(d.get('rank', 0))
        else:
            self.suit = int(d.suit)
            self.rank = int(d.rank)


class _DictAction:
    __slots__ = ('action', 'amount')

    def __init__(self, action_type, amount):
        self.action = int(action_type)
        self.amount = float(amount)

    def __int__(self):
        return self.action


class _DictActionRecord:
    __slots__ = ('action', 'amount')

    def __init__(self, action_type, amount):
        self.action = _DictAction(action_type, amount)
        self.amount = float(amount)


class DictFromAction:
    __slots__ = ('action',)

    def __init__(self, fa):
        if isinstance(fa, (list, tuple)):
            self.action = _DictActionRecord(int(fa[0]), float(fa[1]))
        elif isinstance(fa, dict):
            self.action = _DictActionRecord(
                int(fa.get('action', 0)), float(fa.get('amount', 0.0)))
        else:
            self.action = _DictActionRecord(
                int(fa.action.action), float(fa.action.amount))


class DictPlayerState:
    __slots__ = ('hand', 'active', 'bet_chips', 'pot_chips', 'stake', 'last_stage_action')

    def __init__(self, d):
        if isinstance(d, dict):
            self.hand = [DictCard(c) for c in d.get('hand', [])]
            self.active = bool(d.get('active', True))
            self.bet_chips = float(d.get('bet_chips', 0.0))
            self.pot_chips = float(d.get('pot_chips', 0.0))
            self.stake = float(d.get('stake', 0.0))
            self.last_stage_action = d.get('last_stage_action')
        else:
            self.hand = [DictCard(c) for c in d.hand]
            self.active = bool(d.active)
            self.bet_chips = float(d.bet_chips)
            self.pot_chips = float(d.pot_chips)
            self.stake = float(d.stake)
            self.last_stage_action = getattr(d, 'last_stage_action', None)


class DictGameState:
    __slots__ = ('bb', 'pot', 'min_bet', 'button', 'current_player', 'stage',
                 'legal_actions', 'public_cards', 'from_action',
                 'players_state', 'last_raise_increment', 'action_history_complete',
                 'action_history')

    def __init__(self, d):
        self.bb = float(d.get('bb', 2.0))
        self.pot = float(d.get('pot', 0.0))
        self.min_bet = float(d.get('min_bet', 0.0))
        self.button = int(d.get('button', 0))
        self.current_player = int(d.get('current_player', 0))
        self.stage = int(d.get('stage', 0))
        self.legal_actions = list(d.get('legal_actions', [1]))
        self.public_cards = [DictCard(c) for c in d.get('public_cards', [])]
        self.last_raise_increment = float(d.get('last_raise_increment', 0.0))
        fa = d.get('from_action')
        self.from_action = DictFromAction(fa) if fa else None
        self.players_state = [DictPlayerState(p)
                              for p in d.get('players', [])]
        self.action_history_complete = bool(d.get('action_history_complete', False))
        self.action_history = list(d.get('action_history', []))


def _build_suit_canonical_map(hand_cards, public_cards):
    suit_to_canonical = {}
    next_canonical = 0
    for card in hand_cards:
        s = int(card.suit)
        if s not in suit_to_canonical:
            suit_to_canonical[s] = next_canonical
            next_canonical += 1
    for card in public_cards:
        s = int(card.suit)
        if s not in suit_to_canonical:
            suit_to_canonical[s] = next_canonical
            next_canonical += 1
    return suit_to_canonical


def encode_state(state: GameState, player_id: int = 0) -> np.ndarray:
    encoded = []
    num_players = len(state.players_state)

    hand_cards = state.players_state[player_id].hand
    suit_map = _build_suit_canonical_map(hand_cards, state.public_cards)

    hand_enc = np.zeros(52)
    for card in hand_cards:
        canonical_suit = suit_map[int(card.suit)]
        card_idx = canonical_suit * 13 + int(card.rank)
        hand_enc[card_idx] = 1
    encoded.append(hand_enc)

    community_enc = np.zeros(52)
    for card in state.public_cards:
        canonical_suit = suit_map[int(card.suit)]
        card_idx = canonical_suit * 13 + int(card.rank)
        community_enc[card_idx] = 1
    encoded.append(community_enc)

    stage_enc = np.zeros(5)
    stage_enc[int(state.stage)] = 1
    encoded.append(stage_enc)

    norm_unit = _normalization_unit(state, player_id)

    pot_enc = [float(state.pot) / norm_unit]
    encoded.append(pot_enc)

    button_enc = np.zeros(num_players)
    button_enc[(int(state.button) - player_id) % num_players] = 1
    encoded.append(button_enc)

    current_player_enc = np.zeros(num_players)
    current_player_enc[(int(state.current_player) - player_id) % num_players] = 1
    encoded.append(current_player_enc)

    for offset in range(num_players):
        p = (player_id + offset) % num_players
        ps = state.players_state[p]
        active_enc = [1.0 if ps.active else 0.0]
        bet_enc = [float(ps.bet_chips) / norm_unit]
        pot_chips_enc = [float(ps.pot_chips) / norm_unit]
        stake_enc = [float(ps.stake) / norm_unit]
        encoded.append(np.concatenate([active_enc, bet_enc, pot_chips_enc,
                                       stake_enc]))

    min_bet_enc = [float(state.min_bet) / norm_unit]
    encoded.append(min_bet_enc)

    player_stake_enc = float(state.players_state[player_id].stake) / norm_unit
    player_pot_enc = float(state.pot) / norm_unit
    pot_commitment = player_pot_enc / (player_pot_enc + player_stake_enc) if (player_pot_enc + player_stake_enc) > 0 else 0.0
    encoded.append([pot_commitment])

    legal_actions_enc = np.zeros(4)
    for action_enum in state.legal_actions:
        legal_actions_enc[int(action_enum)] = 1
    encoded.append(legal_actions_enc)

    prev_action_enc = np.zeros(4 + 1)
    if state.from_action is not None:
        prev_action_enc[int(state.from_action.action.action)] = 1
        prev_action_enc[4] = float(state.from_action.action.amount) / norm_unit
    encoded.append(prev_action_enc)

    return np.concatenate(encoded)


def encode_state_with_position(state: GameState, player_id: int = 0) -> np.ndarray:
    base = encode_state(state, player_id)
    num_players = len(state.players_state)
    one_hot = np.zeros(num_players, dtype=np.float32)
    one_hot[int(player_id)] = 1.0
    return np.concatenate([base, one_hot])


def _normalization_unit(state: GameState, player_id: int) -> float:
    norm_unit = float(state.bb) * 100.0
    if norm_unit < 1.0:
        player = state.players_state[player_id]
        norm_unit = float(player.stake + player.bet_chips + player.pot_chips)
    return max(norm_unit, 1.0)


def _history_record_value(record, field: str):
    if isinstance(record, dict):
        return record[field]
    return getattr(record, field)


def _history_action_type(record) -> int:
    action = _history_record_value(record, "requested_action")
    if isinstance(action, dict):
        return int(action["action"])
    return int(action.action)


def encode_state_history_summary_v3(state: GameState, player_id: int = 0) -> np.ndarray:
    if not bool(getattr(state, "action_history_complete", False)):
        raise ValueError("history_summary_v3 требует полной публичной истории")

    num_players = len(state.players_state)
    norm_unit = _normalization_unit(state, player_id)
    last_actor = np.full(4, num_players, dtype=np.int64)
    last_action = np.full(4, 4, dtype=np.int64)
    raise_actors = np.zeros((4, num_players), dtype=np.float32)
    raise_count = np.zeros(4, dtype=np.float32)
    raise_amount_total = np.zeros(4, dtype=np.float32)
    for record in state.action_history:
        street = int(_history_record_value(record, "street"))
        if street < 0 or street >= 4:
            continue
        actor = (int(_history_record_value(record, "actor_id")) - int(player_id)) % num_players
        last_actor[street] = actor
        last_action[street] = _history_action_type(record)
        if bool(_history_record_value(record, "is_effective_raise")):
            raise_actors[street, actor] = 1.0
            raise_count[street] += 1.0
            raise_amount_total[street] += float(_history_record_value(record, "applied_raise_increment")) / norm_unit

    summary = []
    for street in range(4):
        actor_enc = np.zeros(num_players + 1, dtype=np.float32)
        actor_enc[last_actor[street]] = 1.0
        action_enc = np.zeros(5, dtype=np.float32)
        action_enc[last_action[street]] = 1.0
        summary.extend((actor_enc, action_enc, raise_actors[street]))
        summary.append(np.asarray([raise_count[street] / max(num_players, 1)], dtype=np.float32))
        summary.append(np.asarray([raise_amount_total[street]], dtype=np.float32))
    return np.concatenate([encode_state(state, player_id), *summary]).astype(np.float32, copy=False)


def encode_state_for_version(
    state: GameState,
    player_id: int,
    encoding_version: str,
) -> np.ndarray:
    if encoding_version == LEGACY_ENCODING_VERSION:
        return encode_state(state, player_id).astype(np.float32, copy=False)
    if encoding_version == HISTORY_SUMMARY_V3_ENCODING_VERSION:
        return encode_state_history_summary_v3(state, player_id)
    raise ValueError(f"Неизвестная версия encoder: {encoding_version}")


def encoder_input_size(num_players: int, encoding_version: str, use_multi_agent: bool) -> int:
    base_size = 121 + 6 * int(num_players)
    if encoding_version == HISTORY_SUMMARY_V3_ENCODING_VERSION:
        base_size += 4 * (2 * int(num_players) + 8)
    elif encoding_version != LEGACY_ENCODING_VERSION:
        raise ValueError(f"Неизвестная версия encoder: {encoding_version}")
    return base_size + (int(num_players) if use_multi_agent else 0)


class PokerNetwork(nn.Module):
    def __init__(self, input_size=INPUT_SIZE, hidden_size=DEFAULT_HIDDEN,
                 num_actions=NUM_ACTIONS, architecture=MONOLITHIC_ARCHITECTURE):
        super().__init__()
        if architecture not in NETWORK_ARCHITECTURES:
            raise ValueError(f"Неизвестная архитектура сети: {architecture}")
        if architecture in CARD_CONTEXT_ARCHITECTURES and int(input_size) < CARD_FEATURE_SIZE:
            raise ValueError(
                f"Архитектура {architecture} требует не менее {CARD_FEATURE_SIZE} признаков"
            )

        self.num_actions = num_actions
        self.architecture = architecture
        if architecture == MONOLITHIC_ARCHITECTURE:
            self.base = nn.Sequential(
                nn.Linear(input_size, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, hidden_size),
                nn.ReLU()
            )
            action_input_size = hidden_size
        else:
            self.card_encoder = nn.Sequential(
                nn.Linear(CARD_FEATURE_SIZE, hidden_size),
                nn.ReLU(),
            )
            self.context_encoder = nn.Sequential(
                nn.Linear(int(input_size) - CARD_FEATURE_SIZE, hidden_size),
                nn.ReLU(),
            )
            if architecture == CARD_CONTEXT_ARCHITECTURE:
                action_input_size = hidden_size * 2
            else:
                self.fusion = nn.Sequential(
                    nn.Linear(hidden_size * 2, hidden_size),
                    nn.ReLU(),
                    nn.Linear(hidden_size, hidden_size),
                    nn.ReLU(),
                )
                action_input_size = hidden_size
        self.action_head = nn.Linear(action_input_size, num_actions)
        self._init_output_layers()

    def _init_output_layers(self):
        nn.init.zeros_(self.action_head.weight)
        nn.init.zeros_(self.action_head.bias)

    def forward(self, x):
        if self.architecture == MONOLITHIC_ARCHITECTURE:
            features = self.base(x)
        else:
            features = self._encode_card_context(x)
        return self.action_head(features)

    def _encode_card_context(self, x):
        features = torch.cat((self.encode_cards(x), self.encode_context(x)), dim=-1)
        if self.architecture == CARD_CONTEXT_V2_ARCHITECTURE:
            return self.fusion(features)
        return features

    def encode_cards(self, x):
        if self.architecture not in CARD_CONTEXT_ARCHITECTURES:
            raise ValueError("Кодировщик карт доступен только для card_context архитектур")
        return self.card_encoder(x[..., :CARD_FEATURE_SIZE])

    def encode_context(self, x):
        if self.architecture not in CARD_CONTEXT_ARCHITECTURES:
            raise ValueError("Контекстный кодировщик доступен только для card_context архитектур")
        return self.context_encoder(x[..., CARD_FEATURE_SIZE:])


FEATURE_SPEC = {
    0: "hand_enc", 52: "community_enc", 104: "stage_enc",
    109: "pot_enc", 110: "button_enc", 116: "current_player_enc",
    122: "per_player_0", 126: "per_player_1", 130: "per_player_2",
    134: "per_player_3", 138: "per_player_4", 142: "per_player_5",
    146: "min_bet_enc", 147: "pot_commitment",
    148: "legal_actions_enc", 152: "prev_action_enc",
}

PER_PLAYER_SPEC = {
    0: "active", 1: "bet_enc", 2: "pot_chips_enc",
    3: "stake_enc",
}


class PolicyRuntimeAgent:
    ACTION_FOLD = 0
    ACTION_CHECK = 1
    ACTION_CALL = 2
    ACTION_RAISE_HALF_POT = 3
    ACTION_RAISE_POT = 4
    ACTION_ALL_IN = 5

    def __init__(self, checkpoint_path: str, device: str = 'cpu',
                 min_action_prob: float | None = None):
        self.device = torch.device(device)
        self.checkpoint_path = checkpoint_path
        self._min_action_prob_override = min_action_prob
        self._load(checkpoint_path)

    def _load(self, path: str):
        checkpoint = torch.load(path, map_location=self.device,
                                weights_only=False)

        if 'strategy_net' not in checkpoint:
            raise ValueError(
                f"strategy_net не найден в чекпоинте {path}. "
                f"Доступные ключи: {list(checkpoint.keys())}")

        if checkpoint.get('checkpoint_format_version') != CHECKPOINT_FORMAT_VERSION:
            raise ValueError("Нужен checkpoint формата history_summary_v3")
        if checkpoint.get('game_rules_version') != GAME_RULES_VERSION:
            raise ValueError("Checkpoint создан до исправления правил HU")
        if checkpoint.get('action_space_version') != 'six_fixed_v2':
            raise ValueError("Checkpoint имеет другое пространство действий")

        strategy_sd = checkpoint['strategy_net']
        if not isinstance(strategy_sd, dict):
            raise ValueError("strategy_net имеет некорректный формат")
        cfg = checkpoint.get('config', {})
        if not isinstance(cfg, dict):
            cfg = {}
        if 'network_architecture' in checkpoint:
            architecture = checkpoint['network_architecture']
        elif 'network_architecture' in cfg:
            architecture = cfg['network_architecture']
        elif 'card_encoder.0.weight' in strategy_sd:
            raise ValueError("Checkpoint card_context_v1 не содержит метаданные архитектуры")
        else:
            architecture = MONOLITHIC_ARCHITECTURE
        if architecture not in NETWORK_ARCHITECTURES:
            raise ValueError("Checkpoint имеет некорректную архитектуру сети")
        if architecture == CARD_CONTEXT_ARCHITECTURE:
            card_feature_size = checkpoint.get('card_feature_size', cfg.get('card_feature_size'))
            if card_feature_size != CARD_FEATURE_SIZE:
                raise ValueError("Checkpoint имеет несовместимый размер card-признаков")
            card_weight = strategy_sd.get('card_encoder.0.weight')
            context_weight = strategy_sd.get('context_encoder.0.weight')
            if card_weight is None or context_weight is None:
                raise ValueError("Checkpoint не содержит веса card_context_v1")
            self.hidden_size = int(card_weight.shape[0])
            self.input_size = int(card_weight.shape[1]) + int(context_weight.shape[1])
        elif architecture == CARD_CONTEXT_V2_ARCHITECTURE:
            self.input_size, self.hidden_size = _validate_v2_fusion_contract(
                checkpoint,
                cfg,
                strategy_sd,
            )
        else:
            first_weight_key = 'base.0.weight'
            if first_weight_key not in strategy_sd:
                raise ValueError(
                    f"Ожидался ключ '{first_weight_key}' в strategy_net. "
                    f"Несовместимая архитектура.")
            self.hidden_size = strategy_sd[first_weight_key].shape[0]
            self.input_size = strategy_sd[first_weight_key].shape[1]
        self.num_players = int(checkpoint.get('num_players', 6))
        self.use_multi_agent = bool(checkpoint.get('use_multi_agent_advantage',
            cfg.get('use_multi_agent_advantage', False)))
        self.encoding_version = checkpoint.get(
            'encoding_version', cfg.get('encoding_version'))
        if not isinstance(self.encoding_version, str):
            raise ValueError("Checkpoint не содержит версию encoder")
        self.base_input_size = encoder_input_size(
            self.num_players, self.encoding_version, self.use_multi_agent)
        self.strategy_actor_conditioned = bool(checkpoint.get(
            'strategy_actor_conditioned', cfg.get('strategy_actor_conditioned', False)
        ))
        self.strategy_actor_count = int(checkpoint.get(
            'strategy_actor_count', cfg.get('strategy_actor_count', 0)
        ))
        if self.strategy_actor_conditioned:
            if self.num_players != 2 or self.strategy_actor_count != 2:
                raise ValueError("HU strategy actor-conditioning требует ровно двух игроков")
        elif self.strategy_actor_count != 0:
            raise ValueError("Checkpoint содержит actor_count без actor-conditioning")
        expected_input_size = self.base_input_size + self.strategy_actor_count
        if (
            self.input_size != expected_input_size
            or int(checkpoint.get('encoder_input_size', -1)) != self.base_input_size
            or int(checkpoint.get('base_encoder_input_size', self.base_input_size)) != self.base_input_size
            or int(checkpoint.get('strategy_input_size', self.input_size)) != self.input_size
        ):
            raise ValueError("Checkpoint имеет несовместимый размер входа encoder")

        self.iteration = int(checkpoint.get('iteration', 0))

        self.strategy_net = PokerNetwork(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            num_actions=NUM_ACTIONS,
            architecture=architecture,
        ).to(self.device)
        self.strategy_net.load_state_dict(strategy_sd, strict=True)
        self.strategy_net.eval()
        checkpoint_min_action_prob = cfg.get(
            'policy_runtime_min_action_prob',
            cfg.get('inference_min_action_prob', 0.0),
        )
        self.policy_runtime_min_action_prob = float(
            checkpoint_min_action_prob
            if self._min_action_prob_override is None
            else self._min_action_prob_override
        )

    def validate(self) -> list[str]:
        errors = []

        expected_size = self.base_input_size + self.strategy_actor_count
        if self.input_size != expected_size:
            errors.append(
                f"input_size={self.input_size}, ожидалось {expected_size}. "
                f"Чекпоинт несовместим с текущим encode_state.")

        ah_key = 'action_head.weight'
        if ah_key in self.strategy_net.state_dict():
            ah = self.strategy_net.state_dict()[ah_key]
            if ah.shape[0] != NUM_ACTIONS:
                errors.append(
                    f"action_head: {ah.shape[0]} действий, "
                    f"ожидалось {NUM_ACTIONS}")

        if not errors:
            errors.append("OK: чекпоинт совместим")

        return errors

    def get_legal_action_mask(self, state: GameState) -> np.ndarray:
        mask = np.zeros(NUM_ACTIONS, dtype=np.float32)
        if any(int(a) == 0 for a in state.legal_actions):
            mask[self.ACTION_FOLD] = 1.0
        if any(int(a) == 1 for a in state.legal_actions):
            mask[self.ACTION_CHECK] = 1.0
        if any(int(a) == 2 for a in state.legal_actions):
            mask[self.ACTION_CALL] = 1.0
        if any(int(a) == 3 for a in state.legal_actions):
            player = state.players_state[int(state.current_player)]
            call_amount = max(0.0, float(state.min_bet) - float(player.bet_chips))
            remaining = max(0.0, float(player.stake) - call_amount)
            min_raise = max(1.0, float(getattr(state, 'last_raise_increment', 0.0) or getattr(state, 'bb', 1.0)))
            pot = float(state.pot)
            has_raise_on_current_street = any(
                player_state.last_stage_action is not None
                and int(player_state.last_stage_action) == 3
                for player_state in state.players_state
            )
            if not has_raise_on_current_street and 0.5 * pot >= min_raise and 0.5 * pot < remaining:
                mask[self.ACTION_RAISE_HALF_POT] = 1.0
            if pot >= min_raise and pot < remaining:
                mask[self.ACTION_RAISE_POT] = 1.0
            if remaining > 0.0:
                mask[self.ACTION_ALL_IN] = 1.0

        return mask

    def choose_action(self, state: GameState, player_id: int = 0,
                      deterministic: bool = False) -> int:
        legal_mask = self.get_legal_action_mask(state)
        legal_action_types = [a for a in range(NUM_ACTIONS) if legal_mask[a] > 0]

        if not legal_action_types:
            raise ValueError("В текущем состоянии нет допустимых action-слотов")

        state_vec = encode_state_for_version(state, player_id, self.encoding_version)
        if self.use_multi_agent:
            one_hot = np.zeros(len(state.players_state), dtype=np.float32)
            one_hot[int(player_id)] = 1.0
            state_vec = np.concatenate([state_vec, one_hot], dtype=np.float32)
        if self.strategy_actor_conditioned:
            actor = int(player_id)
            if actor < 0 or actor >= self.strategy_actor_count:
                raise ValueError("player_id не входит в actor-conditioning strategy")
            actor_one_hot = np.zeros(self.strategy_actor_count, dtype=np.float32)
            actor_one_hot[actor] = 1.0
            state_vec = np.concatenate([state_vec, actor_one_hot], dtype=np.float32)
        state_tensor = torch.FloatTensor(state_vec).unsqueeze(0).to(self.device)
        mask_tensor = torch.FloatTensor(legal_mask).unsqueeze(0).to(self.device)

        with torch.inference_mode():
            action_logits = self.strategy_net(state_tensor)
            masked_logits = torch.where(
                mask_tensor == 1,
                action_logits[:, :NUM_ACTIONS],
                torch.tensor(-1e20, device=self.device))
            action_probs = F.softmax(masked_logits, dim=1)[0].cpu().numpy()
            if self.policy_runtime_min_action_prob > 0:
                keep = action_probs >= self.policy_runtime_min_action_prob
                if keep.sum() > 0:
                    action_probs = action_probs * keep
                    action_probs = action_probs / action_probs.sum()
            total_act = action_probs.sum()
            if total_act > 1e-8:
                action_probs = np.clip(action_probs, 0.0, None)
                action_probs = action_probs / action_probs.sum()

        if deterministic:
            best_idx = int(np.argmax([action_probs[a] for a in legal_action_types]))
            action_type = legal_action_types[best_idx]
        else:
            legal_probs = np.array([action_probs[a] for a in legal_action_types])
            total = legal_probs.sum()
            if total > 0:
                legal_probs = legal_probs / total
            else:
                legal_probs = np.ones(len(legal_action_types)) / len(legal_action_types)
            action_idx = np.random.choice(len(legal_action_types), p=legal_probs)
            action_type = legal_action_types[action_idx]

        return int(action_type)

    def choose_action_from_dict(self, data_dict: dict,
                                player_id: int = 0,
                                deterministic: bool = False) -> dict:
        gs = DictGameState(data_dict)
        action_type = self.choose_action(gs, player_id, deterministic)
        return {"action_type": action_type}

    def export_spec(self, path: Optional[str] = None) -> dict:
        spec = {
            "input_size": self.input_size,
            "base_input_size": self.base_input_size,
            "num_players": self.num_players,
            "encoding_version": self.encoding_version,
            "num_actions": NUM_ACTIONS,
            "hidden_size": self.hidden_size,
            "action_labels": ["fold", "check", "call", "raise_0.5pot", "raise_1pot", "all_in"],
            "iteration": self.iteration,
            "strategy_actor_conditioned": self.strategy_actor_conditioned,
            "strategy_actor_count": self.strategy_actor_count,
            "normalization": "100*bb",
            "feature_layout": FEATURE_SPEC,
            "per_player_layout": PER_PLAYER_SPEC,
            "checkpoint_path": self.checkpoint_path,
        }

        if path:
            with open(path, 'w', encoding='utf-8') as f:
                json.dump(spec, f, indent=2, ensure_ascii=False)

        return spec


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Deep CFR Policy Runtime — standalone")
    parser.add_argument("checkpoint", help="Путь к _light.pt")
    parser.add_argument("--input", required=True,
                        help="Путь к JSON файлу состояния стола")
    parser.add_argument("--player", type=int, default=0,
                        help="ID игрока (по умолчанию 0)")
    parser.add_argument("--deterministic", action="store_true",
                        help="Детерминированный выбор (argmax)")
    args = parser.parse_args()

    agent = PolicyRuntimeAgent(args.checkpoint)

    errors = agent.validate()
    for e in errors:
        print(f"  Валидация: {e}", file=__import__('sys').stderr)

    with open(args.input, 'r', encoding='utf-8') as f:
        data_dict = json.load(f)

    result = agent.choose_action_from_dict(
        data_dict, player_id=args.player, deterministic=args.deterministic)

    print(json.dumps(result, ensure_ascii=False))
