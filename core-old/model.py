# src/code/model.py
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

VERBOSE = False

def set_verbose(verbose_mode):
    global VERBOSE
    VERBOSE = verbose_mode


class PokerNetwork(nn.Module):
    """Сеть action-policy: предсказывает только action_logits.

    Sizing вынесен в отдельную SizingNetwork (см. ниже) — изоляция голов
    устраняет конфликт градиентов между PG-обучением sizing и MSE-обучением
    advantage. Раньше PG-шум в shared base разрушал regression-таргет
    advantage_net и убивал обучение action-policy.
    """

    def __init__(self, input_size=500, hidden_size=256, num_actions=4):
        super().__init__()
        self.base = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU()
        )
        self.action_head = nn.Linear(hidden_size, num_actions)
        self._init_output_layers()

    def _init_output_layers(self):
        nn.init.zeros_(self.action_head.weight)
        nn.init.zeros_(self.action_head.bias)

    def forward(self, x, opponent_features=None):
        features = self.base(x)
        return self.action_head(features)


class StrategySizingNet(nn.Module):
    """Sizing-сеть для инференса над фиксированной сеткой анкеров.

    Возвращает slot_logits[K] и legacy scalar_bet. Покрытие sizing-групп
    обеспечивается traversal-структурой, поэтому отдельной bucket-head нет.
    """

    def __init__(self, input_size, hidden_size=128, num_sizes=15, num_buckets=3):
        super().__init__()
        self.num_sizes = num_sizes
        self.num_buckets = num_buckets
        self.base = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
        )
        self.anchor_head = nn.Linear(hidden_size, num_sizes)
        self.sizing_head = nn.Linear(hidden_size, 1)
        self._init_output_layers()

    def _init_output_layers(self):
        nn.init.normal_(self.anchor_head.weight, mean=0.0, std=0.01)
        nn.init.zeros_(self.anchor_head.bias)
        nn.init.zeros_(self.sizing_head.weight)
        nn.init.constant_(self.sizing_head.bias, 0.01)

    def forward(self, x):
        features = self.base(x)
        slot_logits = self.anchor_head(features)
        scalar_bet = self.sizing_head(features)
        return slot_logits, scalar_bet


class SizingAnchorNet(nn.Module):
    """Сеть advantage-sizing над фиксированной сеткой (advantage-сторона).

    Bug #50: переход на фиксированную сетку сайзингов.
    Output: slot_logits[B, K].
    """

    def __init__(self, input_size, hidden_size=128, num_sizes=15, num_buckets=3):
        super().__init__()
        self.num_sizes = num_sizes
        self.num_buckets = num_buckets
        self.base = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
        )
        self.anchor_head = nn.Linear(hidden_size, num_sizes)
        self._init_output_layers()

    def _init_output_layers(self):
        nn.init.zeros_(self.anchor_head.weight)
        nn.init.constant_(self.anchor_head.bias, 0.01)

    def forward(self, x):
        features = self.base(x)
        return self.anchor_head(features)


class SizingNetwork(nn.Module):
    """Отдельная sizing-policy: Tanh-squashed Gaussian над [min_bet, max_bet].

    Имеет собственный base — градиенты PG не пересекаются с action/value
    обучением. value_head внутри сети используется как baseline для PG.
    """

    def __init__(self, input_size, hidden_size=128, min_bet=0.1, max_bet=3.0,
                 log_std_min=-1.3, log_std_max=0.0):
        super().__init__()
        self.min_bet = min_bet
        self.max_bet = max_bet
        self.action_range = max_bet - min_bet
        # Динамический диапазон log_std: нижняя граница защищает std от схлопывания
        # (Bug #47-v3: на 1700 ит. LogStd упёрся в -1.993 при clamp=-2 → policy collapse).
        self.log_std_min = float(log_std_min)
        self.log_std_max = float(log_std_max)
        self.log_std_range = self.log_std_max - self.log_std_min

        self.base = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU()
        )
        self.mean_head = nn.Linear(hidden_size, 1)
        self.log_std_head = nn.Linear(hidden_size, 1)
        self.value_head = nn.Linear(hidden_size, 1)

        self._init_output_layers()

    def _init_output_layers(self):
        # Zero-init выходных слоёв: старт с z_mean=0 → squash → midpoint диапазона.
        # log_std стартует в середине диапазона [log_std_min, log_std_max] через сигмоиду от нуля.
        nn.init.zeros_(self.mean_head.weight)
        nn.init.zeros_(self.mean_head.bias)
        nn.init.zeros_(self.log_std_head.weight)
        nn.init.zeros_(self.log_std_head.bias)
        nn.init.zeros_(self.value_head.weight)
        nn.init.zeros_(self.value_head.bias)

    def forward(self, x):
        features = self.base(x)

        raw_z_mean = self.mean_head(features)
        z_mean = 3.0 * torch.tanh(raw_z_mean / 3.0)

        raw_log_std = self.log_std_head(features)
        log_std = self.log_std_min + self.log_std_range * torch.sigmoid(raw_log_std)

        value = self.value_head(features.detach())

        return z_mean, log_std, value

    def _squash(self, z):
        """Tanh-squash: z ∈ (-∞,+∞) → bet_size ∈ [min_bet, max_bet]."""
        return self.min_bet + self.action_range * (torch.tanh(z) + 1.0) / 2.0

    def _squash_inv(self, bet_size):
        """Обратное преобразование: bet_size ∈ [min_bet, max_bet] → z ∈ (-∞,+∞).
        Clamp для численной стабильности atanh."""
        t = 2.0 * (bet_size - self.min_bet) / self.action_range - 1.0
        t = torch.clamp(t, -0.999, 0.999)
        return torch.atanh(t)

    def _log_prob_correction(self, z):
        correction = 2.0 * (z + F.softplus(-2.0 * z) - math.log(2.0))
        return torch.clamp(correction, 0.0, 20.0)

    def sample_sizing(self, z_mean, log_std):
        """Сэмплировать bet_size с reparameterization-trick (для PG).

        Returns:
            bet_size: Tensor [B, 1] — множитель рейза ∈ [min_bet, max_bet]
            log_prob: Tensor [B, 1] — корректный log_prob с Tanh-коррекцией
            z_raw:    Tensor [B, 1] — не-squashed сэмпл (для PG обучения)
        """
        std = torch.exp(log_std)
        dist = torch.distributions.Normal(z_mean, std)
        z_raw = dist.rsample()
        bet_size = self._squash(z_raw)
        log_prob_z = dist.log_prob(z_raw)
        log_prob = log_prob_z + self._log_prob_correction(z_raw)
        return bet_size, log_prob, z_raw

    def sample_sizing_eval(self, z_mean, log_std):
        """Сэмплировать без градиентов для CFR-обхода. Возвращает Python-скаляры."""
        std = torch.exp(log_std)
        z_raw = torch.normal(z_mean, std)
        bet_size = self._squash(z_raw)
        log_prob_z = -0.5 * ((z_raw - z_mean) / std) ** 2 - log_std - 0.5 * math.log(2 * math.pi)
        log_prob = log_prob_z + self._log_prob_correction(z_raw)
        return bet_size.item(), log_prob.item(), z_raw.item()

    def mean_sizing(self, z_mean):
        """Детерминированный bet_size = squash(z_mean). Для инференса и distillation."""
        return self._squash(z_mean)

    def compute_log_prob_from_z(self, z_mean, log_std, z_raw):
        std = torch.exp(log_std)
        dist = torch.distributions.Normal(z_mean, std)
        log_prob_z = dist.log_prob(z_raw)
        log_prob = log_prob_z + self._log_prob_correction(z_raw)
        return torch.clamp(log_prob, -20.0, 5.0)

    def entropy_upper_bound(self, z_mean, log_std):
        """Верхняя граница энтропии squashed-Gaussian через энтропию Normal."""
        std = torch.exp(log_std)
        return torch.distributions.Normal(z_mean, std).entropy()


class SizingQNetwork(nn.Module):
    """Q(state, size) — аппроксимация value конкретного размера рейза.

    Bug #48: Continuous Q/Advantage для sizing вместо scalar PG.
    Отвечает на CFR-вопрос: «Насколько этот size лучше/хуже других в этом info set?»,
    а не PG-вопрос: «Был ли sampled size лучше baseline V(s)?».

    Вход: concat(encoded_state, normalized_size), где
      normalized_size = (bet_size - min_bet) / (max_bet - min_bet) ∈ [0, 1].
    Выход: scalar Q(state, size).
    """

    def __init__(self, state_dim, hidden_size=128, size_embed_dim=16):
        super().__init__()
        self.size_embed = nn.Linear(1, size_embed_dim)
        self.base = nn.Sequential(
            nn.Linear(state_dim + size_embed_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
        )
        self.q_head = nn.Linear(hidden_size, 1)
        nn.init.zeros_(self.q_head.weight)
        nn.init.zeros_(self.q_head.bias)

    def forward(self, state, normalized_size):
        size_feat = torch.relu(self.size_embed(normalized_size.unsqueeze(-1)))
        x = torch.cat([state, size_feat], dim=-1)
        return self.q_head(self.base(x)).squeeze(-1)


class QValueNetwork(nn.Module):
    """Q(s, a) — variance reduction на opponent nodes.

    НЕ стандартный Q-learning: предсказывает ожидаемый reward
    traversing_player, если оппонент выберет действие a в состоянии s.

    Control variate: adjusted = baseline + correction, где
      baseline   = Σ σ_opp(a) · Q(s, a)
      correction = v_sampled - Q(s, â)
    Несмещённость сохраняется при ЛЮБОМ Q; хорошее Q → низкая variance.
    """

    def __init__(self, input_size, hidden_size=128, num_actions=4):
        super().__init__()
        self.base = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
        )
        self.q_head = nn.Linear(hidden_size, num_actions)
        nn.init.zeros_(self.q_head.weight)
        nn.init.zeros_(self.q_head.bias)

    def forward(self, x):
        return self.q_head(self.base(x))


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


def encode_state(state, player_id=0):
    encoded = []
    num_players = len(state.players_state)

    if VERBOSE:
        print(f"Encoding state: current_player={state.current_player}, stage={state.stage}, hero={player_id}")
        print(f"Player states: {[(p.player, p.stake, p.bet_chips) for p in state.players_state]}")
        print(f"Pot: {state.pot}")

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

    norm_unit = float(state.bb) * 100.0 if hasattr(state, 'bb') and float(state.bb) > 0 else float(
        state.players_state[player_id].stake
        + state.players_state[player_id].bet_chips
        + state.players_state[player_id].pot_chips
    )
    if norm_unit < 1.0:
        norm_unit = 1.0

    pot_enc = [state.pot / norm_unit]
    encoded.append(pot_enc)

    button_enc = np.zeros(num_players)
    button_enc[(int(state.button) - player_id) % num_players] = 1
    encoded.append(button_enc)

    current_player_enc = np.zeros(num_players)
    current_player_enc[(int(state.current_player) - player_id) % num_players] = 1
    encoded.append(current_player_enc)

    for offset in range(num_players):
        p = (player_id + offset) % num_players
        player_state = state.players_state[p]
        active_enc = [1.0 if player_state.active else 0.0]
        bet_enc = [player_state.bet_chips / norm_unit]
        pot_chips_enc = [player_state.pot_chips / norm_unit]
        stake_enc = [player_state.stake / norm_unit]
        encoded.append(np.concatenate([active_enc, bet_enc, pot_chips_enc, stake_enc]))

    min_bet_enc = [state.min_bet / norm_unit]
    encoded.append(min_bet_enc)

    player_stake_enc = state.players_state[player_id].stake / norm_unit
    player_pot_enc = state.pot / norm_unit
    pot_commitment = player_pot_enc / (player_pot_enc + player_stake_enc) if (player_pot_enc + player_stake_enc) > 0 else 0.0
    encoded.append([pot_commitment])

    legal_actions_enc = np.zeros(4)
    for action_enum in state.legal_actions:
        legal_actions_enc[int(action_enum)] = 1
    encoded.append(legal_actions_enc)

    prev_action_enc = np.zeros(4 + 1)
    if state.from_action is not None:
        prev_action_enc[int(state.from_action.action.action)] = 1
        prev_action_enc[4] = state.from_action.action.amount / norm_unit
    encoded.append(prev_action_enc)

    return np.concatenate(encoded)


def encode_state_with_position(state, player_id=0):
    base = encode_state(state, player_id)
    num_players = len(state.players_state)
    one_hot = np.zeros(num_players, dtype=np.float32)
    one_hot[int(player_id)] = 1.0
    return np.concatenate([base, one_hot])
