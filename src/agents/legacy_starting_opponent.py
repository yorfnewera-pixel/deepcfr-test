"""Адаптер стартового оппонента из legacy Deep CFR с четырьмя действиями."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pokers as pkrs
import torch
import torch.nn as nn

from src.core.action_space import (
    ActionSlot,
    legal_action_mask,
    remaining_after_call,
    resolve_action,
)


_LEGACY_INPUT_SIZE = 156
_LEGACY_NUM_ACTIONS = 4


def _build_suit_canonical_map(hand_cards, public_cards) -> dict[int, int]:
    suit_to_canonical: dict[int, int] = {}
    next_canonical = 0
    for card in [*hand_cards, *public_cards]:
        suit = int(card.suit)
        if suit not in suit_to_canonical:
            suit_to_canonical[suit] = next_canonical
            next_canonical += 1
    return suit_to_canonical


def is_legacy_starting_opponent_checkpoint(checkpoint: object) -> bool:
    """Проверяет формат единственного поддерживаемого legacy-оппонента."""
    if not isinstance(checkpoint, dict):
        return False
    strategy_state = checkpoint.get("strategy_net")
    if not isinstance(strategy_state, dict):
        return False
    first_layer = strategy_state.get("fc1.weight")
    output_layer = strategy_state.get("fc6.weight")
    return (
        isinstance(first_layer, torch.Tensor)
        and isinstance(output_layer, torch.Tensor)
        and tuple(first_layer.shape) == (256, _LEGACY_INPUT_SIZE)
        and tuple(output_layer.shape) == (_LEGACY_NUM_ACTIONS, 256)
    )


class _LegacyPokerNetwork(nn.Module):
    """Точная архитектура strategy_net checkpoint mixed_checkpoint_iter_11200."""

    def __init__(self) -> None:
        super().__init__()
        self.fc1 = nn.Linear(_LEGACY_INPUT_SIZE, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 256)
        self.fc4 = nn.Linear(256, 256)
        self.fc5 = nn.Linear(256, 256)
        self.fc6 = nn.Linear(256, _LEGACY_NUM_ACTIONS)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        state = torch.relu(self.fc1(state))
        state = torch.relu(self.fc2(state))
        state = torch.relu(self.fc3(state))
        state = torch.relu(self.fc4(state))
        state = torch.relu(self.fc5(state))
        return self.fc6(state)


def _encode_legacy_state(state, player_id: int) -> np.ndarray:
    """Воспроизводит порядок и нормализацию признаков исходного checkpoint."""
    num_players = len(state.players_state)
    if num_players != 6:
        raise ValueError("Legacy starting opponent поддерживает только стол из шести игроков")

    encoded = []
    player_state = state.players_state[player_id]
    suit_map = _build_suit_canonical_map(player_state.hand, state.public_cards)

    hand = np.zeros(52, dtype=np.float32)
    for card in player_state.hand:
        hand[suit_map[int(card.suit)] * 13 + int(card.rank)] = 1.0
    encoded.append(hand)

    community = np.zeros(52, dtype=np.float32)
    for card in state.public_cards:
        community[suit_map[int(card.suit)] * 13 + int(card.rank)] = 1.0
    encoded.append(community)

    stage = np.zeros(5, dtype=np.float32)
    stage[int(state.stage)] = 1.0
    encoded.append(stage)

    norm_unit = float(getattr(state, "bb", 0.0)) * 100.0
    if norm_unit < 1.0:
        norm_unit = float(player_state.stake + player_state.bet_chips + player_state.pot_chips)
    norm_unit = max(norm_unit, 1.0)

    encoded.append(np.asarray([float(state.pot) / norm_unit], dtype=np.float32))

    button = np.zeros(num_players, dtype=np.float32)
    button[(int(state.button) - player_id) % num_players] = 1.0
    encoded.append(button)

    current_player = np.zeros(num_players, dtype=np.float32)
    current_player[(int(state.current_player) - player_id) % num_players] = 1.0
    encoded.append(current_player)

    for offset in range(num_players):
        player_state = state.players_state[(player_id + offset) % num_players]
        encoded.append(np.asarray([
            float(bool(player_state.active)),
            float(player_state.bet_chips) / norm_unit,
            float(player_state.pot_chips) / norm_unit,
            float(player_state.stake) / norm_unit,
        ], dtype=np.float32))

    encoded.append(np.asarray([float(state.min_bet) / norm_unit], dtype=np.float32))

    engine_actions = np.zeros(4, dtype=np.float32)
    for action in state.legal_actions:
        engine_actions[int(action)] = 1.0
    encoded.append(engine_actions)

    previous_action = np.zeros(5, dtype=np.float32)
    if state.from_action is not None:
        previous_action[int(state.from_action.action.action)] = 1.0
        previous_action[4] = float(state.from_action.action.amount) / norm_unit
    encoded.append(previous_action)

    result = np.concatenate(encoded, dtype=np.float32)
    if result.shape != (_LEGACY_INPUT_SIZE,):
        raise ValueError(f"Некорректный legacy state: {result.shape}, ожидалось 156 признаков")
    return result


class LegacyStartingOpponent:
    """Замороженная legacy policy, совместимая с нынешним движком действий."""

    def __init__(self, checkpoint_path: str | Path, device: str = "cpu") -> None:
        self.checkpoint_path = Path(checkpoint_path)
        self.device = torch.device(device)
        checkpoint = torch.load(self.checkpoint_path, map_location=self.device, weights_only=True)
        strategy_state = checkpoint.get("strategy_net") if isinstance(checkpoint, dict) else None
        if not isinstance(strategy_state, dict):
            raise ValueError("В legacy checkpoint отсутствует strategy_net")
        if tuple(strategy_state.get("fc1.weight", ()).shape) != (256, _LEGACY_INPUT_SIZE):
            raise ValueError("Legacy checkpoint имеет несовместимый размер входа")
        if tuple(strategy_state.get("fc6.weight", ()).shape) != (_LEGACY_NUM_ACTIONS, 256):
            raise ValueError("Legacy checkpoint имеет несовместимое пространство действий")

        self.input_size = _LEGACY_INPUT_SIZE
        self.num_actions = _LEGACY_NUM_ACTIONS
        self.network = _LegacyPokerNetwork().to(self.device)
        self.network.load_state_dict(strategy_state, strict=True)
        self.network.eval()
        for parameter in self.network.parameters():
            parameter.requires_grad_(False)

    @staticmethod
    def legal_choices(state) -> dict[int, ActionSlot]:
        """Возвращает старый двухставочный контракт action id."""
        choices: dict[int, ActionSlot] = {}
        engine_actions = tuple(state.legal_actions)
        slots = legal_action_mask(state)
        if pkrs.ActionEnum.Fold in engine_actions:
            choices[0] = ActionSlot.FOLD
        if pkrs.ActionEnum.Check in engine_actions:
            choices[1] = ActionSlot.CHECK
        elif pkrs.ActionEnum.Call in engine_actions:
            choices[1] = ActionSlot.CALL
        if slots[ActionSlot.RAISE_HALF_POT] > 0.0:
            choices[2] = ActionSlot.RAISE_HALF_POT
        elif (
            slots[ActionSlot.ALL_IN] > 0.0
            and 0.5 * float(state.pot) >= remaining_after_call(state)
        ):
            choices[2] = ActionSlot.ALL_IN
        if slots[ActionSlot.RAISE_POT] > 0.0:
            choices[3] = ActionSlot.RAISE_POT
        elif (
            slots[ActionSlot.ALL_IN] > 0.0
            and float(state.pot) >= remaining_after_call(state)
        ):
            choices[3] = ActionSlot.ALL_IN
        return choices

    def choose_action(self, state):
        choices = self.legal_choices(state)
        if not choices:
            raise ValueError("Для legacy starting opponent нет допустимого действия")

        player_id = int(state.current_player)
        encoded = _encode_legacy_state(state, player_id)
        state_tensor = torch.from_numpy(encoded).unsqueeze(0).to(self.device)
        with torch.inference_mode():
            logits = self.network(state_tensor)[0].cpu().numpy()

        action_ids = np.fromiter(choices.keys(), dtype=np.int64)
        probabilities = torch.softmax(torch.from_numpy(logits[action_ids]), dim=0).numpy()
        action_id = int(np.random.choice(action_ids, p=probabilities))
        return resolve_action(choices[action_id], state).action
