"""Frozen checkpoint policy for deterministic paired evaluation."""
from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import pokers as pkrs

import torch

from src.core.action_space import ACTION_SPACE_VERSION, NUM_ACTIONS, legal_action_mask, resolve_action
from src.core.deep_cfr import CHECKPOINT_FORMAT_VERSION, DeepCFRAgent
from src.core.model import (
    CARD_CONTEXT_ARCHITECTURE,
    CARD_FEATURE_SIZE,
    MONOLITHIC_ARCHITECTURE,
    NETWORK_ARCHITECTURES,
    PokerNetwork,
    encoder_input_size,
    encode_state_for_version,
    encode_state_with_position,
)


class FrozenBlueprintPolicy:
    """Неизменяемая стратегия из checkpoint для парной оценки."""

    def __init__(
        self,
        strategy_net: PokerNetwork,
        *,
        num_players: int,
        use_multi_agent: bool,
        encoding_version: str,
        device: str,
        agent: DeepCFRAgent | None = None,
    ):
        self.agent = agent
        self.strategy_net = strategy_net
        self.num_players = int(num_players)
        self.use_multi_agent = bool(use_multi_agent)
        self.encoding_version = str(encoding_version)
        self.device = torch.device(device)
        self.strategy_net.eval()
        for parameter in self.strategy_net.parameters():
            parameter.requires_grad_(False)

    @classmethod
    def from_checkpoint(cls, path: str | Path, *, num_players: int = 6, device: str = "cpu"):
        checkpoint = torch.load(path, map_location=device, weights_only=False)
        if checkpoint.get("checkpoint_kind") == "strategy_only":
            architecture, input_size, hidden_size = cls._validate_light_checkpoint(
                checkpoint,
                num_players,
            )
            state_dict = checkpoint["strategy_net"]
            strategy_net = PokerNetwork(
                input_size,
                hidden_size,
                NUM_ACTIONS,
                architecture,
            ).to(device)
            strategy_net.load_state_dict(state_dict, strict=True)
            use_multi_agent = bool(checkpoint.get("config", {}).get("use_multi_agent_advantage", False))
            encoding_version = str(checkpoint["encoding_version"])
            return cls(
                strategy_net,
                num_players=num_players,
                use_multi_agent=use_multi_agent,
                encoding_version=encoding_version,
                device=device,
            )

        architecture = cls._checkpoint_network_architecture(checkpoint)
        agent = DeepCFRAgent(
            player_id=0,
            num_players=num_players,
            device=device,
            network_architecture=architecture,
        )
        agent.load_model(str(path))
        return cls(
            agent.strategy_net,
            num_players=agent.num_players,
            use_multi_agent=agent.use_multi_agent,
            encoding_version=agent.encoding_version,
            device=device,
            agent=agent,
        )

    @staticmethod
    def _checkpoint_network_architecture(checkpoint: dict) -> str:
        checkpoint_config = checkpoint.get("config", {})
        if not isinstance(checkpoint_config, dict):
            checkpoint_config = {}
        architecture = checkpoint.get(
            "network_architecture",
            checkpoint_config.get("network_architecture"),
        )
        if architecture is None:
            state_dict = checkpoint.get("strategy_net")
            if isinstance(state_dict, dict) and "card_encoder.0.weight" in state_dict:
                raise ValueError(
                    "Чекпоинт с card_context_v1 не содержит метаданные архитектуры"
                )
            return MONOLITHIC_ARCHITECTURE
        if architecture not in NETWORK_ARCHITECTURES:
            raise ValueError("Чекпоинт имеет неизвестную архитектуру сети")
        if architecture == CARD_CONTEXT_ARCHITECTURE:
            card_feature_size = checkpoint.get(
                "card_feature_size",
                checkpoint_config.get("card_feature_size"),
            )
            if card_feature_size != CARD_FEATURE_SIZE:
                raise ValueError("Чекпоинт имеет несовместимый размер card-признаков")
        return architecture

    @classmethod
    def _validate_light_checkpoint(cls, checkpoint: dict, num_players: int) -> tuple[str, int, int]:
        if checkpoint.get("checkpoint_format_version") != CHECKPOINT_FORMAT_VERSION:
            raise ValueError("Light checkpoint имеет несовместимую версию формата")
        if checkpoint.get("action_space_version") != ACTION_SPACE_VERSION:
            raise ValueError("Light checkpoint имеет другое пространство действий")
        if int(checkpoint.get("num_actions", -1)) != NUM_ACTIONS:
            raise ValueError("Light checkpoint имеет неверное число действий")
        if int(checkpoint.get("num_players", -1)) != int(num_players):
            raise ValueError("Число игроков не совпадает с light checkpoint")
        encoding_version = checkpoint.get("encoding_version")
        if not isinstance(encoding_version, str):
            raise ValueError("В light checkpoint отсутствует версия encoder")
        expected_input_size = encoder_input_size(
            num_players,
            encoding_version,
            bool(checkpoint.get("config", {}).get("use_multi_agent_advantage", False)),
        )
        if int(checkpoint.get("encoder_input_size", -1)) != expected_input_size:
            raise ValueError("Light checkpoint имеет неверный размер входа encoder")
        state_dict = checkpoint.get("strategy_net")
        if not isinstance(state_dict, dict):
            raise ValueError("В light checkpoint отсутствуют веса strategy_net")
        architecture = cls._checkpoint_network_architecture(checkpoint)
        if architecture == MONOLITHIC_ARCHITECTURE:
            weight = state_dict.get("base.0.weight")
            if weight is None:
                raise ValueError("В light checkpoint отсутствуют веса strategy_net")
            actual_input_size = int(weight.shape[1])
            hidden_size = int(weight.shape[0])
        else:
            card_weight = state_dict.get("card_encoder.0.weight")
            context_weight = state_dict.get("context_encoder.0.weight")
            if card_weight is None or context_weight is None:
                raise ValueError("В light checkpoint отсутствуют веса card_context_v1")
            if int(card_weight.shape[1]) != CARD_FEATURE_SIZE:
                raise ValueError("Light checkpoint имеет несовместимый размер card-признаков")
            actual_input_size = int(card_weight.shape[1]) + int(context_weight.shape[1])
            hidden_size = int(card_weight.shape[0])
        if actual_input_size != expected_input_size:
            raise ValueError("Light checkpoint имеет неверный размер входа encoder в strategy_net")
        return architecture, actual_input_size, hidden_size

    def _encode_state(self, state: pkrs.State, player_id: int) -> np.ndarray:
        if self.use_multi_agent:
            return encode_state_with_position(state, player_id, self.encoding_version)
        return encode_state_for_version(state, player_id, self.encoding_version)

    def probabilities(self, state: pkrs.State, player_id: int | None = None) -> np.ndarray:
        """Возвращает маскированное распределение по шести слотам."""
        if player_id is None or int(player_id) == int(state.current_player):
            return self.probabilities_batch((state,))[0]

        mask = legal_action_mask(state)
        if not mask.any():
            raise ValueError("В текущем состоянии нет допустимых действий")
        encoded = self._encode_state(state, int(player_id))
        state_t = torch.from_numpy(encoded).float().unsqueeze(0).to(self.device)
        mask_t = torch.from_numpy(mask).unsqueeze(0).to(self.device)
        with torch.inference_mode():
            logits = self.strategy_net(state_t)
            masked_logits = torch.where(mask_t > 0.0, logits, torch.full_like(logits, -1e20))
            probabilities = torch.softmax(masked_logits, dim=1)[0].cpu().numpy()
        return probabilities.astype(np.float32)

    def probabilities_batch(self, states: Sequence[pkrs.State]) -> np.ndarray:
        """Возвращает маскированные distribution для состояний с разными игроками хода."""
        if not states:
            return np.empty((0, NUM_ACTIONS), dtype=np.float32)

        masks = np.stack([legal_action_mask(state) for state in states])
        if not masks.any(axis=1).all():
            raise ValueError("В одном из состояний нет допустимых действий")
        encoded = np.stack([
            self._encode_state(state, int(state.current_player))
            for state in states
        ])
        state_t = torch.from_numpy(encoded).float().to(self.device)
        mask_t = torch.from_numpy(masks).to(self.device)
        with torch.inference_mode():
            logits = self.strategy_net(state_t)
            masked_logits = torch.where(mask_t > 0.0, logits, torch.full_like(logits, -1e20))
            probabilities = torch.softmax(masked_logits, dim=1).cpu().numpy()
        return probabilities.astype(np.float32)

    def choose_action(self, state: pkrs.State, rng: np.random.Generator) -> pkrs.Action:
        action_probs = self.probabilities(state)
        action_type = int(rng.choice(len(action_probs), p=action_probs))
        return resolve_action(action_type, state).action
