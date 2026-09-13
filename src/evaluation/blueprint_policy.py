"""Frozen checkpoint policy for deterministic paired evaluation."""
from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import pokers as pkrs

import torch

from src.core.action_space import (
    ACTION_LABELS,
    ACTION_SPACE_VERSION,
    NUM_ACTIONS,
    legal_action_mask,
    resolve_action,
)
from src.core.checkpoint_kinds import (
    HU_CURRENT_POLICY_SELF_PLAY_CHECKPOINT_KIND,
    HU_FULL_CHECKPOINT_VERSION,
    HU_STRATEGY_ONLY_CHECKPOINT_KIND,
    STRATEGY_ONLY_CHECKPOINT_KIND,
)
from src.core.deep_cfr import (
    CHECKPOINT_FORMAT_VERSION,
    DeepCFRAgent,
    full_checkpoint_network_spec,
    GAME_RULES_VERSION,
)
from src.core.game_contract import validate_fixed_hu_game_contract
from src.core.model import (
    CARD_CONTEXT_ARCHITECTURE,
    CARD_CONTEXT_V2_ARCHITECTURE,
    CARD_FEATURE_SIZE,
    MONOLITHIC_ARCHITECTURE,
    NETWORK_ARCHITECTURES,
    PokerNetwork,
    encoder_input_size,
    encode_state_for_version,
    encode_state_with_position,
)


def strategy_net_input_size(strategy_net: PokerNetwork) -> int:
    """Возвращает размер фактического входа strategy-сети."""
    if strategy_net.architecture == MONOLITHIC_ARCHITECTURE:
        return int(strategy_net.base[0].in_features)
    return int(
        strategy_net.card_encoder[0].in_features
        + strategy_net.context_encoder[0].in_features
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
        base_input_size: int | None = None,
        strategy_input_size: int | None = None,
        strategy_actor_conditioned: bool = False,
        strategy_actor_count: int = 0,
        agent: DeepCFRAgent | None = None,
    ):
        self.agent = agent
        self.strategy_net = strategy_net
        self.num_players = int(num_players)
        self.use_multi_agent = bool(use_multi_agent)
        self.encoding_version = str(encoding_version)
        self.device = torch.device(device)
        self.base_input_size = int(base_input_size or strategy_net_input_size(strategy_net))
        self.strategy_input_size = int(strategy_input_size or strategy_net_input_size(strategy_net))
        self.strategy_actor_conditioned = bool(strategy_actor_conditioned)
        self.strategy_actor_count = int(strategy_actor_count)
        self.strategy_net.eval()
        for parameter in self.strategy_net.parameters():
            parameter.requires_grad_(False)

    @classmethod
    def from_checkpoint(
        cls,
        path: str | Path,
        *,
        num_players: int | None = None,
        device: str = "cpu",
    ):
        checkpoint = torch.load(path, map_location=device, weights_only=False)
        checkpoint_kind = checkpoint.get("checkpoint_kind")
        if checkpoint_kind in {
            STRATEGY_ONLY_CHECKPOINT_KIND,
            HU_STRATEGY_ONLY_CHECKPOINT_KIND,
        }:
            resolved_num_players = cls._resolve_root_num_players(checkpoint, num_players)
            (
                architecture,
                strategy_input_size,
                hidden_size,
                base_input_size,
                strategy_actor_conditioned,
                strategy_actor_count,
            ) = cls._validate_light_checkpoint(
                checkpoint,
                resolved_num_players,
            )
            state_dict = checkpoint["strategy_net"]
            strategy_net = PokerNetwork(
                strategy_input_size,
                hidden_size,
                NUM_ACTIONS,
                architecture,
            ).to(device)
            strategy_net.load_state_dict(state_dict, strict=True)
            use_multi_agent = bool(
                checkpoint["config"].get("use_multi_agent_advantage", False)
            )
            encoding_version = str(checkpoint["encoding_version"])
            return cls(
                strategy_net,
                num_players=resolved_num_players,
                use_multi_agent=use_multi_agent,
                encoding_version=encoding_version,
                device=device,
                base_input_size=base_input_size,
                strategy_input_size=strategy_input_size,
                strategy_actor_conditioned=strategy_actor_conditioned,
                strategy_actor_count=strategy_actor_count,
            )

        if checkpoint_kind == HU_CURRENT_POLICY_SELF_PLAY_CHECKPOINT_KIND:
            (
                architecture,
                strategy_input_size,
                hidden_size,
                base_input_size,
                use_multi_agent,
                encoding_version,
                strategy_state,
            ) = cls._validate_hu_full_checkpoint(checkpoint, num_players)
            strategy_net = PokerNetwork(
                strategy_input_size,
                hidden_size,
                NUM_ACTIONS,
                architecture,
            ).to(device)
            strategy_net.load_state_dict(strategy_state, strict=True)
            return cls(
                strategy_net,
                num_players=2,
                use_multi_agent=use_multi_agent,
                encoding_version=encoding_version,
                device=device,
                base_input_size=base_input_size,
                strategy_input_size=strategy_input_size,
                strategy_actor_conditioned=True,
                strategy_actor_count=2,
            )

        resolved_num_players = cls._resolve_root_num_players(checkpoint, num_players)
        architecture, _, hidden_size = full_checkpoint_network_spec(checkpoint)
        agent = DeepCFRAgent(
            player_id=0,
            num_players=resolved_num_players,
            device=device,
            network_architecture=architecture,
            hidden_size=hidden_size,
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
    def _resolve_root_num_players(checkpoint: dict, requested_num_players: int | None) -> int:
        checkpoint_num_players = checkpoint.get("num_players")
        if isinstance(checkpoint_num_players, bool) or not isinstance(checkpoint_num_players, int):
            raise ValueError("Чекпоинт имеет некорректное число игроков")
        return checkpoint_num_players if requested_num_players is None else int(requested_num_players)

    @classmethod
    def _validate_hu_full_checkpoint(
        cls,
        checkpoint: dict,
        requested_num_players: int | None,
    ) -> tuple[str, int, int, int, bool, str, dict[str, torch.Tensor]]:
        """Проверяет inference-контракт HU full checkpoint без восстановления training state."""
        if checkpoint.get("checkpoint_kind") != HU_CURRENT_POLICY_SELF_PLAY_CHECKPOINT_KIND:
            raise ValueError("Ожидался полный HU checkpoint")
        if checkpoint.get("hu_checkpoint_version") != HU_FULL_CHECKPOINT_VERSION:
            raise ValueError("HU full checkpoint имеет несовместимую версию")
        validate_fixed_hu_game_contract(checkpoint.get("game_contract"), "HU full checkpoint")
        if checkpoint.get("checkpoint_format_version") != CHECKPOINT_FORMAT_VERSION:
            raise ValueError("HU full checkpoint имеет несовместимую версию формата")
        if checkpoint.get("game_rules_version") != GAME_RULES_VERSION:
            raise ValueError("HU full checkpoint имеет несовместимую версию правил игры")
        if checkpoint.get("action_space_version") != ACTION_SPACE_VERSION:
            raise ValueError("HU full checkpoint имеет другое пространство действий")
        if checkpoint.get("action_labels") != list(ACTION_LABELS):
            raise ValueError("HU full checkpoint имеет несовместимые action labels")
        if checkpoint.get("num_actions") != NUM_ACTIONS:
            raise ValueError("HU full checkpoint имеет неверное число действий")

        mode = checkpoint.get("mode")
        if not isinstance(mode, dict):
            raise ValueError("HU full checkpoint не содержит корректный mode")
        mode_num_players = mode.get("num_players")
        base_input_size = mode.get("encoder_input_size")
        encoding_version = mode.get("encoding_version")
        use_multi_agent = mode.get("use_multi_agent_advantage")
        if (
            mode.get("hu_current_policy_self_play") is not True
            or mode.get("num_trainable_players") != 2
            or mode_num_players != 2
            or use_multi_agent is not False
            or isinstance(base_input_size, bool)
            or not isinstance(base_input_size, int)
            or base_input_size <= 0
            or not isinstance(encoding_version, str)
        ):
            raise ValueError("HU full checkpoint имеет несовместимый mode")
        if requested_num_players is not None and int(requested_num_players) != mode_num_players:
            raise ValueError("HU full checkpoint поддерживает только двух игроков")
        try:
            expected_base_input_size = encoder_input_size(
                mode_num_players,
                encoding_version,
                use_multi_agent,
            )
        except ValueError as error:
            raise ValueError("HU full checkpoint имеет неизвестную версию encoder") from error
        if base_input_size != expected_base_input_size:
            raise ValueError("HU full checkpoint имеет несовместимый base encoder input size")

        architecture_payload = checkpoint.get("architecture")
        if not isinstance(architecture_payload, dict):
            raise ValueError("HU full checkpoint не содержит strategy architecture")
        strategy_schema = architecture_payload.get("strategy")
        if not isinstance(strategy_schema, dict):
            raise ValueError("HU full checkpoint не содержит strategy architecture")
        architecture = strategy_schema.get("network_architecture")
        hidden_size = strategy_schema.get("hidden_size")
        strategy_input_size = strategy_schema.get("input_size")
        if not isinstance(architecture, str) or architecture not in NETWORK_ARCHITECTURES:
            raise ValueError("HU full checkpoint имеет неизвестную strategy архитектуру")
        if (
            isinstance(hidden_size, bool)
            or not isinstance(hidden_size, int)
            or hidden_size <= 0
            or isinstance(strategy_input_size, bool)
            or not isinstance(strategy_input_size, int)
            or strategy_input_size != base_input_size + 2
            or strategy_schema.get("num_actions") != NUM_ACTIONS
        ):
            raise ValueError("HU full checkpoint имеет несовместимый actor-conditioned strategy input")
        if architecture in (CARD_CONTEXT_ARCHITECTURE, CARD_CONTEXT_V2_ARCHITECTURE):
            if strategy_schema.get("card_feature_size") != CARD_FEATURE_SIZE:
                raise ValueError("HU full checkpoint имеет несовместимый размер card-признаков")
        if architecture == CARD_CONTEXT_V2_ARCHITECTURE and (
            strategy_schema.get("fusion_input_size") != hidden_size * 2
            or strategy_schema.get("fusion_output_size") != hidden_size
        ):
            raise ValueError("HU full checkpoint имеет несовместимую fusion architecture")

        strategy_payload = checkpoint.get("strategy")
        if not isinstance(strategy_payload, dict):
            raise ValueError("HU full checkpoint не содержит strategy state")
        strategy_state = strategy_payload.get("network")
        expected_shapes = cls._strategy_state_shapes(
            architecture,
            strategy_input_size,
            hidden_size,
        )
        if not isinstance(strategy_state, dict) or set(strategy_state) != set(expected_shapes):
            raise ValueError("HU full checkpoint имеет несовместимые strategy weights")
        for parameter_name, expected_shape in expected_shapes.items():
            parameter = strategy_state[parameter_name]
            if not torch.is_tensor(parameter) or tuple(parameter.shape) != expected_shape:
                if architecture == CARD_CONTEXT_V2_ARCHITECTURE and parameter_name.startswith("fusion"):
                    raise ValueError("HU full checkpoint имеет несовместимые fusion weights")
                raise ValueError("HU full checkpoint имеет несовместимые strategy weights")
        return (
            architecture,
            strategy_input_size,
            hidden_size,
            base_input_size,
            use_multi_agent,
            encoding_version,
            strategy_state,
        )

    @staticmethod
    def _strategy_state_shapes(
        architecture: str,
        strategy_input_size: int,
        hidden_size: int,
    ) -> dict[str, tuple[int, ...]]:
        if architecture == MONOLITHIC_ARCHITECTURE:
            return {
                "base.0.weight": (hidden_size, strategy_input_size),
                "base.0.bias": (hidden_size,),
                "base.2.weight": (hidden_size, hidden_size),
                "base.2.bias": (hidden_size,),
                "base.4.weight": (hidden_size, hidden_size),
                "base.4.bias": (hidden_size,),
                "action_head.weight": (NUM_ACTIONS, hidden_size),
                "action_head.bias": (NUM_ACTIONS,),
            }
        context_size = strategy_input_size - CARD_FEATURE_SIZE
        if context_size < 0:
            raise ValueError("HU full checkpoint имеет несовместимый card/context input size")
        shapes = {
            "card_encoder.0.weight": (hidden_size, CARD_FEATURE_SIZE),
            "card_encoder.0.bias": (hidden_size,),
            "context_encoder.0.weight": (hidden_size, context_size),
            "context_encoder.0.bias": (hidden_size,),
        }
        if architecture == CARD_CONTEXT_V2_ARCHITECTURE:
            shapes.update({
                "fusion.0.weight": (hidden_size, hidden_size * 2),
                "fusion.0.bias": (hidden_size,),
                "fusion.2.weight": (hidden_size, hidden_size),
                "fusion.2.bias": (hidden_size,),
                "action_head.weight": (NUM_ACTIONS, hidden_size),
                "action_head.bias": (NUM_ACTIONS,),
            })
        else:
            shapes.update({
                "action_head.weight": (NUM_ACTIONS, hidden_size * 2),
                "action_head.bias": (NUM_ACTIONS,),
            })
        return shapes

    @staticmethod
    def _checkpoint_network_architecture(checkpoint: dict) -> str:
        checkpoint_config = checkpoint.get("config", {})
        if not isinstance(checkpoint_config, dict):
            checkpoint_config = {}
        if "network_architecture" in checkpoint:
            architecture = checkpoint["network_architecture"]
        elif "network_architecture" in checkpoint_config:
            architecture = checkpoint_config["network_architecture"]
        else:
            state_dict = checkpoint.get("strategy_net")
            if isinstance(state_dict, dict) and "card_encoder.0.weight" in state_dict:
                raise ValueError(
                    "Чекпоинт с card_context_v1 не содержит метаданные архитектуры"
                )
            return MONOLITHIC_ARCHITECTURE
        if not isinstance(architecture, str):
            raise ValueError("Чекпоинт имеет некорректное значение архитектуры сети")
        if architecture not in NETWORK_ARCHITECTURES:
            raise ValueError("Чекпоинт имеет неизвестную архитектуру сети")
        if architecture != MONOLITHIC_ARCHITECTURE:
            card_feature_size = checkpoint.get(
                "card_feature_size",
                checkpoint_config.get("card_feature_size"),
            )
            if card_feature_size != CARD_FEATURE_SIZE:
                raise ValueError("Чекпоинт имеет несовместимый размер card-признаков")
        return architecture

    @classmethod
    def _validate_light_checkpoint(
        cls,
        checkpoint: dict,
        num_players: int,
    ) -> tuple[str, int, int, int, bool, int]:
        checkpoint_kind = checkpoint.get("checkpoint_kind")
        is_hu_strategy = checkpoint_kind == HU_STRATEGY_ONLY_CHECKPOINT_KIND
        checkpoint_config = checkpoint.get("config", {})
        if not isinstance(checkpoint_config, dict):
            raise ValueError("Light checkpoint имеет некорректный config")
        if checkpoint.get("checkpoint_format_version") != CHECKPOINT_FORMAT_VERSION:
            raise ValueError("Light checkpoint имеет несовместимую версию формата")
        if checkpoint.get("game_rules_version") != GAME_RULES_VERSION:
            raise ValueError("Light checkpoint имеет несовместимую версию правил игры")
        if checkpoint.get("action_space_version") != ACTION_SPACE_VERSION:
            raise ValueError("Light checkpoint имеет другое пространство действий")
        if int(checkpoint.get("num_actions", -1)) != NUM_ACTIONS:
            raise ValueError("Light checkpoint имеет неверное число действий")
        if int(checkpoint.get("num_players", -1)) != int(num_players):
            raise ValueError("Число игроков не совпадает с light checkpoint")
        encoding_version = checkpoint.get("encoding_version")
        if not isinstance(encoding_version, str):
            raise ValueError("В light checkpoint отсутствует версия encoder")
        expected_base_input_size = encoder_input_size(
            num_players,
            encoding_version,
            bool(checkpoint_config.get("use_multi_agent_advantage", False)),
        )
        base_input_size = checkpoint.get("base_encoder_input_size")
        if isinstance(base_input_size, bool) or not isinstance(base_input_size, int):
            raise ValueError("Light checkpoint не содержит base encoder input size")
        if int(checkpoint.get("encoder_input_size", -1)) != expected_base_input_size:
            raise ValueError("Light checkpoint имеет неверный размер входа encoder")
        if base_input_size != expected_base_input_size:
            raise ValueError("Light checkpoint имеет неверный базовый размер encoder")
        strategy_input_size = checkpoint.get("strategy_input_size")
        if isinstance(strategy_input_size, bool) or not isinstance(strategy_input_size, int):
            raise ValueError("Light checkpoint не содержит размер входа strategy")
        strategy_actor_conditioned = checkpoint.get("strategy_actor_conditioned")
        strategy_actor_count = checkpoint.get("strategy_actor_count")
        if not isinstance(strategy_actor_conditioned, bool):
            raise ValueError("Light checkpoint не содержит actor-conditioning contract")
        if isinstance(strategy_actor_count, bool) or not isinstance(strategy_actor_count, int):
            raise ValueError("Light checkpoint не содержит число actor features")
        if is_hu_strategy:
            if checkpoint.get("hu_checkpoint_version") != HU_FULL_CHECKPOINT_VERSION:
                raise ValueError("HU light checkpoint имеет несовместимую версию")
            validate_fixed_hu_game_contract(checkpoint.get("game_contract"), "HU light checkpoint")
            if (
                int(checkpoint.get("num_players", -1)) != 2
                or int(num_players) != 2
                or not strategy_actor_conditioned
                or strategy_actor_count != 2
            ):
                raise ValueError("HU light checkpoint требует actor conditioning для двух игроков")
            if strategy_input_size != base_input_size + strategy_actor_count:
                raise ValueError("HU light checkpoint имеет несовместимые base/strategy input sizes")
        elif strategy_actor_conditioned or strategy_actor_count != 0 or strategy_input_size != base_input_size:
            raise ValueError("Обычный light checkpoint не должен иметь HU actor conditioning")
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
            if int(context_weight.shape[0]) != hidden_size:
                raise ValueError("Light checkpoint имеет несовместимые card/context веса")
            if architecture != CARD_CONTEXT_ARCHITECTURE:
                fusion_input = state_dict.get("fusion.0.weight")
                fusion_output = state_dict.get("fusion.2.weight")
                if (
                    fusion_input is None
                    or fusion_output is None
                    or tuple(fusion_input.shape) != (hidden_size, hidden_size * 2)
                    or tuple(fusion_output.shape) != (hidden_size, hidden_size)
                ):
                    raise ValueError("Light checkpoint имеет несовместимые fusion-веса")
        if actual_input_size != strategy_input_size:
            raise ValueError("Light checkpoint имеет неверный размер входа encoder в strategy_net")
        return (
            architecture,
            actual_input_size,
            hidden_size,
            base_input_size,
            strategy_actor_conditioned,
            strategy_actor_count,
        )

    def _encode_state(self, state: pkrs.State, player_id: int) -> np.ndarray:
        if self.use_multi_agent:
            base = encode_state_with_position(state, player_id, self.encoding_version)
        else:
            base = encode_state_for_version(state, player_id, self.encoding_version)
        if len(base) != self.base_input_size:
            raise ValueError("Кодировщик состояния имеет несовместимый размер base input")
        if not self.strategy_actor_conditioned:
            return base
        if not 0 <= int(player_id) < self.strategy_actor_count:
            raise ValueError("player_id выходит за контракт HU actor conditioning")
        actor = np.zeros(self.strategy_actor_count, dtype=np.float32)
        actor[int(player_id)] = 1.0
        return np.concatenate((base, actor))

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
