"""Action-only реализация Deep CFR для фиксированного пространства действий."""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from numbers import Integral
from pathlib import Path
from typing import Any, NoReturn

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
import pokers as pkrs

from src.core.action_space import (
    ACTION_LABELS,
    ACTION_SPACE_VERSION,
    NUM_ACTIONS,
    is_raise_slot,
    legal_action_mask,
    resolve_action,
)
from src.core.buffers import AdvantageBuffer, StrategyBuffer
from src.core.checkpointing import _resolve_model_save_path
from src.core.model import (
    CARD_CONTEXT_ARCHITECTURE,
    CARD_FEATURE_SIZE,
    HISTORY_SUMMARY_V3_ENCODING_VERSION,
    MONOLITHIC_ARCHITECTURE,
    NETWORK_ARCHITECTURES,
    PokerNetwork,
    encoder_input_size,
    encode_state_for_version,
    encode_state_with_position,
    history_summary_size,
    legacy_base_input_size,
)
from src.core.traversal_errors import TraversalFailure, TraversalFailureContext
from src.utils.config import (
    cfg_clear_strategy_buffer_each_iteration,
    cfg_get,
    cfg_reservoir_flag,
)
from src.utils.logging import log_game_error
from src.utils.traversal_profiler import TRAVERSAL_PROFILER, profile_section


CHECKPOINT_FORMAT_VERSION = 6


def full_checkpoint_network_spec(checkpoint: dict) -> tuple[str, int, int]:
    """Возвращает и проверяет сетевой контракт полного checkpoint до загрузки весов."""
    checkpoint_config = checkpoint.get("config", {})
    if not isinstance(checkpoint_config, dict):
        checkpoint_config = {}

    if "network_architecture" in checkpoint:
        architecture = checkpoint["network_architecture"]
    elif "network_architecture" in checkpoint_config:
        architecture = checkpoint_config["network_architecture"]
    else:
        strategy_state = checkpoint.get("strategy_net")
        if isinstance(strategy_state, dict) and "card_encoder.0.weight" in strategy_state:
            raise ValueError(
                "Чекпоинт с card_context_v1 не содержит метаданные архитектуры"
            )
        architecture = MONOLITHIC_ARCHITECTURE
    if not isinstance(architecture, str) or architecture not in NETWORK_ARCHITECTURES:
        raise ValueError("Чекпоинт имеет некорректное значение архитектуры сети")

    if architecture == CARD_CONTEXT_ARCHITECTURE:
        card_feature_size = checkpoint.get(
            "card_feature_size",
            checkpoint_config.get("card_feature_size"),
        )
        if card_feature_size != CARD_FEATURE_SIZE:
            raise ValueError("Чекпоинт имеет несовместимый размер card-признаков")

    input_size = checkpoint.get("encoder_input_size")
    if isinstance(input_size, bool) or not isinstance(input_size, int) or input_size <= 0:
        raise ValueError("Чекпоинт имеет некорректный размер входа encoder")

    hidden_size = checkpoint_config.get("hidden_size")
    if hidden_size is None and architecture == MONOLITHIC_ARCHITECTURE:
        strategy_state = checkpoint.get("strategy_net")
        if isinstance(strategy_state, dict):
            base_weight = strategy_state.get("base.0.weight")
            if torch.is_tensor(base_weight) and base_weight.ndim == 2:
                hidden_size = int(base_weight.shape[0])
    if isinstance(hidden_size, bool) or not isinstance(hidden_size, int) or hidden_size <= 0:
        raise ValueError("В checkpoint отсутствует корректный hidden_size сети")

    if architecture == MONOLITHIC_ARCHITECTURE:
        expected_shapes = {
            "base.0.weight": (hidden_size, input_size),
            "base.0.bias": (hidden_size,),
            "base.2.weight": (hidden_size, hidden_size),
            "base.2.bias": (hidden_size,),
            "base.4.weight": (hidden_size, hidden_size),
            "base.4.bias": (hidden_size,),
            "action_head.weight": (NUM_ACTIONS, hidden_size),
            "action_head.bias": (NUM_ACTIONS,),
        }
    else:
        context_size = input_size - CARD_FEATURE_SIZE
        if context_size < 0:
            raise ValueError("Чекпоинт имеет несовместимый размер входа card_context_v1")
        expected_shapes = {
            "card_encoder.0.weight": (hidden_size, CARD_FEATURE_SIZE),
            "card_encoder.0.bias": (hidden_size,),
            "context_encoder.0.weight": (hidden_size, context_size),
            "context_encoder.0.bias": (hidden_size,),
            "action_head.weight": (NUM_ACTIONS, hidden_size * 2),
            "action_head.bias": (NUM_ACTIONS,),
        }

    for network_key in ("advantage_net", "advantage_target_net", "strategy_net"):
        state_dict = checkpoint.get(network_key)
        if not isinstance(state_dict, dict):
            raise ValueError(f"В checkpoint отсутствуют веса {network_key}")
        if set(state_dict) != set(expected_shapes):
            raise ValueError(f"{network_key} имеет несовместимый набор параметров")
        for parameter_name, expected_shape in expected_shapes.items():
            parameter = state_dict[parameter_name]
            if not torch.is_tensor(parameter) or tuple(parameter.shape) != expected_shape:
                raise ValueError(
                    f"{network_key} имеет несовместимую форму параметра {parameter_name}"
                )

    return architecture, input_size, hidden_size


@dataclass
class _TraversalSampleCollector:
    """Накапливает samples до успешного завершения корневого traversal."""

    advantage_samples: list[tuple[np.ndarray, np.ndarray, np.ndarray, int]] = field(
        default_factory=list
    )
    strategy_samples: list[tuple[np.ndarray, np.ndarray, np.ndarray, int]] = field(
        default_factory=list
    )


class DeepCFRAgent:
    """Deep CFR без непрерывного sizing и Q-control-variate."""

    def __init__(self, player_id=0, num_players=None, memory_size=None,
                 device="cpu", network_architecture=None, hidden_size=None, **_legacy_options):
        self.player_id = int(player_id)
        self.num_players = int(num_players or cfg_get("num_players", 6))
        configured_trainable_players = int(cfg_get("num_trainable_players", 1))
        self.num_trainable_players = max(1, min(configured_trainable_players, self.num_players))
        self.device = torch.device(device)
        self.num_actions = NUM_ACTIONS
        configured_actions = int(cfg_get("num_actions", NUM_ACTIONS))
        if configured_actions != NUM_ACTIONS:
            raise ValueError(
                f"Конфигурация требует {configured_actions} действий, но контракт "
                f"{ACTION_SPACE_VERSION} требует {NUM_ACTIONS}."
            )

        self.use_multi_agent = bool(cfg_get("use_multi_agent_advantage", False))
        self.encoding_version = str(
            cfg_get("encoding_version", HISTORY_SUMMARY_V3_ENCODING_VERSION)
        )
        self.input_size = encoder_input_size(
            self.num_players,
            self.encoding_version,
            self.use_multi_agent,
        )
        hidden_size = int(
            cfg_get("hidden_size", 256) if hidden_size is None else hidden_size
        )
        self.network_architecture = str(
            network_architecture
            if network_architecture is not None
            else cfg_get("network_architecture", MONOLITHIC_ARCHITECTURE)
        )

        self.advantage_net = PokerNetwork(
            self.input_size, hidden_size, NUM_ACTIONS, self.network_architecture
        ).to(self.device)
        self.advantage_target_net = PokerNetwork(
            self.input_size, hidden_size, NUM_ACTIONS, self.network_architecture
        ).to(self.device)
        self.advantage_target_net.load_state_dict(self.advantage_net.state_dict())
        self.advantage_target_net.eval()
        for parameter in self.advantage_target_net.parameters():
            parameter.requires_grad_(False)
        self.optimizer = optim.AdamW(
            self.advantage_net.parameters(),
            lr=float(cfg_get("advantage_lr", 1e-4)),
            weight_decay=float(cfg_get("advantage_weight_decay", 1e-5)),
        )
        self.strategy_net = PokerNetwork(
            self.input_size, hidden_size, NUM_ACTIONS, self.network_architecture
        ).to(self.device)
        self.strategy_optimizer = optim.AdamW(
            self.strategy_net.parameters(),
            lr=float(cfg_get("strategy_lr", 5e-5)),
            weight_decay=float(cfg_get("strategy_weight_decay", 1e-5)),
        )
        self.teacher_strategy_net: PokerNetwork | None = None
        self.teacher_strategy_input_size: int | None = None
        self.teacher_strategy_num_players: int | None = None
        self.teacher_strategy_use_multi_agent = False
        self.teacher_strategy_encoding_version: str | None = None
        self.strategy_distillation_lambda = float(cfg_get("strategy_distillation_lambda", 0.0))
        self.strategy_distillation_temperature = max(
            float(cfg_get("strategy_distillation_temperature", 1.0)),
            1e-6,
        )
        self.strategy_distillation_anneal_iterations = int(
            cfg_get("strategy_distillation_anneal_iterations", 0) or 0
        )

        advantage_memory_size = int(memory_size or cfg_get("advantage_memory_size", 300000))
        strategy_memory_size = int(cfg_get("strategy_memory_size", 300000))
        self.advantage_buffer = AdvantageBuffer(advantage_memory_size, self.input_size, NUM_ACTIONS)
        self.strategy_buffer = StrategyBuffer(
            strategy_memory_size,
            self.input_size,
            NUM_ACTIONS,
            reservoir=bool(cfg_get("strategy_buffer_reservoir", False)),
        )
        self.advantage_buffer_reservoir = cfg_reservoir_flag("advantage_buffer_reservoir")
        self.clear_strategy_buffer_each_iteration = cfg_clear_strategy_buffer_each_iteration()
        self.save_replay_buffers_in_checkpoint = bool(cfg_get("save_replay_buffers_in_checkpoint", False))

        self.advantage_batch_size = int(cfg_get("advantage_batch_size", 256))
        self.strategy_batch_size = int(cfg_get("strategy_batch_size", 128))
        self.advantage_epochs = int(cfg_get("advantage_epochs", 1))
        self.strategy_epochs = int(cfg_get("strategy_epochs", 1))
        advantage_train_steps = cfg_get("advantage_train_steps", None)
        strategy_train_steps = cfg_get("strategy_train_steps", None)
        self.advantage_train_steps = int(advantage_train_steps) if advantage_train_steps is not None else None
        self.strategy_train_steps = int(strategy_train_steps) if strategy_train_steps is not None else None
        self.training_preload_to_device = bool(cfg_get("training_preload_to_device", False))
        self.discount_alpha = float(cfg_get("discount_alpha", 2.0))
        self.discount_gamma = float(cfg_get("discount_gamma", 1.0))
        self.advantage_accumulation = str(cfg_get("advantage_accumulation", "dcfr_plus"))
        self.advantage_regret_norm = str(cfg_get("advantage_regret_norm", "none"))
        regret_clip = cfg_get("advantage_regret_clip", None)
        self.advantage_regret_clip = float(regret_clip) if regret_clip is not None else None
        self.advantage_reward_scale = float(cfg_get("advantage_reward_scale", 1.0))
        self.advantage_loss = str(cfg_get("advantage_loss", "mse"))
        self.advantage_huber_delta = float(cfg_get("advantage_huber_delta", 1.0))
        self.iteration_count = 0
        self.last_advantage_train_steps = 0
        self.last_advantage_effective_batch_size = 0
        self.last_advantage_target_stats = None
        self.last_advantage_profile = None
        self.last_strategy_profile = None
        self._traversal_random_agent = None
        self._active_traversal_collector: _TraversalSampleCollector | None = None
        self._active_traversal_index: int | None = None
        self._opponent_policy_agents: dict[int, Any] = {}
        self._opponent_advantage_nets: dict[int, PokerNetwork] = {}
        self._opponent_strategy_nets: dict[int, PokerNetwork] = {}
        self.reset_traversal_stats()

    @staticmethod
    def _network_input_size(
        num_players,
        use_multi_agent=False,
        encoding_version=HISTORY_SUMMARY_V3_ENCODING_VERSION,
    ):
        return encoder_input_size(num_players, encoding_version, use_multi_agent)

    @staticmethod
    def _network_hidden_size(network):
        if network.architecture == CARD_CONTEXT_ARCHITECTURE:
            return int(network.card_encoder[0].out_features)
        return int(network.base[0].out_features)

    def _network_metadata(self):
        metadata = {"network_architecture": self.network_architecture}
        if self.network_architecture == CARD_CONTEXT_ARCHITECTURE:
            metadata["card_feature_size"] = CARD_FEATURE_SIZE
        return metadata

    def _validate_checkpoint_network_metadata(self, checkpoint):
        checkpoint_config = checkpoint.get("config", {})
        if not isinstance(checkpoint_config, dict):
            checkpoint_config = {}
        if "network_architecture" in checkpoint:
            checkpoint_architecture = checkpoint["network_architecture"]
        elif "network_architecture" in checkpoint_config:
            checkpoint_architecture = checkpoint_config["network_architecture"]
        else:
            if self.network_architecture != MONOLITHIC_ARCHITECTURE:
                raise ValueError(
                    "Чекпоинт не содержит метаданные архитектуры и несовместим с card_context_v1"
                )
            return
        if not isinstance(checkpoint_architecture, str):
            raise ValueError("Чекпоинт имеет некорректное значение архитектуры сети")
        if checkpoint_architecture != self.network_architecture:
            raise ValueError("Чекпоинт имеет несовместимую архитектуру сети")
        if checkpoint_architecture == CARD_CONTEXT_ARCHITECTURE:
            checkpoint_card_feature_size = checkpoint.get(
                "card_feature_size",
                checkpoint_config.get("card_feature_size"),
            )
            if checkpoint_card_feature_size != CARD_FEATURE_SIZE:
                raise ValueError("Чекпоинт имеет несовместимый размер card-признаков")

    @staticmethod
    def _network_hidden_size_from_state(state_dict):
        weight = state_dict.get("base.0.weight")
        if weight is None:
            raise ValueError("Teacher strategy state_dict не содержит base.0.weight")
        return int(weight.shape[0])

    def set_teacher_strategy_network(
        self,
        state_dict,
        num_players=None,
        use_multi_agent=None,
        encoding_version=None,
    ):
        """Подключает замороженную strategy-сеть teacher-а для мягкой дистилляции."""
        teacher_num_players = int(num_players if num_players is not None else self.num_players)
        teacher_use_multi_agent = bool(self.use_multi_agent if use_multi_agent is None else use_multi_agent)
        teacher_encoding_version = str(encoding_version or self.encoding_version)
        teacher_input_size = self._network_input_size(
            teacher_num_players,
            teacher_use_multi_agent,
            teacher_encoding_version,
        )
        teacher = PokerNetwork(
            teacher_input_size,
            self._network_hidden_size_from_state(state_dict),
            NUM_ACTIONS,
        )
        teacher.load_state_dict(state_dict, strict=True)
        teacher.to(self.device)
        teacher.eval()
        for parameter in teacher.parameters():
            parameter.requires_grad_(False)
        self.teacher_strategy_net = teacher
        self.teacher_strategy_input_size = teacher_input_size
        self.teacher_strategy_num_players = teacher_num_players
        self.teacher_strategy_use_multi_agent = teacher_use_multi_agent
        self.teacher_strategy_encoding_version = teacher_encoding_version

    def load_teacher_strategy_checkpoint(self, path):
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        if checkpoint.get("checkpoint_format_version") != CHECKPOINT_FORMAT_VERSION:
            raise ValueError("Teacher checkpoint имеет несовместимую версию формата")
        if checkpoint.get("action_space_version") != ACTION_SPACE_VERSION:
            raise ValueError("Teacher checkpoint имеет другое пространство действий")
        if int(checkpoint.get("num_actions", -1)) != NUM_ACTIONS:
            raise ValueError("Teacher checkpoint имеет другое число действий")
        teacher_num_players = int(checkpoint.get("num_players", self.num_players))
        checkpoint_config = checkpoint.get("config", {})
        if not isinstance(checkpoint_config, dict):
            checkpoint_config = {}
        teacher_architecture = checkpoint.get(
            "network_architecture",
            checkpoint_config.get("network_architecture"),
        )
        if (
            teacher_num_players == 2
            and self.num_players != 2
            and teacher_architecture == CARD_CONTEXT_ARCHITECTURE
        ):
            raise ValueError(
                "HU projection teacher checkpoint с card_context_v1 не поддерживается"
            )
        teacher_use_multi_agent = bool(checkpoint_config.get("use_multi_agent_advantage", False))
        teacher_encoding_version = checkpoint.get(
            "encoding_version",
            checkpoint_config.get("encoding_version"),
        )
        if teacher_encoding_version != self.encoding_version:
            raise ValueError("Teacher checkpoint имеет несовместимую версию encoder")
        if teacher_num_players not in (self.num_players, 2):
            raise ValueError("Teacher checkpoint поддержан только для того же стола или HU projection")
        if teacher_num_players == self.num_players and teacher_use_multi_agent != self.use_multi_agent:
            raise ValueError("Teacher checkpoint имеет несовместимый multi-agent режим")
        teacher_input_size = self._network_input_size(
            teacher_num_players,
            teacher_use_multi_agent,
            teacher_encoding_version,
        )
        if int(checkpoint.get("encoder_input_size", -1)) != teacher_input_size:
            raise ValueError("Teacher checkpoint имеет несовместимый размер входа encoder")
        strategy_state = checkpoint.get("strategy_net")
        if not isinstance(strategy_state, dict):
            raise ValueError("В teacher checkpoint нет strategy_net")
        if "base.0.weight" not in strategy_state or int(strategy_state["base.0.weight"].shape[1]) != teacher_input_size:
            raise ValueError("Teacher checkpoint имеет несовместимый размер входа encoder в strategy_net")
        self.set_teacher_strategy_network(
            strategy_state,
            num_players=teacher_num_players,
            use_multi_agent=teacher_use_multi_agent,
            encoding_version=teacher_encoding_version,
        )
        return checkpoint

    def _effective_strategy_distillation_lambda(self, iteration_now):
        base = max(float(self.strategy_distillation_lambda), 0.0)
        anneal_iterations = int(self.strategy_distillation_anneal_iterations)
        if base <= 0.0 or anneal_iterations <= 0:
            return base
        progress = min(max(float(iteration_now), 0.0), float(anneal_iterations))
        return base * max(0.0, 1.0 - progress / float(anneal_iterations))

    @staticmethod
    def _layout_offsets(num_players):
        hand = 0
        community = hand + 52
        stage = community + 52
        pot = stage + 5
        button = pot + 1
        current_player = button + int(num_players)
        players = current_player + int(num_players)
        tail = players + int(num_players) * 4
        return hand, community, stage, pot, button, current_player, players, tail

    def _project_states_for_teacher_strategy(self, state_t):
        if self.teacher_strategy_num_players is None:
            return state_t
        if (
            self.teacher_strategy_num_players == self.num_players
            and self.teacher_strategy_use_multi_agent == self.use_multi_agent
            and self.teacher_strategy_encoding_version == self.encoding_version
        ):
            return state_t
        if self.teacher_strategy_num_players != 2:
            raise ValueError("Неподдерживаемый teacher projection")

        hand, community, stage, pot, button, current_player, players, tail = self._layout_offsets(self.num_players)
        player_blocks = state_t[:, players:tail].reshape(state_t.shape[0], self.num_players, 4)
        opponent_active = player_blocks[:, 1:, 0] > 0.5
        fallback = torch.ones(state_t.shape[0], dtype=torch.long, device=state_t.device)
        first_active = torch.argmax(opponent_active.float(), dim=1) + 1
        opponent_offsets = torch.where(opponent_active.any(dim=1), first_active, fallback)
        batch_offsets = torch.arange(state_t.shape[0], device=state_t.device)
        hero_block = player_blocks[:, 0, :]
        opponent_block = player_blocks[batch_offsets, opponent_offsets, :]

        button_offsets = torch.argmax(state_t[:, button:current_player], dim=1)
        current_offsets = torch.argmax(state_t[:, current_player:players], dim=1)
        button_hu = torch.zeros((state_t.shape[0], 2), dtype=state_t.dtype, device=state_t.device)
        current_hu = torch.zeros((state_t.shape[0], 2), dtype=state_t.dtype, device=state_t.device)
        button_hu[:, 0] = (button_offsets == 0).to(state_t.dtype)
        button_hu[:, 1] = 1.0 - button_hu[:, 0]
        current_hu[:, 0] = (current_offsets == 0).to(state_t.dtype)
        current_hu[:, 1] = 1.0 - current_hu[:, 0]

        tail_features = state_t[:, tail:tail + 11]
        projected = [
            state_t[:, hand:community],
            state_t[:, community:stage],
            state_t[:, stage:pot],
            state_t[:, pot:button],
            button_hu,
            current_hu,
            hero_block,
            opponent_block,
            tail_features,
        ]
        if self.encoding_version == HISTORY_SUMMARY_V3_ENCODING_VERSION:
            source_history_start = legacy_base_input_size(self.num_players)
            source_block_size = 2 * self.num_players + 8
            source_history = state_t[:, source_history_start:source_history_start + history_summary_size(self.num_players)]
            source_history = source_history.reshape(state_t.shape[0], 4, source_block_size)
            projected_history = []
            for street_history in source_history.unbind(dim=1):
                last_actor = street_history[:, :self.num_players + 1]
                action = street_history[:, self.num_players + 1:self.num_players + 6]
                raise_actors = street_history[:, self.num_players + 6:self.num_players * 2 + 6]
                scalars = street_history[:, self.num_players * 2 + 6:]
                projected_last_actor = torch.stack(
                    (
                        last_actor[:, 0],
                        last_actor[:, 1:self.num_players].sum(dim=1).clamp(max=1.0),
                        last_actor[:, self.num_players],
                    ),
                    dim=1,
                )
                projected_raise_actors = torch.stack(
                    (
                        raise_actors[:, 0],
                        raise_actors[:, 1:].amax(dim=1),
                    ),
                    dim=1,
                )
                projected_history.append(
                    torch.cat((projected_last_actor, action, projected_raise_actors, scalars), dim=1)
                )
            projected.extend(projected_history)
        if self.teacher_strategy_use_multi_agent:
            teacher_position = torch.zeros((state_t.shape[0], 2), dtype=state_t.dtype, device=state_t.device)
            teacher_position[:, 0] = 1.0
            projected.append(teacher_position)
        return torch.cat(projected, dim=1)

    @staticmethod
    def _full_epoch_batches(size, batch_size, epochs):
        size = int(size)
        if size <= 0:
            return
        batch_size = max(1, min(int(batch_size), size))
        for epoch in range(max(0, int(epochs))):
            order = np.random.permutation(size)
            for start in range(0, size, batch_size):
                yield epoch, order[start:start + batch_size]

    @staticmethod
    def _fixed_step_batches(size, batch_size, steps):
        size = int(size)
        if size <= 0:
            return
        batch_size = max(1, int(batch_size))
        for step in range(max(0, int(steps))):
            yield step, np.random.choice(size, batch_size, replace=True)

    def _synchronize_training_device(self):
        if self.device.type == "cuda" and torch.cuda.is_available():
            torch.cuda.synchronize(self.device)

    def _preload_training_arrays(self, arrays, label):
        if not self.training_preload_to_device:
            return None
        try:
            return tuple(torch.as_tensor(array, device=self.device) for array in arrays)
        except RuntimeError as exc:
            if self.device.type != "cuda":
                raise
            print(f"[Training] Preload {label} на GPU не удался, использую обычные батчи: {exc}")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            return None

    def _training_indices(self, size, batch_size):
        return torch.randint(int(size), (max(1, int(batch_size)),), device=self.device)

    def _encode_state(self, state, player_id):
        if self.use_multi_agent:
            return encode_state_with_position(state, player_id, self.encoding_version)
        return encode_state_for_version(state, player_id, self.encoding_version)

    @staticmethod
    def _regret_matching(advantages, mask):
        positive = np.maximum(np.asarray(advantages, dtype=np.float32), 0.0) * mask
        total = float(positive.sum())
        if total > 0.0:
            return positive / total
        legal_count = float(mask.sum())
        return mask / legal_count if legal_count else np.zeros(NUM_ACTIONS, dtype=np.float32)

    def _validate_finite_traversal_values(
        self,
        values,
        value_name,
        mask,
        iteration,
        traversing_player,
        acting_player,
        depth,
        policy_source,
    ):
        details = {
            "policy_source": policy_source,
            "mask": np.asarray(mask, dtype=np.float32).tolist(),
        }
        try:
            values = np.asarray(values, dtype=np.float32)
        except (TypeError, ValueError, OverflowError) as error:
            self._raise_traversal_failure(
                iteration,
                traversing_player,
                acting_player,
                depth,
                f"Не удалось прочитать {value_name} policy {policy_source}",
                details=details,
                cause=error,
            )
        details[value_name] = values.tolist()
        if values.shape != (NUM_ACTIONS,):
            self._raise_traversal_failure(
                iteration,
                traversing_player,
                acting_player,
                depth,
                f"Некорректная форма {value_name} policy {policy_source}",
                details=details,
            )
        if not np.all(np.isfinite(values)):
            self._raise_traversal_failure(
                iteration,
                traversing_player,
                acting_player,
                depth,
                f"Неконечные {value_name} policy {policy_source}",
                details=details,
            )
        return values

    def _validate_traversal_policy(
        self,
        policy,
        mask,
        iteration,
        traversing_player,
        acting_player,
        depth,
        policy_source,
    ):
        policy = self._validate_finite_traversal_values(
            policy,
            "policy",
            mask,
            iteration,
            traversing_player,
            acting_player,
            depth,
            policy_source,
        )
        legal_mask = np.asarray(mask, dtype=np.float32) > 0.0
        details = {
            "policy_source": policy_source,
            "mask": np.asarray(mask, dtype=np.float32).tolist(),
            "policy": policy.tolist(),
        }
        if legal_mask.shape != (NUM_ACTIONS,) or not legal_mask.any():
            self._raise_traversal_failure(
                iteration,
                traversing_player,
                acting_player,
                depth,
                f"Некорректная mask для policy {policy_source}",
                details=details,
            )
        if np.any(policy < 0.0):
            self._raise_traversal_failure(
                iteration,
                traversing_player,
                acting_player,
                depth,
                f"Policy {policy_source} содержит отрицательные вероятности",
                details=details,
            )
        if not np.allclose(policy[~legal_mask], 0.0, atol=1e-6, rtol=0.0):
            self._raise_traversal_failure(
                iteration,
                traversing_player,
                acting_player,
                depth,
                f"Policy {policy_source} содержит вероятность недопустимого действия",
                details=details,
            )
        if not np.isclose(policy[legal_mask].sum(), 1.0, atol=1e-6, rtol=0.0):
            self._raise_traversal_failure(
                iteration,
                traversing_player,
                acting_player,
                depth,
                f"Policy {policy_source} не нормирована по допустимым действиям",
                details=details,
            )
        return policy

    def _advantage_policy(self, state, player_id):
        mask = self.get_legal_action_mask(state)
        if not mask.any():
            return mask, np.zeros(NUM_ACTIONS, dtype=np.float32)
        encoded = self._encode_state(state, player_id)
        state_t = torch.from_numpy(encoded).float().unsqueeze(0).to(self.device)
        with torch.inference_mode():
            advantages = self.advantage_net(state_t)[0].cpu().numpy()
        advantages = self._validate_finite_traversal_values(
            advantages,
            "advantages",
            mask,
            self.iteration_count,
            player_id,
            player_id,
            0,
            "advantage_net",
        )
        policy = self._regret_matching(advantages, mask)
        return mask, self._validate_traversal_policy(
            policy,
            mask,
            self.iteration_count,
            player_id,
            player_id,
            0,
            "advantage_net",
        )

    @staticmethod
    def _is_random_opponent_turn(random_agent, current_player, traversing_player):
        return random_agent is not None and int(current_player) != int(traversing_player)

    def set_opponent_advantage_states(self, state_dicts, traversing_player):
        """Устанавливает замороженные advantage-сети оппонентов для одной итерации."""
        opponent_ids = [
            player_id for player_id in range(self.num_players)
            if player_id != int(traversing_player)
        ]
        if len(state_dicts) != len(opponent_ids):
            raise ValueError(
                "Число состояний оппонентов не совпадает с числом мест за столом"
            )

        networks: dict[int, PokerNetwork] = {}
        for player_id, state_dict in zip(opponent_ids, state_dicts, strict=True):
            network = PokerNetwork(
                self.input_size,
                self._network_hidden_size(self.advantage_net),
                NUM_ACTIONS,
                self.network_architecture,
            )
            network.load_state_dict(state_dict, strict=True)
            network.to(self.device)
            network.eval()
            for parameter in network.parameters():
                parameter.requires_grad_(False)
            networks[player_id] = network
        self._opponent_advantage_nets = networks

    def set_opponent_strategy_states(self, state_dicts, traversing_player):
        """Устанавливает замороженные strategy-сети оппонентов для одной итерации."""
        opponent_ids = [
            player_id for player_id in range(self.num_players)
            if player_id != int(traversing_player)
        ]
        if len(state_dicts) != len(opponent_ids):
            raise ValueError(
                "Число состояний оппонентов не совпадает с числом мест за столом"
            )

        self.set_opponent_strategy_states_by_player(
            dict(zip(opponent_ids, state_dicts, strict=True)), traversing_player
        )

    def set_opponent_strategy_states_by_player(self, state_dicts, traversing_player):
        """Устанавливает strategy-сети только на указанные позиции оппонентов."""
        opponent_ids = {
            player_id for player_id in range(self.num_players)
            if player_id != int(traversing_player)
        }
        if not set(state_dicts).issubset(opponent_ids):
            raise ValueError("Strategy-сеть назначена traversing player или несуществующей позиции")

        networks: dict[int, PokerNetwork] = {}
        for player_id, state_dict in state_dicts.items():
            network = PokerNetwork(
                self.input_size,
                self._network_hidden_size(self.strategy_net),
                NUM_ACTIONS,
                self.network_architecture,
            )
            network.load_state_dict(state_dict, strict=True)
            network.to(self.device)
            network.eval()
            for parameter in network.parameters():
                parameter.requires_grad_(False)
            networks[player_id] = network
        self._opponent_strategy_nets = networks

    def set_opponent_policy_agents(self, agents_by_player, traversing_player):
        """Устанавливает внешние политики на подмножество позиций оппонентов."""
        opponent_ids = {
            player_id for player_id in range(self.num_players)
            if player_id != int(traversing_player)
        }
        if not set(agents_by_player).issubset(opponent_ids):
            raise ValueError("Внешняя политика назначена traversing player или несуществующей позиции")
        self._opponent_policy_agents = dict(agents_by_player)

    def clear_opponent_advantage_states(self):
        """Возвращает оппонентов к общей advantage-сети для обратной совместимости."""
        self._opponent_advantage_nets = {}
        self._opponent_strategy_nets = {}
        self._opponent_policy_agents = {}

    @staticmethod
    def _masked_softmax(logits, mask):
        """Возвращает softmax только по допустимым слотам действий."""
        mask_t = torch.as_tensor(mask, dtype=logits.dtype, device=logits.device)
        if mask_t.dim() == 1:
            mask_t = mask_t.unsqueeze(0)
        if logits.dim() == 1:
            logits = logits.unsqueeze(0)
        masked_logits = torch.where(mask_t > 0.0, logits, torch.full_like(logits, -1e20))
        return F.softmax(masked_logits, dim=1)

    def action_type_to_pokers_action(self, action_type, state):
        """Конвертирует слот в точное действие движка без fallback."""
        return resolve_action(action_type, state).action

    def get_legal_action_types(self, state):
        return np.flatnonzero(self.get_legal_action_mask(state)).astype(int).tolist()

    def get_legal_action_mask(self, state):
        return legal_action_mask(state)

    def prepare_iteration(self, iteration, traversing_player):
        del iteration, traversing_player
        if not self.advantage_buffer_reservoir:
            self.advantage_buffer.clear()
        if self.clear_strategy_buffer_each_iteration:
            self.strategy_buffer.clear()

    def reset_traversal_stats(self):
        self.traversal_attempts = 0
        self.traversal_successes = 0
        self.traversal_failures = 0
        self.traversal_failure_reasons = {}
        self.traversal_failure_diagnostics = []
        self.traversal_cancelled_samples = 0
        self.traversal_depth_limit_failures = 0
        self.traversal_nodes = 0
        self.traversal_terminal_nodes = 0
        self.traversal_max_depth_observed = 0
        self.traversal_max_depth_hits = 0
        self.traversal_traversing_decision_nodes = 0
        self.traversal_opponent_decision_nodes = 0
        self.traverser_children_total = 0
        self.opponent_children_total = 0
        self.traverser_child_fanout = {}
        self.opponent_child_fanout = {}
        self.action_decision_count = 0
        self.action_raise_count = 0
        self.recorded_nodes = 0
        self.evicted_nodes = 0
        self.buffer_skip_nodes = 0
        self.depth_histogram = {}

    def record_traversal_attempt(self):
        self.traversal_attempts += 1

    def record_traversal_success(self):
        self.traversal_successes += 1

    def record_traversal_failure(self, error: TraversalFailure) -> dict[str, object]:
        if not isinstance(error, TraversalFailure):
            raise TypeError("Ожидается TraversalFailure")
        context = error.context
        reason = context.reason
        self.traversal_failures += 1
        self.traversal_failure_reasons[reason] = self.traversal_failure_reasons.get(reason, 0) + 1
        if "глубин" in reason.lower():
            self.traversal_depth_limit_failures += 1
        diagnostic = {
            "iteration": context.iteration,
            "traversal_index": context.traversal_index,
            "traversing_player": context.traversing_player,
            "acting_player": context.acting_player,
            "reason": reason,
            "depth": context.depth,
            "action_trace": context.action_trace,
        }
        diagnostic.update(
            (key, value) for key, value in context.details.items() if key not in diagnostic
        )
        self.traversal_failure_diagnostics.append(diagnostic)
        return diagnostic

    def _record_child_fanout(self, role, count):
        if role == "traverser":
            self.traverser_children_total += count
            histogram = self.traverser_child_fanout
        else:
            self.opponent_children_total += count
            histogram = self.opponent_child_fanout
        histogram[count] = histogram.get(count, 0) + 1

    def _raise_invalid_training_sample(self, iteration, reason):
        raise TraversalFailure(
            TraversalFailureContext(
                iteration=iteration,
                traversal_index=self._active_traversal_index,
                traversing_player=self.player_id,
                acting_player=None,
                depth=0,
                reason=reason,
            )
        )

    def _validate_training_sample(self, state, values, mask, iteration, sample_kind):
        if sample_kind not in {"advantage", "strategy"}:
            self._raise_invalid_training_sample(iteration, "Неизвестный тип training sample")
        if not isinstance(iteration, Integral) or isinstance(iteration, bool) or iteration < 1:
            self._raise_invalid_training_sample(iteration, "Iteration должен быть целым числом не меньше 1")
        try:
            iteration_value = float(iteration)
        except (OverflowError, ValueError):
            self._raise_invalid_training_sample(iteration, "Iteration не преобразуется в конечное число")
        if not math.isfinite(iteration_value):
            self._raise_invalid_training_sample(iteration, "Iteration не преобразуется в конечное число")
        for name, value, shape in (
            ("state", state, (self.input_size,)),
            ("values", values, (NUM_ACTIONS,)),
            ("mask", mask, (NUM_ACTIONS,)),
        ):
            if not isinstance(value, np.ndarray) or value.dtype != np.float32:
                self._raise_invalid_training_sample(iteration, f"{name} должен быть numpy array с dtype float32")
            if value.shape != shape:
                self._raise_invalid_training_sample(iteration, f"{name} имеет неверную форму {value.shape}")
            if not np.all(np.isfinite(value)):
                self._raise_invalid_training_sample(iteration, f"{name} содержит нечисловые значения")
        if not np.all(np.isin(mask, (0.0, 1.0))):
            self._raise_invalid_training_sample(iteration, "Mask должен содержать только 0 или 1")
        legal_actions = mask == 1.0
        if not np.any(legal_actions):
            self._raise_invalid_training_sample(iteration, "Mask не содержит допустимых действий")
        if sample_kind == "strategy":
            if np.any(values < 0.0):
                self._raise_invalid_training_sample(iteration, "Strategy не может содержать отрицательные вероятности")
            if not np.allclose(values[~legal_actions], 0.0, atol=1e-6, rtol=0.0):
                self._raise_invalid_training_sample(iteration, "Strategy содержит вероятность недопустимого действия")
            if not np.isclose(values[legal_actions].sum(), 1.0, atol=1e-6, rtol=0.0):
                self._raise_invalid_training_sample(iteration, "Сумма strategy по допустимым действиям должна быть равна 1")

    def _record_advantage_sample(self, state, regrets, mask, iteration):
        self._validate_training_sample(state, regrets, mask, iteration, "advantage")
        collector = self._active_traversal_collector
        if collector is not None:
            collector.advantage_samples.append((state, regrets, mask, iteration))
            return
        self._add_advantage_sample(state, regrets, mask, iteration)

    def _add_advantage_sample(self, state, regrets, mask, iteration):
        status = self.advantage_buffer.add(state, regrets, mask, iteration)
        if status == "recorded":
            self.recorded_nodes += 1
        elif status == "evicted":
            self.recorded_nodes += 1
            self.evicted_nodes += 1
        else:
            self.buffer_skip_nodes += 1

    def _record_strategy_sample(self, state, strategy, mask, iteration):
        self._validate_training_sample(state, strategy, mask, iteration, "strategy")
        collector = self._active_traversal_collector
        if collector is not None:
            collector.strategy_samples.append((state, strategy, mask, iteration))
            return
        self.strategy_buffer.add(state, strategy, mask, iteration)

    def _commit_traversal_collector(self, collector):
        for sample in collector.advantage_samples:
            self._validate_training_sample(*sample, "advantage")
        for sample in collector.strategy_samples:
            self._validate_training_sample(*sample, "strategy")
        for sample in collector.advantage_samples:
            self._add_advantage_sample(*sample)
        for sample in collector.strategy_samples:
            self.strategy_buffer.add(*sample)

    def get_traversal_stats(self):
        attempts = self.recorded_nodes + self.buffer_skip_nodes
        decisions = self.action_decision_count
        return {
            "attempted": self.traversal_attempts,
            "successful": self.traversal_successes,
            "failed": self.traversal_failures,
            "failure_reasons": dict(self.traversal_failure_reasons),
            "failure_diagnostics": [dict(diagnostic) for diagnostic in self.traversal_failure_diagnostics],
            "cancelled_samples": self.traversal_cancelled_samples,
            "depth_limit_hits": max(
                self.traversal_max_depth_hits,
                self.traversal_depth_limit_failures,
            ),
            "nodes": self.traversal_nodes,
            "terminal_nodes": self.traversal_terminal_nodes,
            "max_depth": self.traversal_max_depth_observed,
            "max_depth_hits": self.traversal_max_depth_hits,
            "traversing_decision_nodes": self.traversal_traversing_decision_nodes,
            "opponent_decision_nodes": self.traversal_opponent_decision_nodes,
            "recorded_nodes": self.recorded_nodes,
            "evicted_nodes": self.evicted_nodes,
            "buffer_skip_nodes": self.buffer_skip_nodes,
            "buffer_attempted_nodes": attempts,
            "buffer_skip_ratio": self.buffer_skip_nodes / attempts if attempts else 0.0,
            "buffer_waste_ratio": (self.evicted_nodes + self.buffer_skip_nodes) / attempts if attempts else 0.0,
            "traverser_children_total": self.traverser_children_total,
            "opponent_children_total": self.opponent_children_total,
            "traverser_children_per_node": self.traverser_children_total / self.traversal_traversing_decision_nodes if self.traversal_traversing_decision_nodes else 0.0,
            "opponent_children_per_node": self.opponent_children_total / self.traversal_opponent_decision_nodes if self.traversal_opponent_decision_nodes else 0.0,
            "traverser_child_fanout": dict(self.traverser_child_fanout),
            "opponent_child_fanout": dict(self.opponent_child_fanout),
            "action_decisions": decisions,
            "action_raise_count": self.action_raise_count,
            "opponent_raise_frequency": self.action_raise_count / decisions if decisions else 0.0,
        }

    def _normalise_regrets(self, regrets, state, legal_slots):
        result = regrets.astype(np.float32, copy=True)
        if self.advantage_regret_norm == "pot_stack":
            player = state.players_state[int(state.current_player)]
            result /= max(float(state.pot) + float(player.stake), 1.0)
        elif self.advantage_regret_norm == "per_node_max" and legal_slots:
            result /= max(float(np.max(np.abs(result[legal_slots]))), 1.0)
        if self.advantage_regret_clip is not None:
            result = np.clip(result, -self.advantage_regret_clip, self.advantage_regret_clip)
        if self.advantage_reward_scale != 1.0:
            result /= self.advantage_reward_scale
        return result.astype(np.float32)

    def cfr_traverse(self, state, iteration, random_agents, depth=0):
        random_agent = random_agents[-1] if random_agents else None
        return self.cfr_traverse_multi(state, iteration, self.player_id, depth, random_agent)

    def cfr_traverse_multi(
        self,
        state,
        iteration,
        traversing_player,
        depth=0,
        random_agent=None,
        traversal_index=None,
    ):
        if depth != 0:
            return self._cfr_traverse_multi(state, iteration, traversing_player, depth)
        collector = _TraversalSampleCollector()
        previous_traversal_index = self._active_traversal_index
        self._active_traversal_collector = collector
        self._traversal_random_agent = random_agent
        self._active_traversal_index = traversal_index
        try:
            result = self._cfr_traverse_multi(state, iteration, traversing_player, depth)
            self._commit_traversal_collector(collector)
            return result
        except TraversalFailure:
            self.traversal_cancelled_samples += (
                len(collector.advantage_samples) + len(collector.strategy_samples)
            )
            raise
        finally:
            self._active_traversal_collector = None
            self._traversal_random_agent = None
            self._active_traversal_index = previous_traversal_index

    def _raise_traversal_failure(
        self,
        iteration,
        traversing_player,
        acting_player,
        depth,
        reason,
        action_description=None,
        cause=None,
        details=None,
    ) -> NoReturn:
        context_details = {} if details is None else dict(details)
        if action_description is not None:
            context_details.setdefault("action", action_description)
        context = TraversalFailureContext(
            iteration=iteration,
            traversal_index=self._active_traversal_index,
            traversing_player=traversing_player,
            acting_player=acting_player,
            depth=depth,
            reason=reason,
            action_trace=() if action_description is None else (action_description,),
            details=context_details,
        )
        error = TraversalFailure(context, cause)
        if cause is None:
            raise error
        raise error from cause

    def _validate_transition(
        self,
        state,
        next_state,
        action,
        action_description,
        iteration,
        traversing_player,
        acting_player,
        depth,
    ) -> None:
        try:
            next_status = next_state.status
        except (AttributeError, TypeError) as error:
            self._raise_traversal_failure(
                iteration,
                traversing_player,
                acting_player,
                depth,
                f"Не удалось получить status следующего состояния после действия {action_description}",
                action_description,
                cause=error,
                details={"status": None},
            )
        if next_status == pkrs.StateStatus.Ok:
            return
        log_game_error(state, action, f"State status not OK ({next_status})")
        self._raise_traversal_failure(
            iteration,
            traversing_player,
            acting_player,
            depth,
            f"Недопустимый status состояния после действия {action_description}: {next_status}",
            action_description,
            details={"status": str(next_status)},
        )

    def _cfr_traverse_multi(self, state, iteration, traversing_player, depth):
        self.traversal_nodes += 1
        self.traversal_max_depth_observed = max(self.traversal_max_depth_observed, depth)
        self.depth_histogram[depth] = self.depth_histogram.get(depth, 0) + 1
        is_terminal = bool(state.final_state)
        current_player = None if is_terminal else int(state.current_player)
        if depth > 200:
            self.traversal_max_depth_hits += 1
            self._raise_traversal_failure(
                iteration,
                traversing_player,
                current_player,
                depth,
                f"Превышена допустимая глубина обхода: depth={depth}",
            )
        if is_terminal:
            self.traversal_terminal_nodes += 1
            return float(state.players_state[traversing_player].reward)

        external_policy = self._opponent_policy_agents.get(current_player)
        if external_policy is not None:
            try:
                action = external_policy.choose_action(state)
            except (AttributeError, TypeError, ValueError, RuntimeError) as error:
                self._raise_traversal_failure(
                    iteration,
                    traversing_player,
                    current_player,
                    depth,
                    "Не удалось выбрать действие внешней policy",
                    "external_policy",
                    error,
                )
            try:
                next_state = state.apply_action(action)
            except (AttributeError, TypeError, ValueError, RuntimeError) as error:
                self._raise_traversal_failure(
                    iteration,
                    traversing_player,
                    current_player,
                    depth,
                    "Ошибка apply_action для действия внешней policy",
                    f"external_policy: {action!r}",
                    error,
                )
            self._validate_transition(
                state,
                next_state,
                action,
                f"external_policy: {action!r}",
                iteration,
                traversing_player,
                current_player,
                depth,
            )
            return self._cfr_traverse_multi(next_state, iteration, traversing_player, depth + 1)
        if self._is_random_opponent_turn(
            self._traversal_random_agent, current_player, traversing_player
        ):
            try:
                action = self._traversal_random_agent.choose_action(state)
            except (AttributeError, TypeError, ValueError, RuntimeError) as error:
                self._raise_traversal_failure(
                    iteration,
                    traversing_player,
                    current_player,
                    depth,
                    "Не удалось выбрать действие случайного оппонента",
                    "random_agent",
                    error,
                )
            try:
                next_state = state.apply_action(action)
            except (AttributeError, TypeError, ValueError, RuntimeError) as error:
                self._raise_traversal_failure(
                    iteration,
                    traversing_player,
                    current_player,
                    depth,
                    "Ошибка apply_action для действия случайного оппонента",
                    f"random_agent: {action!r}",
                    error,
                )
            self._validate_transition(
                state,
                next_state,
                action,
                f"random_agent: {action!r}",
                iteration,
                traversing_player,
                current_player,
                depth,
            )
            return self._cfr_traverse_multi(next_state, iteration, traversing_player, depth + 1)

        legal_mask = self.get_legal_action_mask(state)
        legal_slots = np.flatnonzero(legal_mask).astype(int).tolist()
        if not legal_slots:
            self._raise_traversal_failure(
                iteration,
                traversing_player,
                current_player,
                depth,
                "У нетерминального состояния отсутствуют допустимые действия",
                details={"mask": np.asarray(legal_mask, dtype=np.float32).tolist()},
            )

        if current_player == traversing_player:
            self.traversal_traversing_decision_nodes += 1
            encoded = self._encode_state(state, traversing_player).astype(np.float32, copy=False)
            state_t = torch.from_numpy(encoded).unsqueeze(0).to(self.device)
            with torch.inference_mode():
                advantages = self.advantage_net(state_t)[0].cpu().numpy()
            advantages = self._validate_finite_traversal_values(
                advantages,
                "advantages",
                legal_mask,
                iteration,
                traversing_player,
                current_player,
                depth,
                "advantage_net",
            )
            action_values = np.zeros(NUM_ACTIONS, dtype=np.float32)
            applied_slots = []
            for slot in legal_slots:
                try:
                    action = self.action_type_to_pokers_action(slot, state)
                except (AttributeError, TypeError, ValueError, RuntimeError) as error:
                    self._raise_traversal_failure(
                        iteration,
                        traversing_player,
                        current_player,
                        depth,
                        f"Не удалось преобразовать действие slot {slot}",
                        f"slot {slot}",
                        error,
                    )
                try:
                    with profile_section(TRAVERSAL_PROFILER, "clone_state"):
                        next_state = state.apply_action(action)
                except (AttributeError, TypeError, ValueError, RuntimeError) as error:
                    self._raise_traversal_failure(
                        iteration,
                        traversing_player,
                        current_player,
                        depth,
                        f"Ошибка apply_action для slot {slot}",
                        f"slot {slot}: {action!r}",
                        error,
                    )
                self._validate_transition(
                    state,
                    next_state,
                    action,
                    f"slot {slot}: {action!r}",
                    iteration,
                    traversing_player,
                    current_player,
                    depth,
                )
                action_values[slot] = self._cfr_traverse_multi(next_state, iteration, traversing_player, depth + 1)
                applied_slots.append(slot)
            self._record_child_fanout("traverser", len(applied_slots))
            applied_mask = np.zeros(NUM_ACTIONS, dtype=np.float32)
            applied_mask[applied_slots] = 1.0
            strategy = self._regret_matching(advantages, applied_mask)
            strategy = self._validate_traversal_policy(
                strategy,
                applied_mask,
                iteration,
                traversing_player,
                current_player,
                depth,
                "advantage_net",
            )
            ev = float(sum(float(strategy[slot]) * float(action_values[slot]) for slot in applied_slots))
            regrets = np.zeros(NUM_ACTIONS, dtype=np.float32)
            for slot in applied_slots:
                regrets[slot] = action_values[slot] - ev
            regrets = self._normalise_regrets(regrets, state, applied_slots)
            self._record_advantage_sample(encoded, regrets, applied_mask, iteration)
            self._record_strategy_sample(encoded, strategy, applied_mask, iteration)
            return ev

        self.traversal_opponent_decision_nodes += 1
        mask = legal_mask
        encoded = self._encode_state(state, current_player)
        state_t = torch.from_numpy(encoded).float().unsqueeze(0).to(self.device)
        strategy_net = self._opponent_strategy_nets.get(current_player)
        if strategy_net is not None:
            with torch.inference_mode():
                strategy = self._masked_softmax(strategy_net(state_t), mask)[0].cpu().numpy()
            strategy = self._validate_traversal_policy(
                strategy,
                mask,
                iteration,
                traversing_player,
                current_player,
                depth,
                "strategy_net",
            )
        else:
            opponent_net = self._opponent_advantage_nets.get(current_player, self.advantage_net)
            with torch.inference_mode():
                advantages = opponent_net(state_t)[0].cpu().numpy()
            advantages = self._validate_finite_traversal_values(
                advantages,
                "advantages",
                mask,
                iteration,
                traversing_player,
                current_player,
                depth,
                "advantage_net",
            )
            strategy = self._regret_matching(advantages, mask)
            strategy = self._validate_traversal_policy(
                strategy,
                mask,
                iteration,
                traversing_player,
                current_player,
                depth,
                "advantage_net",
            )
        weights = strategy[legal_slots]
        try:
            slot = int(np.random.choice(legal_slots, p=weights / weights.sum()))
        except (TypeError, ValueError) as error:
            self._raise_traversal_failure(
                iteration,
                traversing_player,
                current_player,
                depth,
                "Не удалось выбрать действие по policy оппонента",
                details={
                    "policy_source": "opponent_sampling",
                    "mask": np.asarray(mask, dtype=np.float32).tolist(),
                    "policy": strategy.tolist(),
                },
                cause=error,
            )
        self.action_decision_count += 1
        if is_raise_slot(slot):
            self.action_raise_count += 1
        try:
            action = self.action_type_to_pokers_action(slot, state)
        except (AttributeError, TypeError, ValueError, RuntimeError) as error:
            self._raise_traversal_failure(
                iteration,
                traversing_player,
                current_player,
                depth,
                f"Не удалось преобразовать действие slot {slot} оппонента",
                f"slot {slot}",
                error,
            )
        try:
            next_state = state.apply_action(action)
        except (AttributeError, TypeError, ValueError, RuntimeError) as error:
            self._raise_traversal_failure(
                iteration,
                traversing_player,
                current_player,
                depth,
                f"Ошибка apply_action для slot {slot} оппонента",
                f"slot {slot}: {action!r}",
                error,
            )
        self._validate_transition(
            state,
            next_state,
            action,
            f"slot {slot}: {action!r}",
            iteration,
            traversing_player,
            current_player,
            depth,
        )
        self._record_child_fanout("opponent", 1)
        return self._cfr_traverse_multi(next_state, iteration, traversing_player, depth + 1)

    def train_advantage_network(self, *args, **kwargs):
        return self.train_advantage_network_multi(*args, **kwargs)

    def _assert_finite_training_tensors(self, loss, parameters, optimizer, stage):
        loss_is_finite = bool(torch.isfinite(loss).all())
        gradients_are_finite = all(
            parameter.grad is None or bool(torch.isfinite(parameter.grad).all())
            for parameter in parameters
        )
        if loss_is_finite and gradients_are_finite:
            return
        optimizer.zero_grad(set_to_none=True)
        raise FloatingPointError(
            f"Обнаружены NaN или Inf на этапе {stage}"
        )

    def train_advantage_network_multi(self, batch_size=None, epochs=None, player_id=0):
        del player_id
        batch_size = int(batch_size or self.advantage_batch_size)
        epochs = int(epochs or self.advantage_epochs)
        count = len(self.advantage_buffer)
        if count == 0:
            self.last_advantage_train_steps = 0
            self.last_advantage_effective_batch_size = 0
            return 0.0
        effective_batch = min(batch_size, count)
        samples = self.advantage_buffer.sample(count)
        if samples is None:
            return 0.0
        states, regrets, masks, iterations = samples
        self.advantage_net.train()
        total_loss, steps = 0.0, 0
        iteration_now = max(int(self.iteration_count), 1)
        configured_steps = getattr(self, "advantage_train_steps", None)
        preloaded = self._preload_training_arrays(
            (states, regrets, masks, iterations),
            "advantage",
        )
        self._synchronize_training_device()
        started = time.perf_counter()
        batch_iterator = (
            self._fixed_step_batches(count, batch_size, configured_steps)
            if configured_steps is not None
            else self._full_epoch_batches(count, effective_batch, epochs)
        )
        fixed_preload_steps = int(configured_steps or 0) if preloaded is not None and configured_steps is not None else 0
        preload_states, preload_regrets, preload_masks, preload_iterations = preloaded or (None, None, None, None)
        step_source = range(max(0, fixed_preload_steps)) if fixed_preload_steps else batch_iterator
        for item in step_source:
            if fixed_preload_steps:
                index_t = self._training_indices(count, batch_size)
                state_t = preload_states.index_select(0, index_t)
                regret_t = preload_regrets.index_select(0, index_t)
                mask_t = preload_masks.index_select(0, index_t)
                source_iteration_t = preload_iterations.index_select(0, index_t)
            else:
                _, indices = item
                if preloaded is None:
                    state_t = torch.from_numpy(states[indices].copy()).to(self.device)
                    regret_t = torch.from_numpy(regrets[indices].copy()).to(self.device)
                    mask_t = torch.from_numpy(masks[indices].copy()).to(self.device)
                    source_iteration_t = torch.from_numpy(iterations[indices].copy()).to(self.device)
                else:
                    index_t = torch.as_tensor(indices, dtype=torch.long, device=self.device)
                    state_t = preload_states.index_select(0, index_t)
                    regret_t = preload_regrets.index_select(0, index_t)
                    mask_t = preload_masks.index_select(0, index_t)
                    source_iteration_t = preload_iterations.index_select(0, index_t)
            with torch.no_grad():
                previous = self.advantage_target_net(state_t)
                previous_positive = torch.clamp(previous, min=0.0)
                is_fresh = (source_iteration_t == iteration_now).unsqueeze(1).float()
                if iteration_now <= 1:
                    targets = is_fresh * regret_t + (1.0 - is_fresh) * previous_positive
                elif self.advantage_accumulation == "plain":
                    targets = is_fresh * (previous + regret_t) + (1.0 - is_fresh) * previous
                else:
                    discount = (iteration_now - 1) ** self.discount_alpha
                    discount /= discount + 1.0
                    targets = is_fresh * (previous_positive * discount + regret_t) + (1.0 - is_fresh) * previous_positive
            predictions = self.advantage_net(state_t)
            if self.advantage_loss == "huber":
                loss = F.smooth_l1_loss(predictions * mask_t, targets * mask_t, beta=self.advantage_huber_delta)
            else:
                loss = F.mse_loss(predictions * mask_t, targets * mask_t)
            parameters = tuple(self.advantage_net.parameters())
            self.optimizer.zero_grad()
            self._assert_finite_training_tensors(
                loss, parameters, self.optimizer, "обучения сети преимуществ"
            )
            loss.backward()
            self._assert_finite_training_tensors(
                loss, parameters, self.optimizer, "обучения сети преимуществ"
            )
            torch.nn.utils.clip_grad_norm_(parameters, max_norm=1.0)
            self.optimizer.step()
            total_loss += float(loss.item())
            steps += 1
        self._synchronize_training_device()
        train_seconds = time.perf_counter() - started
        self.advantage_target_net.load_state_dict(self.advantage_net.state_dict())
        self.last_advantage_train_steps = steps
        self.last_advantage_effective_batch_size = effective_batch
        self.last_advantage_target_stats = {
            "target_abs_max": float(targets.abs().max().item()),
            "error_abs_mean": float((predictions.detach() - targets).abs().mean().item()),
        }
        self.last_advantage_profile = {
            "samples": count, "batch": batch_size if configured_steps is not None else effective_batch, "epochs": epochs,
            "configured_steps": configured_steps,
            "expected_steps": configured_steps if configured_steps is not None else epochs * math.ceil(count / effective_batch),
            "actual_steps": steps, "total_seconds": train_seconds,
            "preloaded_to_device": preloaded is not None,
        }
        return total_loss / max(steps, 1)

    def train_strategy_network(self, batch_size=None, epochs=None):
        batch_size = int(batch_size or self.strategy_batch_size)
        epochs = int(epochs or self.strategy_epochs)
        count = len(self.strategy_buffer)
        if count == 0:
            return 0.0
        effective_batch = min(batch_size, count)
        samples = self.strategy_buffer.sample(count)
        if samples is None:
            return 0.0
        states, policies, masks, iterations = samples
        self.strategy_net.train()
        if self.teacher_strategy_net is not None:
            self.teacher_strategy_net.eval()
        total_loss, steps = 0.0, 0
        total_supervised_loss, total_distillation_loss = 0.0, 0.0
        iteration_now = max(int(self.iteration_count), 1)
        distillation_lambda = self._effective_strategy_distillation_lambda(iteration_now)
        configured_steps = getattr(self, "strategy_train_steps", None)
        preloaded = self._preload_training_arrays(
            (states, policies, masks, iterations),
            "strategy",
        )
        self._synchronize_training_device()
        started = time.perf_counter()
        batch_iterator = (
            self._fixed_step_batches(count, batch_size, configured_steps)
            if configured_steps is not None
            else self._full_epoch_batches(count, effective_batch, epochs)
        )
        fixed_preload_steps = int(configured_steps or 0) if preloaded is not None and configured_steps is not None else 0
        preload_states, preload_policies, preload_masks, preload_iterations = preloaded or (None, None, None, None)
        step_source = range(max(0, fixed_preload_steps)) if fixed_preload_steps else batch_iterator
        for item in step_source:
            if fixed_preload_steps:
                index_t = self._training_indices(count, batch_size)
                state_t = preload_states.index_select(0, index_t)
                policy_t = preload_policies.index_select(0, index_t)
                mask_t = preload_masks.index_select(0, index_t)
                iteration_t = preload_iterations.index_select(0, index_t)
            else:
                _, indices = item
                if preloaded is None:
                    state_t = torch.from_numpy(states[indices].copy()).to(self.device)
                    policy_t = torch.from_numpy(policies[indices].copy()).to(self.device)
                    mask_t = torch.from_numpy(masks[indices].copy()).to(self.device)
                    iteration_t = torch.from_numpy(iterations[indices].copy()).to(self.device)
                else:
                    index_t = torch.as_tensor(indices, dtype=torch.long, device=self.device)
                    state_t = preload_states.index_select(0, index_t)
                    policy_t = preload_policies.index_select(0, index_t)
                    mask_t = preload_masks.index_select(0, index_t)
                    iteration_t = preload_iterations.index_select(0, index_t)
            logits = self.strategy_net(state_t)
            masked_logits = torch.where(mask_t > 0.0, logits, torch.full_like(logits, -1e20))
            predicted = F.softmax(masked_logits, dim=1)
            weights = torch.pow(torch.clamp(iteration_t / iteration_now, min=1e-6), self.discount_gamma)
            per_sample_loss = ((predicted - policy_t).square() * mask_t).sum(dim=1)
            supervised_loss = torch.sum(per_sample_loss * weights) / torch.clamp(weights.sum(), min=1e-8)
            distillation_loss = torch.zeros((), device=self.device)
            if self.teacher_strategy_net is not None and distillation_lambda > 0.0:
                temperature = self.strategy_distillation_temperature
                with torch.no_grad():
                    teacher_state_t = self._project_states_for_teacher_strategy(state_t)
                    teacher_logits = self.teacher_strategy_net(teacher_state_t)
                    teacher_logits = torch.where(
                        mask_t > 0.0,
                        teacher_logits / temperature,
                        torch.full_like(teacher_logits, -1e20),
                    )
                    teacher_probs = F.softmax(teacher_logits, dim=1)
                student_log_probs = F.log_softmax(masked_logits / temperature, dim=1)
                per_sample_distillation = F.kl_div(
                    student_log_probs,
                    teacher_probs,
                    reduction="none",
                    log_target=False,
                ).sum(dim=1)
                distillation_loss = (
                    torch.sum(per_sample_distillation * weights)
                    / torch.clamp(weights.sum(), min=1e-8)
                ) * (temperature ** 2)
            loss = supervised_loss + float(distillation_lambda) * distillation_loss
            parameters = tuple(self.strategy_net.parameters())
            self.strategy_optimizer.zero_grad()
            self._assert_finite_training_tensors(
                loss, parameters, self.strategy_optimizer, "обучения сети стратегии"
            )
            loss.backward()
            self._assert_finite_training_tensors(
                loss, parameters, self.strategy_optimizer, "обучения сети стратегии"
            )
            torch.nn.utils.clip_grad_norm_(parameters, max_norm=0.5)
            self.strategy_optimizer.step()
            total_loss += float(loss.item())
            total_supervised_loss += float(supervised_loss.item())
            total_distillation_loss += float(distillation_loss.item())
            steps += 1
        self._synchronize_training_device()
        train_seconds = time.perf_counter() - started
        self.last_strategy_profile = {
            "buffer_size": count, "batch_size": batch_size if configured_steps is not None else effective_batch, "epochs": epochs,
            "configured_steps": configured_steps,
            "expected_steps": configured_steps if configured_steps is not None else epochs * math.ceil(count / effective_batch),
            "actual_steps": steps, "train_seconds": train_seconds,
            "preloaded_to_device": preloaded is not None,
            "supervised_loss": total_supervised_loss / max(steps, 1),
            "distillation_loss": total_distillation_loss / max(steps, 1),
            "distillation_lambda": distillation_lambda,
        }
        return total_loss / max(steps, 1)

    def get_policy_distribution(self, state, player_id=None):
        player_id = int(state.current_player if player_id is None else player_id)
        mask = self.get_legal_action_mask(state)
        if not mask.any():
            raise ValueError("В текущем состоянии нет допустимых действий")
        encoded = self._encode_state(state, player_id)
        if getattr(self, "hu_current_policy_self_play", False):
            if player_id not in (0, 1):
                raise ValueError("HU strategy поддерживает только P0 и P1")
            encoded = np.concatenate(
                (encoded, np.eye(2, dtype=np.float32)[player_id])
            )
        state_t = torch.from_numpy(encoded).float().unsqueeze(0).to(self.device)
        mask_t = torch.from_numpy(mask).unsqueeze(0).to(self.device)
        with torch.inference_mode():
            logits = self.strategy_net(state_t)
            probabilities = self._masked_softmax(logits, mask_t)[0].cpu().numpy()
        return self._validate_traversal_policy(
            probabilities,
            mask,
            self.iteration_count,
            player_id,
            player_id,
            0,
            "strategy_net",
        ).astype(np.float32)

    def choose_action(self, state, player_id=None, deterministic=False):
        probabilities = self.get_policy_distribution(state, player_id)
        legal_slots = np.flatnonzero(probabilities > 0.0).astype(int)
        if not legal_slots.size:
            raise ValueError("Нельзя выбрать действие без допустимых слотов")
        if deterministic:
            slot = int(legal_slots[np.argmax(probabilities[legal_slots])])
        else:
            weights = probabilities[legal_slots]
            slot = int(np.random.choice(legal_slots, p=weights / weights.sum()))
        return self.action_type_to_pokers_action(slot, state)

    @staticmethod
    def _buffer_payload(buffer):
        count = min(buffer._cur_id, buffer.capacity)
        payload = {
            "cur_id": buffer._cur_id, "count": count,
            "states": buffer._states[:count].copy(),
            "masks": buffer._masks[:count].copy(),
            "iterations": buffer._iterations[:count].copy(),
        }
        if isinstance(buffer, AdvantageBuffer):
            payload["values"] = buffer._regrets[:count].copy()
        else:
            payload["values"] = buffer._policies[:count].copy()
        return payload

    @staticmethod
    def _restore_buffer(buffer, payload):
        count = int(payload.get("count", 0))
        values = np.asarray(payload.get("values"))
        states = np.asarray(payload.get("states"))
        masks = np.asarray(payload.get("masks"))
        iterations = np.asarray(payload.get("iterations"))
        target = buffer._regrets if isinstance(buffer, AdvantageBuffer) else buffer._policies
        if count <= 0 or count > buffer.capacity or states.shape != (count, *buffer._states.shape[1:]) or values.shape != (count, *target.shape[1:]) or masks.shape != (count, *buffer._masks.shape[1:]) or iterations.shape != (count,):
            return
        buffer._states[:count] = states
        target[:count] = values
        buffer._masks[:count] = masks
        buffer._iterations[:count] = iterations
        buffer._cur_id = max(count, int(payload.get("cur_id", count)))
        if hasattr(buffer, "_size"):
            buffer._size = count

    def _build_checkpoint(self, seed=None, extra=None):
        network_metadata = self._network_metadata()
        checkpoint = {
            "checkpoint_format_version": CHECKPOINT_FORMAT_VERSION,
            "action_space_version": ACTION_SPACE_VERSION,
            "action_labels": list(ACTION_LABELS),
            "iteration": int(self.iteration_count), "seed": seed,
            "num_players": self.num_players, "num_actions": NUM_ACTIONS,
            "use_multi_agent_advantage": self.use_multi_agent,
            "encoding_version": self.encoding_version,
            "encoder_input_size": self.input_size,
            **network_metadata,
            "advantage_net": self.advantage_net.state_dict(),
            "advantage_target_net": self.advantage_target_net.state_dict(),
            "strategy_net": self.strategy_net.state_dict(),
            "advantage_optimizer": self.optimizer.state_dict(),
            "strategy_optimizer": self.strategy_optimizer.state_dict(),
            "config": {
                "num_actions": NUM_ACTIONS, "action_space_version": ACTION_SPACE_VERSION,
                "hidden_size": self._network_hidden_size(self.advantage_net),
                "num_players": self.num_players, "use_multi_agent_advantage": self.use_multi_agent,
                "encoding_version": self.encoding_version,
                "encoder_input_size": self.input_size,
                "advantage_accumulation": self.advantage_accumulation,
                "advantage_train_steps": self.advantage_train_steps,
                "strategy_train_steps": self.strategy_train_steps,
                "discount_alpha": self.discount_alpha,
                "discount_gamma": self.discount_gamma,
                "num_trainable_players": self.num_trainable_players,
                "strategy_distillation_lambda": self.strategy_distillation_lambda,
                "strategy_distillation_temperature": self.strategy_distillation_temperature,
                "strategy_distillation_anneal_iterations": self.strategy_distillation_anneal_iterations,
                **network_metadata,
            },
        }
        if self.save_replay_buffers_in_checkpoint:
            checkpoint["advantage_buffer"] = self._buffer_payload(self.advantage_buffer)
            checkpoint["strategy_buffer"] = self._buffer_payload(self.strategy_buffer)
        if extra:
            checkpoint.update(extra)
        return checkpoint

    def build_light_checkpoint(self, seed=None):
        """Возвращает inference-артефакт только с усреднённой стратегией."""
        network_metadata = self._network_metadata()
        return {
            "checkpoint_format_version": CHECKPOINT_FORMAT_VERSION,
            "checkpoint_kind": "strategy_only",
            "action_space_version": ACTION_SPACE_VERSION,
            "action_labels": list(ACTION_LABELS),
            "iteration": int(self.iteration_count),
            "seed": seed,
            "num_players": self.num_players,
            "num_actions": NUM_ACTIONS,
            "encoding_version": self.encoding_version,
            "encoder_input_size": self.input_size,
            **network_metadata,
            "strategy_net": self.strategy_net.state_dict(),
            "config": {
                "num_actions": NUM_ACTIONS,
                "action_space_version": ACTION_SPACE_VERSION,
                "hidden_size": self._network_hidden_size(self.strategy_net),
                "num_players": self.num_players,
                "use_multi_agent_advantage": self.use_multi_agent,
                "encoding_version": self.encoding_version,
                "encoder_input_size": self.input_size,
                **network_metadata,
            },
        }

    def _load_checkpoint(self, path):
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        if checkpoint.get("checkpoint_format_version") != CHECKPOINT_FORMAT_VERSION:
            raise ValueError(
                "Чекпоинт несовместим: требуется новый запуск обучения с форматом "
                f"{CHECKPOINT_FORMAT_VERSION} ({ACTION_SPACE_VERSION}); старый encoder "
                "не переносится на history_summary_v3."
            )
        if checkpoint.get("action_space_version") != ACTION_SPACE_VERSION or int(checkpoint.get("num_actions", -1)) != NUM_ACTIONS:
            raise ValueError("Чекпоинт имеет другое пространство действий")
        if int(checkpoint.get("num_players", -1)) != self.num_players:
            raise ValueError("Чекпоинт имеет другое число игроков")
        checkpoint_use_multi_agent = bool(
            checkpoint.get(
                "use_multi_agent_advantage",
                checkpoint.get("config", {}).get("use_multi_agent_advantage", False),
            )
        )
        if checkpoint_use_multi_agent != self.use_multi_agent:
            raise ValueError("Чекпоинт имеет другой multi-agent режим")
        checkpoint_encoding_version = checkpoint.get(
            "encoding_version",
            checkpoint.get("config", {}).get("encoding_version"),
        )
        if checkpoint_encoding_version != self.encoding_version:
            raise ValueError("Чекпоинт имеет несовместимую версию encoder")
        if int(checkpoint.get("encoder_input_size", -1)) != self.input_size:
            raise ValueError("Чекпоинт имеет несовместимый размер входа encoder")
        for key in ("advantage_net", "advantage_target_net", "strategy_net"):
            if key not in checkpoint:
                raise ValueError(f"В checkpoint отсутствует обязательный ключ {key}")
        self._validate_checkpoint_network_metadata(checkpoint)
        checkpoint_architecture, checkpoint_input_size, checkpoint_hidden_size = (
            full_checkpoint_network_spec(checkpoint)
        )
        if checkpoint_architecture != self.network_architecture:
            raise ValueError("Чекпоинт имеет несовместимую архитектуру сети")
        if checkpoint_input_size != self.input_size:
            raise ValueError("Чекпоинт имеет несовместимый размер входа encoder")
        if checkpoint_hidden_size != self._network_hidden_size(self.strategy_net):
            raise ValueError("Чекпоинт имеет несовместимый hidden_size сети")
        self.advantage_net.load_state_dict(checkpoint["advantage_net"], strict=True)
        self.advantage_target_net.load_state_dict(checkpoint["advantage_target_net"], strict=True)
        self.strategy_net.load_state_dict(checkpoint["strategy_net"], strict=True)
        for key, optimizer in (("advantage_optimizer", self.optimizer), ("strategy_optimizer", self.strategy_optimizer)):
            if key in checkpoint:
                try:
                    optimizer.load_state_dict(checkpoint[key])
                except (ValueError, KeyError):
                    pass
        if "advantage_buffer" in checkpoint:
            self._restore_buffer(self.advantage_buffer, checkpoint["advantage_buffer"])
        if "strategy_buffer" in checkpoint:
            self._restore_buffer(self.strategy_buffer, checkpoint["strategy_buffer"])
        self.iteration_count = int(checkpoint.get("iteration", 0))
        return checkpoint

    def save_model(self, path_prefix, seed=None):
        path = _resolve_model_save_path(path_prefix, self.iteration_count)
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(self._build_checkpoint(seed=seed), path)
        return path

    def load_model(self, path):
        return self._load_checkpoint(path)

    def get_num_params(self):
        return sum(parameter.numel() for parameter in self.advantage_net.parameters())
