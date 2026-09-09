"""Безопасный перенос card_encoder из HU teacher в six-max student."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from io import BytesIO
from pathlib import Path
import pickle
from typing import Any

import torch

from src.core.action_space import ACTION_LABELS, ACTION_SPACE_VERSION, NUM_ACTIONS
from src.core.model import (
    CARD_CONTEXT_ARCHITECTURE,
    CARD_FEATURE_SIZE,
    HISTORY_SUMMARY_V3_ENCODING_VERSION,
    encoder_input_size,
)
from src.training.train import (
    CHECKPOINT_FORMAT_VERSION,
    _HU_CHECKPOINT_KIND,
    _HU_CHECKPOINT_VERSION,
    _HU_RUNTIME_CONFIG_ALLOWLIST,
    _HU_UPDATE_ORDER,
)


_HU_INPUT_SIZE = encoder_input_size(2, HISTORY_SUMMARY_V3_ENCODING_VERSION)
_HU_STRATEGY_INPUT_SIZE = _HU_INPUT_SIZE + 2
_CARD_ENCODER_PARAMETER_NAMES = ("0.weight", "0.bias")
_STRATEGY_PARAMETER_NAMES = {
    "card_encoder.0.weight",
    "card_encoder.0.bias",
    "context_encoder.0.weight",
    "context_encoder.0.bias",
    "action_head.weight",
    "action_head.bias",
}
_TRAJECTORY_CONFIG_KEYS = frozenset({
    "advantage_accumulation",
    "discount_alpha",
    "discount_gamma",
    "advantage_regret_norm",
    "advantage_regret_clip",
    "advantage_reward_scale",
    "advantage_loss",
    "advantage_huber_delta",
    "advantage_batch_size",
    "strategy_batch_size",
    "advantage_epochs",
    "strategy_epochs",
    "advantage_train_steps",
    "strategy_train_steps",
    "advantage_buffer_reservoir",
    "clear_strategy_buffer_each_iteration",
    "hu_strategy_buffer_reservoir",
    "strategy_distillation_lambda",
    "strategy_distillation_temperature",
    "strategy_distillation_anneal_iterations",
    "advantage_optimizers",
    "strategy_optimizer",
    "advantage_buffer_capacities",
    "strategy_buffer_capacity",
    "training_error_mode",
    "training_max_failed_traversals_per_iteration",
})


@dataclass(frozen=True, slots=True)
class CardEncoderWarmstartProvenance:
    """Неизменяемое происхождение перенесённого блока параметров."""

    mode: str
    copied_blocks: tuple[str, ...]
    source_path: Path
    checksum_sha256: str
    source_architecture: str
    source_encoding_version: str
    teacher_num_players: int
    freeze: bool


def transfer_card_encoder_from_hu_checkpoint(
    student: Any,
    source_path: str | Path,
    *,
    freeze: bool = False,
) -> CardEncoderWarmstartProvenance:
    """Копирует только strategy card_encoder из совместимого HU full checkpoint."""
    resolved_path = _resolve_source_path(source_path)
    checkpoint, checksum = _load_checkpoint_snapshot(resolved_path)
    source_card_state, source_metadata = _validated_source_card_encoder(checkpoint)
    target_parameters = _validated_student_card_encoder(student, source_card_state)
    staged_card_state = _stage_card_state(source_card_state, target_parameters)

    with torch.no_grad():
        for parameter_name, source_tensor in staged_card_state.items():
            target_parameters[parameter_name].copy_(source_tensor)
    if freeze:
        for parameter in target_parameters.values():
            parameter.requires_grad_(False)

    return CardEncoderWarmstartProvenance(
        mode="card_encoder_warmstart",
        copied_blocks=("strategy_net.card_encoder",),
        source_path=resolved_path,
        checksum_sha256=checksum,
        source_architecture=source_metadata["architecture"],
        source_encoding_version=source_metadata["encoding_version"],
        teacher_num_players=source_metadata["num_players"],
        freeze=bool(freeze),
    )


def _resolve_source_path(source_path: str | Path) -> Path:
    path = Path(source_path).expanduser().resolve()
    if not path.is_file():
        raise ValueError("HU checkpoint для переноса card_encoder не найден")
    return path


def _load_checkpoint_snapshot(path: Path) -> tuple[dict[str, Any], str]:
    """Читает checkpoint один раз, чтобы checksum и загрузка описывали один байтовый снимок."""
    try:
        snapshot = path.read_bytes()
        checkpoint = torch.load(BytesIO(snapshot), map_location="cpu", weights_only=True)
    except (OSError, pickle.UnpicklingError, RuntimeError, ValueError) as error:
        raise ValueError("HU checkpoint не удалось безопасно прочитать") from error
    if not isinstance(checkpoint, dict):
        raise ValueError("Для переноса требуется полный HU checkpoint в виде словаря")
    return checkpoint, hashlib.sha256(snapshot).hexdigest()


def _validated_source_card_encoder(
    checkpoint: dict[str, Any],
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    _validate_hu_checkpoint_header(checkpoint)
    mode = _validate_hu_mode(checkpoint)
    _validate_hu_config(checkpoint, mode)
    _validate_hu_full_training_state(checkpoint)
    strategy_architecture = _validate_hu_architecture(checkpoint)
    strategy_state = _validate_strategy_state(checkpoint, strategy_architecture)

    return (
        {
            parameter_name: strategy_state[f"card_encoder.{parameter_name}"].detach().clone(
                memory_format=torch.contiguous_format
            )
            for parameter_name in _CARD_ENCODER_PARAMETER_NAMES
        },
        {
            "architecture": CARD_CONTEXT_ARCHITECTURE,
            "encoding_version": mode["encoding_version"],
            "num_players": mode["num_players"],
        },
    )


def _validate_hu_checkpoint_header(checkpoint: dict[str, Any]) -> None:
    if checkpoint.get("checkpoint_kind") != _HU_CHECKPOINT_KIND:
        raise ValueError("Для переноса требуется полный HU checkpoint")
    if checkpoint.get("hu_checkpoint_version") != _HU_CHECKPOINT_VERSION:
        raise ValueError("HU checkpoint имеет несовместимую версию")
    if checkpoint.get("checkpoint_format_version") != CHECKPOINT_FORMAT_VERSION:
        raise ValueError("HU checkpoint имеет несовместимый общий формат")
    if (
        checkpoint.get("action_space_version") != ACTION_SPACE_VERSION
        or checkpoint.get("num_actions") != NUM_ACTIONS
    ):
        raise ValueError("HU checkpoint имеет другое пространство действий")
    if checkpoint.get("action_labels") != list(ACTION_LABELS):
        raise ValueError("HU checkpoint имеет несовместимый action_labels контракт")
    iteration = checkpoint.get("iteration")
    if isinstance(iteration, bool) or not isinstance(iteration, int) or iteration < 0:
        raise ValueError("HU checkpoint имеет некорректный iteration")
    if checkpoint.get("update_order") != _HU_UPDATE_ORDER:
        raise ValueError("HU checkpoint имеет неизвестный порядок обновления")


def _validate_hu_mode(checkpoint: dict[str, Any]) -> dict[str, Any]:
    expected_mode = {
        "hu_current_policy_self_play": True,
        "num_players": 2,
        "num_trainable_players": 2,
        "use_multi_agent_advantage": False,
        "encoding_version": HISTORY_SUMMARY_V3_ENCODING_VERSION,
        "encoder_input_size": _HU_INPUT_SIZE,
    }
    mode = checkpoint.get("mode")
    if mode != expected_mode:
        raise ValueError("HU checkpoint имеет несовместимый режим или encoder")
    return mode


def _validate_hu_config(checkpoint: dict[str, Any], mode: dict[str, Any]) -> None:
    config = checkpoint.get("config")
    if not isinstance(config, dict) or any(config.get(key) != value for key, value in mode.items()):
        raise ValueError("HU checkpoint имеет несовместимую конфигурацию")
    required_keys = set(mode) | _TRAJECTORY_CONFIG_KEYS
    unsupported_keys = set(config) - required_keys - _HU_RUNTIME_CONFIG_ALLOWLIST
    if not required_keys.issubset(config) or unsupported_keys:
        raise ValueError("HU checkpoint имеет неполную конфигурацию")


def _validate_hu_full_training_state(checkpoint: dict[str, Any]) -> None:
    advantage_legs = checkpoint.get("advantage_legs")
    strategy = checkpoint.get("strategy")
    if not isinstance(advantage_legs, list) or len(advantage_legs) != 2 or not isinstance(strategy, dict):
        raise ValueError("HU checkpoint не содержит полный набор training state")
    required_leg_keys = {"network", "target_network", "optimizer", "buffer"}
    if any(not isinstance(leg, dict) or set(leg) != required_leg_keys for leg in advantage_legs):
        raise ValueError("HU checkpoint не содержит полный набор training state")
    if set(strategy) != {"network", "optimizer", "buffer"}:
        raise ValueError("HU checkpoint не содержит полный набор training state")
    if any(
        not _is_optimizer_state(leg["optimizer"])
        or not _is_advantage_buffer_payload(leg["buffer"])
        for leg in advantage_legs
    ) or not _is_optimizer_state(strategy["optimizer"]) or not _is_strategy_buffer_payload(
        strategy["buffer"]
    ):
        raise ValueError("HU checkpoint содержит повреждённый replay-буфер или optimizer")
    rng = checkpoint.get("rng")
    if not isinstance(rng, dict) or not {"python", "numpy", "torch_cpu"}.issubset(rng):
        raise ValueError("HU checkpoint не содержит полное состояние RNG")


def _is_optimizer_state(payload: object) -> bool:
    return isinstance(payload, dict) and isinstance(payload.get("state"), dict) and isinstance(
        payload.get("param_groups"), list
    )


def _is_advantage_buffer_payload(payload: object) -> bool:
    return _is_replay_buffer_payload(
        payload,
        {"capacity", "state_dim", "cur_id", "count", "eviction_count", "skip_count", "states", "regrets", "masks", "iterations"},
    )


def _is_strategy_buffer_payload(payload: object) -> bool:
    return _is_replay_buffer_payload(
        payload,
        {"capacity", "state_dim", "cur_id", "count", "eviction_count", "skip_count", "states", "actor_ids", "policies", "masks", "iterations"},
    )


def _is_replay_buffer_payload(payload: object, expected_keys: set[str]) -> bool:
    if not isinstance(payload, dict) or set(payload) != expected_keys:
        return False
    for key in ("capacity", "state_dim", "cur_id", "count", "eviction_count", "skip_count"):
        if isinstance(payload[key], bool) or not isinstance(payload[key], int):
            return False
    count = payload["count"]
    if payload["capacity"] <= 0 or count < 0 or count > payload["capacity"] or payload["cur_id"] < count:
        return False
    return all(
        torch.is_tensor(payload[key]) and payload[key].layout == torch.strided
        for key in expected_keys - {"capacity", "state_dim", "cur_id", "count", "eviction_count", "skip_count"}
    )


def _validate_hu_architecture(checkpoint: dict[str, Any]) -> dict[str, Any]:
    architecture = checkpoint.get("architecture")
    if not isinstance(architecture, dict) or set(architecture) != {
        "advantage", "advantage_target", "strategy"
    }:
        raise ValueError("HU checkpoint не содержит полное описание архитектуры")
    strategy = architecture["strategy"]
    hidden_size = _positive_int(
        strategy.get("hidden_size") if isinstance(strategy, dict) else None,
        "hidden_size",
    )
    _validate_network_architecture(strategy, _HU_STRATEGY_INPUT_SIZE, hidden_size)
    for group_name in ("advantage", "advantage_target"):
        group = architecture[group_name]
        if not isinstance(group, list) or len(group) != 2:
            raise ValueError("HU checkpoint имеет несовместимую архитектуру advantage-сетей")
        for schema in group:
            _validate_network_architecture(schema, _HU_INPUT_SIZE, hidden_size)
    return strategy


def _validate_network_architecture(schema: object, input_size: int, hidden_size: int) -> None:
    expected_schema = {
        "network_architecture": CARD_CONTEXT_ARCHITECTURE,
        "card_feature_size": CARD_FEATURE_SIZE,
        "input_size": input_size,
        "hidden_size": hidden_size,
        "num_actions": NUM_ACTIONS,
    }
    if schema != expected_schema:
        raise ValueError("HU checkpoint имеет несовместимую архитектуру сети")


def _validate_strategy_state(
    checkpoint: dict[str, Any],
    strategy_architecture: dict[str, Any],
) -> dict[str, torch.Tensor]:
    strategy = checkpoint["strategy"]
    strategy_state = strategy["network"]
    if not isinstance(strategy_state, dict) or set(strategy_state) != _STRATEGY_PARAMETER_NAMES:
        raise ValueError("HU checkpoint содержит некорректные strategy-веса")
    hidden_size = strategy_architecture["hidden_size"]
    expected_shapes = {
        "card_encoder.0.weight": (hidden_size, CARD_FEATURE_SIZE),
        "card_encoder.0.bias": (hidden_size,),
        "context_encoder.0.weight": (hidden_size, _HU_STRATEGY_INPUT_SIZE - CARD_FEATURE_SIZE),
        "context_encoder.0.bias": (hidden_size,),
        "action_head.weight": (NUM_ACTIONS, hidden_size * 2),
        "action_head.bias": (NUM_ACTIONS,),
    }
    for name, shape in expected_shapes.items():
        value = strategy_state[name]
        if not torch.is_tensor(value) or tuple(value.shape) != shape:
            raise ValueError(f"HU checkpoint имеет несовместимый параметр strategy: {name}")
        if value.layout != torch.strided or value.dtype != torch.float32 or not value.is_contiguous():
            raise ValueError(f"HU checkpoint содержит недопустимый tensor strategy: {name}")
    return strategy_state


def _validated_student_card_encoder(
    student: Any,
    source_card_state: dict[str, torch.Tensor],
) -> dict[str, torch.nn.Parameter]:
    if int(getattr(student, "num_players", -1)) != 6:
        raise ValueError("Перенос card_encoder поддержан только для six-max student")
    if getattr(student, "encoding_version", None) != HISTORY_SUMMARY_V3_ENCODING_VERSION:
        raise ValueError("Six-max student должен использовать encoder history_summary_v3")
    strategy_net = getattr(student, "strategy_net", None)
    if getattr(strategy_net, "architecture", None) != CARD_CONTEXT_ARCHITECTURE:
        raise ValueError("Six-max student должен использовать архитектуру card_context_v1")
    card_encoder = getattr(strategy_net, "card_encoder", None)
    parameters = dict(card_encoder.named_parameters()) if card_encoder is not None else {}
    if set(parameters) != set(_CARD_ENCODER_PARAMETER_NAMES):
        raise ValueError("Six-max student имеет некорректный card_encoder")
    for parameter_name, source_tensor in source_card_state.items():
        target_parameter = parameters[parameter_name]
        if tuple(target_parameter.shape) != tuple(source_tensor.shape):
            raise ValueError("Card_encoder teacher и student имеют несовместимые формы")
        if (
            target_parameter.layout != torch.strided
            or target_parameter.dtype != torch.float32
            or not target_parameter.is_contiguous()
        ):
            raise ValueError("Six-max student имеет недопустимый card_encoder tensor")
    return parameters


def _stage_card_state(
    source_card_state: dict[str, torch.Tensor],
    target_parameters: dict[str, torch.nn.Parameter],
) -> dict[str, torch.Tensor]:
    """Готовит все device-local копии до первой мутации student."""
    return {
        parameter_name: source_tensor.to(
            device=target_parameters[parameter_name].device,
            dtype=target_parameters[parameter_name].dtype,
            copy=True,
        )
        for parameter_name, source_tensor in source_card_state.items()
    }


def _positive_int(value: object, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"HU checkpoint имеет некорректный {field_name} strategy-сети")
    return int(value)
