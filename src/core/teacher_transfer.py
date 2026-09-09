"""Безопасный перенос card_encoder из HU teacher в six-max student."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
import pickle
from typing import Any

import torch

from src.core.action_space import ACTION_SPACE_VERSION, NUM_ACTIONS
from src.core.model import (
    CARD_CONTEXT_ARCHITECTURE,
    CARD_FEATURE_SIZE,
    HISTORY_SUMMARY_V3_ENCODING_VERSION,
)


_HU_CHECKPOINT_KIND = "hu_current_policy_self_play"
_HU_CHECKPOINT_VERSION = 2
_CHECKPOINT_FORMAT_VERSION = 6
_CARD_ENCODER_PARAMETER_NAMES = ("0.weight", "0.bias")
_STRATEGY_PARAMETER_NAMES = {
    "card_encoder.0.weight",
    "card_encoder.0.bias",
    "context_encoder.0.weight",
    "context_encoder.0.bias",
    "action_head.weight",
    "action_head.bias",
}


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
    checksum = _sha256_file(resolved_path)
    checkpoint = _load_weights_only_checkpoint(resolved_path)
    source_card_state, source_metadata = _validated_source_card_encoder(checkpoint)
    target_card_encoder = _validated_student_card_encoder(student, source_card_state)

    with torch.no_grad():
        for parameter_name, source_tensor in source_card_state.items():
            target_parameter = target_card_encoder.state_dict()[parameter_name]
            target_parameter.copy_(source_tensor.to(target_parameter.device, target_parameter.dtype))
    if freeze:
        for parameter in target_card_encoder.parameters():
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


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source_file:
        for chunk in iter(lambda: source_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_weights_only_checkpoint(path: Path) -> dict[str, Any]:
    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    except (OSError, pickle.UnpicklingError, RuntimeError, ValueError) as error:
        raise ValueError("HU checkpoint не удалось безопасно прочитать") from error
    if not isinstance(checkpoint, dict):
        raise ValueError("Для переноса требуется полный HU checkpoint в виде словаря")
    return checkpoint


def _validated_source_card_encoder(
    checkpoint: dict[str, Any],
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    if checkpoint.get("checkpoint_kind") != _HU_CHECKPOINT_KIND:
        raise ValueError("Для переноса требуется полный HU checkpoint")
    if checkpoint.get("hu_checkpoint_version") != _HU_CHECKPOINT_VERSION:
        raise ValueError("HU checkpoint имеет несовместимую версию")
    if checkpoint.get("checkpoint_format_version") != _CHECKPOINT_FORMAT_VERSION:
        raise ValueError("HU checkpoint имеет несовместимый общий формат")
    if (
        checkpoint.get("action_space_version") != ACTION_SPACE_VERSION
        or checkpoint.get("num_actions") != NUM_ACTIONS
    ):
        raise ValueError("HU checkpoint имеет другое пространство действий")

    mode = checkpoint.get("mode")
    if not isinstance(mode, dict):
        raise ValueError("HU checkpoint не содержит режим обучения")
    if (
        mode.get("hu_current_policy_self_play") is not True
        or mode.get("num_players") != 2
        or mode.get("num_trainable_players") != 2
        or mode.get("use_multi_agent_advantage") is not False
        or mode.get("encoding_version") != HISTORY_SUMMARY_V3_ENCODING_VERSION
    ):
        raise ValueError("HU checkpoint имеет несовместимый режим или encoder")

    advantage_legs = checkpoint.get("advantage_legs")
    strategy = checkpoint.get("strategy")
    if (
        not isinstance(advantage_legs, list)
        or len(advantage_legs) != 2
        or not isinstance(strategy, dict)
        or not isinstance(checkpoint.get("rng"), dict)
    ):
        raise ValueError("HU checkpoint не содержит полный набор training state")
    required_leg_keys = {"network", "target_network", "optimizer", "buffer"}
    if any(not isinstance(leg, dict) or not required_leg_keys.issubset(leg) for leg in advantage_legs):
        raise ValueError("HU checkpoint не содержит полный набор training state")
    if not {"network", "optimizer", "buffer"}.issubset(strategy):
        raise ValueError("HU checkpoint не содержит полный набор training state")

    architecture = checkpoint.get("architecture")
    strategy_architecture = architecture.get("strategy") if isinstance(architecture, dict) else None
    if not isinstance(strategy_architecture, dict):
        raise ValueError("HU checkpoint не содержит описание strategy-архитектуры")
    if strategy_architecture.get("network_architecture") != CARD_CONTEXT_ARCHITECTURE:
        raise ValueError("HU checkpoint требует архитектуру card_context_v1")
    if strategy_architecture.get("card_feature_size") != CARD_FEATURE_SIZE:
        raise ValueError("HU checkpoint имеет несовместимый размер card-признаков")
    hidden_size = _positive_int(strategy_architecture.get("hidden_size"), "hidden_size")
    input_size = _positive_int(strategy_architecture.get("input_size"), "input_size")
    if input_size <= CARD_FEATURE_SIZE or strategy_architecture.get("num_actions") != NUM_ACTIONS:
        raise ValueError("HU checkpoint имеет несовместимое описание strategy-сети")

    strategy_state = strategy.get("network")
    if not isinstance(strategy_state, dict) or set(strategy_state) != _STRATEGY_PARAMETER_NAMES:
        raise ValueError("HU checkpoint содержит некорректные strategy-веса")
    expected_shapes = {
        "card_encoder.0.weight": (hidden_size, CARD_FEATURE_SIZE),
        "card_encoder.0.bias": (hidden_size,),
        "context_encoder.0.weight": (hidden_size, input_size - CARD_FEATURE_SIZE),
        "context_encoder.0.bias": (hidden_size,),
        "action_head.weight": (NUM_ACTIONS, hidden_size * 2),
        "action_head.bias": (NUM_ACTIONS,),
    }
    for name, shape in expected_shapes.items():
        value = strategy_state[name]
        if not torch.is_tensor(value) or tuple(value.shape) != shape:
            raise ValueError(f"HU checkpoint имеет несовместимый параметр strategy: {name}")

    return (
        {
            parameter_name: strategy_state[f"card_encoder.{parameter_name}"].detach().clone()
            for parameter_name in _CARD_ENCODER_PARAMETER_NAMES
        },
        {
            "architecture": CARD_CONTEXT_ARCHITECTURE,
            "encoding_version": HISTORY_SUMMARY_V3_ENCODING_VERSION,
            "num_players": 2,
        },
    )


def _validated_student_card_encoder(
    student: Any,
    source_card_state: dict[str, torch.Tensor],
):
    if int(getattr(student, "num_players", -1)) != 6:
        raise ValueError("Перенос card_encoder поддержан только для six-max student")
    if getattr(student, "encoding_version", None) != HISTORY_SUMMARY_V3_ENCODING_VERSION:
        raise ValueError("Six-max student должен использовать encoder history_summary_v3")
    strategy_net = getattr(student, "strategy_net", None)
    if getattr(strategy_net, "architecture", None) != CARD_CONTEXT_ARCHITECTURE:
        raise ValueError("Six-max student должен использовать архитектуру card_context_v1")
    card_encoder = getattr(strategy_net, "card_encoder", None)
    if card_encoder is None:
        raise ValueError("Six-max student не содержит card_encoder")
    target_state = card_encoder.state_dict()
    if set(target_state) != set(_CARD_ENCODER_PARAMETER_NAMES):
        raise ValueError("Six-max student имеет некорректный card_encoder")
    for parameter_name, source_tensor in source_card_state.items():
        target_tensor = target_state[parameter_name]
        if tuple(target_tensor.shape) != tuple(source_tensor.shape):
            raise ValueError("Card_encoder teacher и student имеют несовместимые формы")
    return card_encoder


def _positive_int(value: object, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"HU checkpoint имеет некорректный {field_name} strategy-сети")
    return int(value)
