"""Безопасный перенос card_encoder из HU teacher в six-max student."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from io import BytesIO
import math
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
_NETWORK_PARAMETER_NAMES = (
    "card_encoder.0.weight",
    "card_encoder.0.bias",
    "context_encoder.0.weight",
    "context_encoder.0.bias",
    "action_head.weight",
    "action_head.bias",
)
_STRATEGY_PARAMETER_NAMES = frozenset(_NETWORK_PARAMETER_NAMES)
_ADAMW_GROUP_KEYS = frozenset({
    "lr",
    "betas",
    "eps",
    "weight_decay",
    "amsgrad",
    "maximize",
    "foreach",
    "capturable",
    "differentiable",
    "fused",
    "decoupled_weight_decay",
    "params",
})
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
    strategy_architecture = _validate_hu_architecture(checkpoint)
    strategy_state = _validate_hu_full_training_state(checkpoint, strategy_architecture)

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


def _validate_hu_full_training_state(
    checkpoint: dict[str, Any],
    strategy_architecture: dict[str, Any],
) -> dict[str, torch.Tensor]:
    advantage_legs = checkpoint.get("advantage_legs")
    strategy = checkpoint.get("strategy")
    if not isinstance(advantage_legs, list) or len(advantage_legs) != 2 or not isinstance(strategy, dict):
        raise ValueError("HU checkpoint не содержит полный набор training state")
    required_leg_keys = {"network", "target_network", "optimizer", "buffer"}
    if any(not isinstance(leg, dict) or set(leg) != required_leg_keys for leg in advantage_legs):
        raise ValueError("HU checkpoint не содержит полный набор training state")
    if set(strategy) != {"network", "optimizer", "buffer"}:
        raise ValueError("HU checkpoint не содержит полный набор training state")
    hidden_size = strategy_architecture["hidden_size"]
    config = checkpoint["config"]
    capacities = config["advantage_buffer_capacities"]
    if not isinstance(capacities, list) or len(capacities) != 2:
        raise ValueError("HU checkpoint имеет некорректную конфигурацию replay-буферов")
    for index, leg in enumerate(advantage_legs):
        advantage_state = _validate_network_state(
            leg["network"], _HU_INPUT_SIZE, hidden_size, "advantage"
        )
        _validate_network_state(leg["target_network"], _HU_INPUT_SIZE, hidden_size, "advantage")
        _validate_optimizer_state(leg["optimizer"], advantage_state)
        _validate_advantage_buffer(leg["buffer"], capacities[index])
    strategy_state = _validate_network_state(
        strategy["network"], _HU_STRATEGY_INPUT_SIZE, hidden_size, "strategy"
    )
    _validate_optimizer_state(strategy["optimizer"], strategy_state)
    _validate_strategy_buffer(strategy["buffer"], config["strategy_buffer_capacity"])
    _validate_rng_state(checkpoint.get("rng"))
    return strategy_state


def _validate_network_state(
    state: object,
    input_size: int,
    hidden_size: int,
    network_kind: str,
) -> dict[str, torch.Tensor]:
    if not isinstance(state, dict) or set(state) != _STRATEGY_PARAMETER_NAMES:
        raise ValueError(f"HU checkpoint содержит повреждённые {network_kind}-сети")
    expected_shapes = {
        "card_encoder.0.weight": (hidden_size, CARD_FEATURE_SIZE),
        "card_encoder.0.bias": (hidden_size,),
        "context_encoder.0.weight": (hidden_size, input_size - CARD_FEATURE_SIZE),
        "context_encoder.0.bias": (hidden_size,),
        "action_head.weight": (NUM_ACTIONS, hidden_size * 2),
        "action_head.bias": (NUM_ACTIONS,),
    }
    for name, shape in expected_shapes.items():
        value = state[name]
        if not torch.is_tensor(value) or tuple(value.shape) != shape:
            raise ValueError(f"HU checkpoint содержит повреждённые {network_kind}-сети")
        if (
            value.device.type != "cpu"
            or value.layout != torch.strided
            or value.dtype != torch.float32
            or not value.is_contiguous()
        ):
            raise ValueError(f"HU checkpoint содержит недопустимый tensor {network_kind}-сети")
    return state


def _validate_optimizer_state(
    payload: object,
    network_state: dict[str, torch.Tensor],
) -> None:
    """Проверяет ровно сериализацию AdamW, созданного для параметров PokerNetwork."""
    if not isinstance(payload, dict) or set(payload) != {"state", "param_groups"}:
        raise ValueError("HU checkpoint содержит повреждённый optimizer")
    state = payload["state"]
    groups = payload["param_groups"]
    expected_parameter_ids = list(range(len(_NETWORK_PARAMETER_NAMES)))
    if (
        not isinstance(state, dict)
        or not isinstance(groups, list)
        or len(groups) != 1
        or not _is_expected_adamw_group(groups[0], expected_parameter_ids)
    ):
        raise ValueError("HU checkpoint содержит повреждённый optimizer")
    parameter_shapes = tuple(
        tuple(network_state[name].shape) for name in _NETWORK_PARAMETER_NAMES
    )
    for parameter_id, parameter_state in state.items():
        if (
            isinstance(parameter_id, bool)
            or not isinstance(parameter_id, int)
            or parameter_id not in expected_parameter_ids
            or not _is_expected_adamw_parameter_state(
                parameter_state, parameter_shapes[parameter_id]
            )
        ):
            raise ValueError("HU checkpoint содержит повреждённый optimizer")


def _is_expected_adamw_group(group: object, expected_parameter_ids: list[int]) -> bool:
    """Отсекает группы, которые невозможно восстановить штатным AdamW HU runtime."""
    if not isinstance(group, dict) or set(group) != _ADAMW_GROUP_KEYS:
        return False
    if group["params"] != expected_parameter_ids:
        return False
    if (
        not _positive_float(group["lr"])
        or not _positive_float(group["eps"])
        or not _nonnegative_float(group["weight_decay"])
        or not isinstance(group["betas"], tuple)
        or len(group["betas"]) != 2
        or any(not _nonnegative_float(beta) or beta >= 1.0 for beta in group["betas"])
    ):
        return False
    if any(
        not isinstance(group[name], bool)
        for name in ("amsgrad", "maximize", "capturable", "differentiable", "decoupled_weight_decay")
    ):
        return False
    return all(group[name] is None or isinstance(group[name], bool) for name in ("foreach", "fused"))


def _is_expected_adamw_parameter_state(
    parameter_state: object,
    parameter_shape: tuple[int, ...],
) -> bool:
    """Проверяет формы и dtype AdamW moments относительно параметра сети."""
    if not isinstance(parameter_state, dict) or set(parameter_state) != {
        "step", "exp_avg", "exp_avg_sq"
    }:
        return False
    return (
        _is_cpu_float32_tensor(parameter_state["step"], ())
        and _is_cpu_float32_tensor(parameter_state["exp_avg"], parameter_shape)
        and _is_cpu_float32_tensor(parameter_state["exp_avg_sq"], parameter_shape)
    )


def _is_cpu_float32_tensor(value: object, shape: tuple[int, ...]) -> bool:
    return (
        torch.is_tensor(value)
        and tuple(value.shape) == shape
        and value.device.type == "cpu"
        and value.layout == torch.strided
        and value.dtype == torch.float32
        and value.is_contiguous()
    )


def _positive_float(value: object) -> bool:
    return isinstance(value, float) and math.isfinite(value) and value > 0.0


def _nonnegative_float(value: object) -> bool:
    return isinstance(value, float) and math.isfinite(value) and value >= 0.0


def _validate_advantage_buffer(payload: object, capacity: object) -> None:
    _validate_replay_buffer(
        payload,
        capacity,
        _HU_INPUT_SIZE,
        {
            "states": (_HU_INPUT_SIZE, torch.float32),
            "regrets": (NUM_ACTIONS, torch.float32),
            "masks": (NUM_ACTIONS, torch.float32),
            "iterations": (None, torch.float32),
        },
    )


def _validate_strategy_buffer(payload: object, capacity: object) -> None:
    _validate_replay_buffer(
        payload,
        capacity,
        _HU_INPUT_SIZE,
        {
            "states": (_HU_INPUT_SIZE, torch.float32),
            "actor_ids": (None, torch.int64),
            "policies": (NUM_ACTIONS, torch.float32),
            "masks": (NUM_ACTIONS, torch.float32),
            "iterations": (None, torch.float32),
        },
    )
    actor_ids = payload["actor_ids"]
    if not bool(torch.all((actor_ids == 0) | (actor_ids == 1))):
        raise ValueError("HU checkpoint содержит недопустимый actor_id replay-буфера")


def _validate_replay_buffer(
    payload: object,
    configured_capacity: object,
    state_dim: int,
    arrays: dict[str, tuple[int | None, torch.dtype]],
) -> None:
    required_keys = {"capacity", "state_dim", "cur_id", "count", "eviction_count", "skip_count", *arrays}
    if not isinstance(payload, dict) or set(payload) != required_keys:
        raise ValueError("HU checkpoint содержит повреждённый replay-буфер")
    if isinstance(configured_capacity, bool) or not isinstance(configured_capacity, int):
        raise ValueError("HU checkpoint имеет некорректную конфигурацию replay-буферов")
    for key in ("capacity", "state_dim", "cur_id", "count", "eviction_count", "skip_count"):
        if isinstance(payload[key], bool) or not isinstance(payload[key], int):
            raise ValueError("HU checkpoint содержит повреждённый replay-буфер")
    count = payload["count"]
    if (
        payload["capacity"] != configured_capacity
        or payload["capacity"] <= 0
        or payload["state_dim"] != state_dim
        or count < 0
        or count > payload["capacity"]
        or payload["cur_id"] < count
        or payload["eviction_count"] < 0
        or payload["skip_count"] < 0
    ):
        raise ValueError("HU checkpoint содержит повреждённый replay-буфер")
    for key, (width, dtype) in arrays.items():
        value = payload[key]
        expected_shape = (count,) if width is None else (count, width)
        if (
            not torch.is_tensor(value)
            or tuple(value.shape) != expected_shape
            or value.device.type != "cpu"
            or value.layout != torch.strided
            or value.dtype != dtype
            or not value.is_contiguous()
        ):
            raise ValueError("HU checkpoint содержит повреждённый replay-буфер")


def _validate_rng_state(payload: object) -> None:
    required_rng_keys = {"python", "numpy", "torch_cpu"}
    if not isinstance(payload, dict) or not required_rng_keys.issubset(payload) or set(payload) - required_rng_keys - {"torch_cuda"}:
        raise ValueError("HU checkpoint не содержит полное состояние RNG")
    python_state = payload["python"]
    if (
        not isinstance(python_state, tuple)
        or len(python_state) != 3
        or python_state[0] != 3
        or not isinstance(python_state[1], tuple)
        or len(python_state[1]) != 625
        or not _is_python_mt19937_state(python_state[1])
        or (
            python_state[2] is not None
            and (not isinstance(python_state[2], float) or not math.isfinite(python_state[2]))
        )
    ):
        raise ValueError("HU checkpoint содержит повреждённое RNG состояние")
    numpy_state = payload["numpy"]
    required_numpy_keys = {"algorithm", "state", "position", "has_gauss", "cached_gaussian"}
    if not isinstance(numpy_state, dict) or set(numpy_state) != required_numpy_keys:
        raise ValueError("HU checkpoint содержит повреждённое RNG состояние")
    state = numpy_state["state"]
    torch_cpu = payload["torch_cpu"]
    if (
        numpy_state["algorithm"] != "MT19937"
        or isinstance(numpy_state["position"], bool)
        or not isinstance(numpy_state["position"], int)
        or not 0 <= numpy_state["position"] <= 624
        or isinstance(numpy_state["has_gauss"], bool)
        or numpy_state["has_gauss"] not in (0, 1)
        or not isinstance(numpy_state["cached_gaussian"], float)
        or not math.isfinite(numpy_state["cached_gaussian"])
        or not _is_cpu_uint32_tensor(state, (624,))
        or not _is_cpu_uint8_rng_tensor(torch_cpu)
    ):
        raise ValueError("HU checkpoint содержит повреждённое RNG состояние")
    cuda_state = payload.get("torch_cuda")
    if "torch_cuda" in payload and (
        not isinstance(cuda_state, list)
        or not cuda_state
        or any(not _is_cpu_uint8_rng_tensor(value) for value in cuda_state)
    ):
        raise ValueError("HU checkpoint содержит повреждённое RNG состояние")


def _is_python_mt19937_state(state: tuple[object, ...]) -> bool:
    """Проверяет формат random.getstate() CPython, который создаёт HU checkpoint."""
    return (
        all(
            isinstance(value, int)
            and not isinstance(value, bool)
            and 0 <= value < 2**32
            for value in state[:-1]
        )
        and isinstance(state[-1], int)
        and not isinstance(state[-1], bool)
        and 0 <= state[-1] <= 624
    )


def _is_cpu_uint32_tensor(value: object, shape: tuple[int, ...]) -> bool:
    return (
        torch.is_tensor(value)
        and tuple(value.shape) == shape
        and value.device.type == "cpu"
        and value.layout == torch.strided
        and value.dtype == torch.uint32
        and value.is_contiguous()
    )


def _is_cpu_uint8_rng_tensor(value: object) -> bool:
    return (
        torch.is_tensor(value)
        and value.ndim == 1
        and value.numel() > 0
        and value.device.type == "cpu"
        and value.layout == torch.strided
        and value.dtype == torch.uint8
        and value.is_contiguous()
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
