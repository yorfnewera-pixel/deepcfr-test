"""Циклы обучения Deep CFR для фиксированных шести действий."""
from __future__ import annotations

import os
import pickle
import random
import re
import tempfile
import time
import argparse
from copy import deepcopy
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import numpy as np
import pokers as pkrs
import torch

from src.agents.random_agent import RandomAgent
from src.core.action_space import ACTION_LABELS, ACTION_SPACE_VERSION, NUM_ACTIONS
from src.core.buffers import AdvantageBuffer
from src.core.deep_cfr import CHECKPOINT_FORMAT_VERSION, DeepCFRAgent
from src.core.hu_self_play import (
    HuCurrentPolicySelfPlayCoordinator,
    HuStrategyBuffer,
    HuTraversalAdapter,
)
from src.core.model import (
    CARD_CONTEXT_ARCHITECTURE,
    CARD_FEATURE_SIZE,
    MONOLITHIC_ARCHITECTURE,
    PokerNetwork,
)
from src.core.traversal_errors import TraversalFailure
from src.utils.config import (
    cfg_get,
    cfg_training_error_mode,
    cfg_training_max_failed_traversals_per_iteration,
)


_HEAVY_CHECKPOINT_PREFIX = "multi_checkpoint_iter_"
_LIGHT_CHECKPOINT_PREFIX = "light_checkpoint_iter_"
_OPPONENT_RECENT_CHECKPOINTS = 11
_OPPONENT_HISTORICAL_CHECKPOINTS = 2
_HU_CHECKPOINT_KIND = "hu_current_policy_self_play"
_HU_CHECKPOINT_VERSION = 2
_HU_UPDATE_ORDER = [
    "traverse_p0",
    "traverse_p1",
    "train_advantage_p0",
    "train_advantage_p1",
    "train_strategy",
]
_HU_RUNTIME_CONFIG_ALLOWLIST = frozenset({
    "save_dir",
    "log_dir",
    "evaluate_every",
    "evaluation_games",
    "checkpoint_save_every",
    "checkpoint_keep_every",
    "process_priority",
    "training_torch_threads",
    "training_preload_to_device",
    "traversal_single_thread",
})


def _teacher_transfer_provenance_payload(provenance: object) -> dict[str, object]:
    """Сериализует provenance Stage A без ссылок на изменяемые объекты teacher-а."""
    return {
        "mode": str(provenance.mode),
        "copied_blocks": list(provenance.copied_blocks),
        "source_path": str(provenance.source_path),
        "checksum_sha256": str(provenance.checksum_sha256),
        "source_architecture": str(provenance.source_architecture),
        "source_encoding_version": str(provenance.source_encoding_version),
        "teacher_num_players": int(provenance.teacher_num_players),
        "freeze": bool(provenance.freeze),
    }


def _teacher_transfer_configuration(
    *,
    enabled: bool | None = None,
    mode: str | None = None,
    checkpoint: str | Path | None = None,
    freeze: bool | None = None,
    auxiliary_enabled: bool | None = None,
    auxiliary_weight: float | None = None,
) -> dict[str, object]:
    """Возвращает единственный поддержанный Stage A контракт warm-start."""
    return {
        "enabled": bool(cfg_get("teacher_transfer_enabled", False))
        if enabled is None
        else bool(enabled),
        "mode": str(cfg_get("teacher_transfer_mode", "card_encoder_warmstart"))
        if mode is None
        else str(mode),
        "checkpoint": cfg_get("teacher_transfer_checkpoint", None)
        if checkpoint is None
        else checkpoint,
        "freeze": bool(cfg_get("teacher_transfer_freeze_card_encoder", False))
        if freeze is None
        else bool(freeze),
        "auxiliary_enabled": bool(cfg_get("teacher_hu_aux_distillation_enabled", False))
        if auxiliary_enabled is None
        else bool(auxiliary_enabled),
        "auxiliary_weight": cfg_get("teacher_hu_aux_distillation_weight", 0.0)
        if auxiliary_weight is None
        else auxiliary_weight,
    }


def _validate_teacher_transfer_runtime(
    configuration: dict[str, object],
    *,
    num_players: int,
    hu_current_policy_self_play: bool,
    teacher_strategy_checkpoint: str | Path | None,
) -> None:
    try:
        auxiliary_weight = float(configuration["auxiliary_weight"])
    except (TypeError, ValueError) as error:
        raise ValueError("teacher_hu_aux_distillation_weight должен быть числом") from error
    if bool(configuration["auxiliary_enabled"]) or auxiliary_weight != 0.0:
        raise ValueError("Stage B auxiliary transfer пока не реализован")
    if not configuration["enabled"]:
        return
    if configuration["mode"] != "card_encoder_warmstart":
        raise ValueError("teacher_transfer_mode поддерживает только card_encoder_warmstart")
    checkpoint = configuration["checkpoint"]
    if not isinstance(checkpoint, (str, Path)) or not str(checkpoint).strip():
        raise ValueError("teacher_transfer_checkpoint обязателен при teacher_transfer_enabled")
    if hu_current_policy_self_play:
        raise ValueError("teacher_transfer_enabled несовместим с hu_current_policy_self_play")
    if int(num_players) != 6:
        raise ValueError("teacher_transfer_enabled поддержан только для six-max")
    if teacher_strategy_checkpoint:
        raise ValueError("teacher_transfer_enabled несовместим с teacher_strategy_checkpoint")


def _checkpoint_iteration(path: Path, prefix: str = _HEAVY_CHECKPOINT_PREFIX) -> int | None:
    match = re.fullmatch(rf"{re.escape(prefix)}(\d+)\.pt", path.name)
    return int(match.group(1)) if match else None


def _current_strategy_opponent_count(iteration: int, checkpoint_every: int) -> int:
    """Определяет число временных слотов общей strategy до заполнения пула checkpoint."""
    completed_slots = max(0, (int(iteration) - 1) // max(1, int(checkpoint_every)))
    return max(0, 5 - min(completed_slots, 5))


def _transition_checkpoint_iterations(iteration: int, checkpoint_every: int) -> list[int]:
    """Возвращает последние собственные checkpoint в период плавного перехода."""
    checkpoint_every = max(1, int(checkpoint_every))
    completed_slots = (int(iteration) - 1) // checkpoint_every
    own_count = min(completed_slots, 4)
    if completed_slots >= 5:
        return []
    return [slot * checkpoint_every for slot in range(completed_slots, 0, -1)][:own_count]


def _assign_checkpoint_opponents(
    traversing_player: int,
    num_players: int,
    strategy_count: int,
    checkpoint_paths: list[Path],
) -> tuple[tuple[int, ...], dict[int, Path]]:
    """Перемешивает общую strategy и исторические checkpoint по местам оппонентов."""
    opponent_ids = [player_id for player_id in range(num_players) if player_id != int(traversing_player)]
    if strategy_count < 0 or strategy_count + len(checkpoint_paths) != len(opponent_ids):
        raise ValueError("Состав смешанного opponent pool не заполняет все позиции")
    entries: list[Path | None] = [None] * strategy_count + list(checkpoint_paths)
    random.shuffle(entries)
    strategy_positions = tuple(
        player_id for player_id, entry in zip(opponent_ids, entries, strict=True) if entry is None
    )
    checkpoint_by_player = {
        player_id: entry
        for player_id, entry in zip(opponent_ids, entries, strict=True)
        if entry is not None
    }
    return strategy_positions, checkpoint_by_player


def _heavy_checkpoints(directory: str | Path) -> dict[int, Path]:
    """Возвращает полные checkpoint для продолжения обучения."""
    return _checkpoint_paths(directory, _HEAVY_CHECKPOINT_PREFIX)


def _latest_strategy_checkpoint_path(iteration: int, checkpoint_dir: str | Path) -> Path:
    """Возвращает последний full checkpoint, предшествующий текущей итерации."""
    checkpoints = {
        number: path
        for number, path in _heavy_checkpoints(checkpoint_dir).items()
        if number < int(iteration)
    }
    if not checkpoints:
        raise ValueError(
            "Для игры против общей strategy нужен предыдущий full checkpoint"
        )
    return checkpoints[max(checkpoints)]


def _checkpoint_paths(directory: str | Path, prefix: str) -> dict[int, Path]:
    """Сканирует checkpoint заданного формата без смешивания артефактов."""
    result: dict[int, Path] = {}
    for path in Path(directory).glob(f"{prefix}*.pt"):
        iteration = _checkpoint_iteration(path, prefix)
        if iteration is not None:
            result[iteration] = path
    return result


def _checkpoint_save_due(iteration: int, every: int) -> bool:
    """Возвращает, следует ли сохранить checkpoint после итерации."""
    return int(iteration) > 0 and int(iteration) % max(1, int(every)) == 0


def _prune_full_checkpoints(directory: str | Path, historical_every: int) -> None:
    """Оставляет 11 свежих и два последних исторических full checkpoint."""
    checkpoints = _heavy_checkpoints(directory)
    historical_every = max(1, int(historical_every))
    historical = sorted(
        iteration for iteration in checkpoints if iteration % historical_every == 0
    )[-_OPPONENT_HISTORICAL_CHECKPOINTS:]
    historical_set = set(historical)
    ordinary = sorted(
        (iteration for iteration in checkpoints if iteration not in historical_set),
        reverse=True,
    )[:_OPPONENT_RECENT_CHECKPOINTS]
    retained = historical_set | set(ordinary)
    for iteration, path in checkpoints.items():
        if iteration not in retained:
            path.unlink(missing_ok=True)


def _prune_light_checkpoints(directory: str | Path) -> None:
    """Сохраняет light-checkpoint только если есть парный тяжёлый checkpoint."""
    retained_heavy = set(_heavy_checkpoints(directory))
    for path in Path(directory).glob(f"{_LIGHT_CHECKPOINT_PREFIX}*.pt"):
        iteration = _checkpoint_iteration(path, _LIGHT_CHECKPOINT_PREFIX)
        if iteration is None or iteration not in retained_heavy:
            path.unlink(missing_ok=True)


def _load_full_checkpoint_strategy_state(
    path: Path,
    agent: DeepCFRAgent | None = None,
) -> dict[str, torch.Tensor]:
    """Извлекает strategy-веса из full checkpoint и проверяет игровой контракт."""
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict):
        raise ValueError(f"Некорректный full checkpoint: {path}")
    version = checkpoint.get("action_space_version")
    if version != ACTION_SPACE_VERSION:
        raise ValueError(
            f"В full checkpoint {path} action-space '{version}', "
            f"ожидался '{ACTION_SPACE_VERSION}'"
        )
    if agent is not None:
        validate_network_metadata = getattr(agent, "_validate_checkpoint_network_metadata", None)
        if validate_network_metadata is not None:
            validate_network_metadata(checkpoint)
    state_dict = checkpoint.get("strategy_net")
    if not isinstance(state_dict, dict):
        raise ValueError(f"В full checkpoint {path} отсутствует strategy_net")
    return state_dict


def _resolve_checkpoint_path(
    requested_iteration: int,
    checkpoints: dict[int, Path],
) -> Path:
    """Берёт точный checkpoint или ближайший доступный более старый."""
    if not checkpoints:
        raise ValueError("Нет full checkpoint для загрузки весов оппонентов")
    if requested_iteration in checkpoints:
        return checkpoints[requested_iteration]
    older = [iteration for iteration in checkpoints if iteration <= requested_iteration]
    if older:
        return checkpoints[max(older)]
    return checkpoints[min(checkpoints)]


def _sample_checkpoint_iterations(candidates: list[int], count: int) -> list[int]:
    """Выбирает без повторов, а при нехватке доступных весов дублирует старые."""
    if not candidates:
        return []
    if len(candidates) >= count:
        return random.sample(candidates, count)
    result = list(candidates)
    result.extend(random.choices(candidates, k=count - len(result)))
    return result


def _opponent_checkpoint_paths(
    iteration: int,
    checkpoint_dir: str | Path,
    num_opponents: int,
    checkpoint_every: int,
    historical_every: int,
) -> list[Path]:
    """Строит warm-up и устойчивую смесь прошлых full checkpoint."""
    if not 1 <= int(num_opponents) <= 5:
        raise ValueError("Число checkpoint-оппонентов должно быть от 1 до 5")
    checkpoint_every = max(1, int(checkpoint_every))
    checkpoints = {
        number: path for number, path in _heavy_checkpoints(checkpoint_dir).items()
        if number < int(iteration)
    }
    if not checkpoints:
        return []

    latest_slot_index = (int(iteration) - 1) // checkpoint_every
    latest_slot = latest_slot_index * checkpoint_every
    if latest_slot_index <= 6:
        requested = [
            slot * checkpoint_every
            for slot in range(latest_slot_index, max(1, latest_slot_index - 4) - 1, -1)
        ]
        requested = requested[:num_opponents]
        requested.extend([checkpoint_every] * (num_opponents - len(requested)))
    else:
        recent_candidates = [
            slot * checkpoint_every
            for slot in range(max(1, latest_slot_index - 10), latest_slot_index)
        ]
        historical = sorted(
            number for number in checkpoints
            if number % max(1, int(historical_every)) == 0
        )
        active_historical = historical[-2] if len(historical) >= 2 else (
            historical[-1] if historical else None
        )
        requested = [latest_slot]
        needs_separate_historical_slot = (
            active_historical is not None
            and active_historical != latest_slot
            and num_opponents > 1
        )
        requested += _sample_checkpoint_iterations(
            recent_candidates,
            num_opponents - 1 - int(needs_separate_historical_slot),
        )
        if needs_separate_historical_slot:
            requested.append(active_historical)

    return [_resolve_checkpoint_path(number, checkpoints) for number in requested]


class OpponentPoolSchedule:
    """Фиксирует состав opponent pool до следующего checkpoint-слота."""

    def __init__(self, checkpoint_dir: str | Path, checkpoint_every: int, historical_every: int):
        self._checkpoint_dir = checkpoint_dir
        self._checkpoint_every = max(1, int(checkpoint_every))
        self._historical_every = max(1, int(historical_every))
        self._slot: int | None = None
        self._paths: tuple[Path, ...] = ()

    def paths_for_iteration(self, iteration: int, num_opponents: int) -> list[Path]:
        slot = (int(iteration) - 1) // self._checkpoint_every
        if slot != self._slot:
            self._paths = tuple(_opponent_checkpoint_paths(
                iteration=iteration,
                checkpoint_dir=self._checkpoint_dir,
                num_opponents=num_opponents,
                checkpoint_every=self._checkpoint_every,
                historical_every=self._historical_every,
            ))
            self._slot = slot
        return list(self._paths)


def _configure_opponent_pool(
    agent: DeepCFRAgent,
    iteration: int,
    checkpoint_dir: str | Path,
    traversing_player: int,
    state_cache: dict[Path, dict[str, torch.Tensor]] | None = None,
    selected_paths: list[Path] | None = None,
) -> list[Path]:
    paths = list(selected_paths) if selected_paths is not None else _opponent_checkpoint_paths(
        iteration=iteration,
        checkpoint_dir=checkpoint_dir,
        num_opponents=agent.num_players - 1,
        checkpoint_every=int(cfg_get("checkpoint_save_every", 1000)),
        historical_every=int(cfg_get("checkpoint_keep_every", 50000)),
    )
    if not paths:
        agent.clear_opponent_advantage_states()
        return []
    random.shuffle(paths)
    if state_cache is None:
        state_cache = {}
    for path in dict.fromkeys(paths):
        resolved_path = path.resolve()
        if resolved_path not in state_cache:
            state_cache[resolved_path] = _load_full_checkpoint_strategy_state(resolved_path, agent)
    agent.set_opponent_strategy_states(
        [state_cache[path.resolve()] for path in paths],
        traversing_player=traversing_player,
    )
    return paths


def _configure_strategy_opponent_pool(
    agent: DeepCFRAgent,
    checkpoint_by_player: dict[int, Path],
    strategy_positions: tuple[int, ...],
    current_strategy_path: Path | None,
    traversing_player: int,
    state_cache: dict[Path, dict[str, torch.Tensor]],
) -> list[Path]:
    """Назначает актуальную strategy и исторические strategy checkpoint по позициям."""
    states_by_player: dict[int, dict[str, torch.Tensor]] = {}
    if strategy_positions:
        if current_strategy_path is None:
            current_strategy_state = agent.strategy_net.state_dict()
        else:
            resolved_current_strategy_path = current_strategy_path.resolve()
            if resolved_current_strategy_path not in state_cache:
                state_cache[resolved_current_strategy_path] = _load_full_checkpoint_strategy_state(
                    resolved_current_strategy_path, agent
                )
            current_strategy_state = state_cache[resolved_current_strategy_path]
        states_by_player = {
            player_id: current_strategy_state for player_id in strategy_positions
        }
    for player_id, path in checkpoint_by_player.items():
        resolved_path = path.resolve()
        if resolved_path not in state_cache:
            state_cache[resolved_path] = _load_full_checkpoint_strategy_state(resolved_path, agent)
        states_by_player[player_id] = state_cache[resolved_path]
    agent.set_opponent_strategy_states_by_player(states_by_player, traversing_player)
    return list(checkpoint_by_player.values())


def _atomic_torch_save(payload: dict[str, Any], path: str | Path) -> None:
    """Записывает checkpoint атомарно, чтобы не оставить битый файл при остановке."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=target.parent, suffix=".pt", delete=False) as file:
        temporary = Path(file.name)
    try:
        torch.save(payload, temporary)
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)


def _network_architecture(network: PokerNetwork) -> dict[str, int | str]:
    """Возвращает минимальный контракт формы сети для строгого HU resume."""
    if network.architecture == CARD_CONTEXT_ARCHITECTURE:
        return {
            "network_architecture": network.architecture,
            "card_feature_size": CARD_FEATURE_SIZE,
            "input_size": int(
                network.card_encoder[0].in_features + network.context_encoder[0].in_features
            ),
            "hidden_size": int(network.card_encoder[0].out_features),
            "num_actions": int(network.action_head.out_features),
        }
    return {
        "network_architecture": network.architecture,
        "input_size": int(network.base[0].in_features),
        "hidden_size": int(network.base[0].out_features),
        "num_actions": int(network.action_head.out_features),
    }


def _normalize_hu_network_architecture(
    schema: object,
    expected_schema: dict[str, int | str],
) -> dict[str, int | str]:
    if not isinstance(schema, dict):
        raise ValueError("HU checkpoint имеет некорректное описание сети")
    normalized = dict(schema)
    expected_architecture = expected_schema["network_architecture"]
    if "network_architecture" not in normalized:
        if expected_architecture != MONOLITHIC_ARCHITECTURE:
            raise ValueError(
                "HU checkpoint не содержит метаданные архитектуры и несовместим с card_context_v1"
            )
        checkpoint_architecture = MONOLITHIC_ARCHITECTURE
        normalized["network_architecture"] = checkpoint_architecture
    else:
        checkpoint_architecture = normalized["network_architecture"]
        if not isinstance(checkpoint_architecture, str):
            raise ValueError("HU checkpoint имеет некорректное значение архитектуры сети")
    if checkpoint_architecture != expected_architecture:
        raise ValueError("HU checkpoint имеет несовместимую архитектуру сети")
    if checkpoint_architecture == CARD_CONTEXT_ARCHITECTURE:
        if normalized.get("card_feature_size") != CARD_FEATURE_SIZE:
            raise ValueError("HU checkpoint имеет несовместимый размер card-признаков")
    return normalized


def _normalize_hu_architecture(
    architecture: object,
    expected_architecture: dict[str, object],
) -> dict[str, object]:
    if not isinstance(architecture, dict):
        raise ValueError("HU checkpoint не содержит описание архитектуры")
    advantage = architecture.get("advantage")
    advantage_target = architecture.get("advantage_target")
    strategy = architecture.get("strategy")
    expected_advantage = expected_architecture["advantage"]
    expected_advantage_target = expected_architecture["advantage_target"]
    expected_strategy = expected_architecture["strategy"]
    if (
        not isinstance(advantage, list)
        or not isinstance(advantage_target, list)
        or not isinstance(expected_advantage, list)
        or not isinstance(expected_advantage_target, list)
        or len(advantage) != len(expected_advantage)
        or len(advantage_target) != len(expected_advantage_target)
        or not isinstance(expected_strategy, dict)
    ):
        raise ValueError("HU checkpoint имеет несовместимую архитектуру")
    return {
        "advantage": [
            _normalize_hu_network_architecture(schema, expected)
            for schema, expected in zip(advantage, expected_advantage, strict=True)
        ],
        "advantage_target": [
            _normalize_hu_network_architecture(schema, expected)
            for schema, expected in zip(advantage_target, expected_advantage_target, strict=True)
        ],
        "strategy": _normalize_hu_network_architecture(strategy, expected_strategy),
    }


def _capture_rng_state() -> dict[str, Any]:
    """Сохраняет все генераторы, влияющие на HU traversal и reservoir."""
    numpy_algorithm, numpy_state, numpy_position, numpy_has_gauss, numpy_cached_gaussian = np.random.get_state()
    state: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": {
            "algorithm": str(numpy_algorithm),
            "state": torch.from_numpy(numpy_state.copy()),
            "position": int(numpy_position),
            "has_gauss": int(numpy_has_gauss),
            "cached_gaussian": float(numpy_cached_gaussian),
        },
        "torch_cpu": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["torch_cuda"] = torch.cuda.get_rng_state_all()
    return state


def _restore_rng_state(payload: dict[str, Any]) -> None:
    """Восстанавливает генераторы только из полного HU checkpoint."""
    required = ("python", "numpy", "torch_cpu")
    if not isinstance(payload, dict) or any(key not in payload for key in required):
        raise ValueError("В HU checkpoint отсутствует полное состояние RNG")
    random.setstate(payload["python"])
    numpy_payload = payload["numpy"]
    numpy_keys = ("algorithm", "state", "position", "has_gauss", "cached_gaussian")
    if not isinstance(numpy_payload, dict) or any(key not in numpy_payload for key in numpy_keys):
        raise ValueError("В HU checkpoint отсутствует корректное NumPy RNG состояние")
    numpy_state = numpy_payload["state"]
    if not isinstance(numpy_state, torch.Tensor) or numpy_state.dtype != torch.uint32:
        raise ValueError("В HU checkpoint повреждено NumPy RNG состояние")
    np.random.set_state((
        str(numpy_payload["algorithm"]),
        numpy_state.detach().cpu().numpy(),
        int(numpy_payload["position"]),
        int(numpy_payload["has_gauss"]),
        float(numpy_payload["cached_gaussian"]),
    ))
    torch.set_rng_state(payload["torch_cpu"])
    cuda_state = payload.get("torch_cuda")
    if cuda_state is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(cuda_state)


def _advantage_buffer_payload(buffer: AdvantageBuffer) -> dict[str, Any]:
    count = len(buffer)
    return {
        "capacity": int(buffer.capacity),
        "state_dim": int(buffer._states.shape[1]),
        "cur_id": int(buffer._cur_id),
        "count": int(count),
        "eviction_count": int(buffer.eviction_count),
        "skip_count": int(buffer.skip_count),
        "states": torch.from_numpy(buffer._states[:count].copy()),
        "regrets": torch.from_numpy(buffer._regrets[:count].copy()),
        "masks": torch.from_numpy(buffer._masks[:count].copy()),
        "iterations": torch.from_numpy(buffer._iterations[:count].copy()),
    }


def _strategy_buffer_payload(buffer: HuStrategyBuffer) -> dict[str, Any]:
    count = len(buffer)
    return {
        "capacity": int(buffer.capacity),
        "state_dim": int(buffer.state_dim),
        "cur_id": int(buffer._cur_id),
        "count": int(count),
        "eviction_count": int(buffer.eviction_count),
        "skip_count": int(buffer.skip_count),
        "states": torch.from_numpy(buffer._states[:count].copy()),
        "actor_ids": torch.from_numpy(buffer._actor_ids[:count].copy()),
        "policies": torch.from_numpy(buffer._policies[:count].copy()),
        "masks": torch.from_numpy(buffer._masks[:count].copy()),
        "iterations": torch.from_numpy(buffer._iterations[:count].copy()),
    }


def _buffer_count(payload: dict[str, Any], capacity: int) -> tuple[int, int]:
    if not isinstance(payload, dict) or int(payload.get("capacity", -1)) != int(capacity):
        raise ValueError("HU checkpoint имеет несовместимую ёмкость replay-буфера")
    count = int(payload.get("count", -1))
    cur_id = int(payload.get("cur_id", -1))
    if count < 0 or count > capacity or cur_id < count:
        raise ValueError("HU checkpoint имеет некорректное состояние replay-буфера")
    return count, cur_id


def _checkpoint_array(payload: dict[str, Any], key: str) -> np.ndarray:
    value = payload.get(key)
    if not isinstance(value, torch.Tensor):
        raise ValueError("HU checkpoint содержит небезопасный или поврежденный replay-буфер")
    return value.detach().cpu().numpy()


def _restore_advantage_buffer(buffer: AdvantageBuffer, payload: dict[str, Any]) -> None:
    count, cur_id = _buffer_count(payload, buffer.capacity)
    state_dim = int(buffer._states.shape[1])
    if int(payload.get("state_dim", -1)) != state_dim:
        raise ValueError("HU checkpoint имеет несовместимый размер advantage-буфера")
    fields = {
        "states": (count, state_dim),
        "regrets": (count, NUM_ACTIONS),
        "masks": (count, NUM_ACTIONS),
        "iterations": (count,),
    }
    arrays = {name: _checkpoint_array(payload, name) for name in fields}
    if any(arrays[name].shape != shape for name, shape in fields.items()):
        raise ValueError("HU checkpoint имеет повреждённый advantage-буфер")
    buffer._states[:count] = arrays["states"]
    buffer._regrets[:count] = arrays["regrets"]
    buffer._masks[:count] = arrays["masks"]
    buffer._iterations[:count] = arrays["iterations"]
    buffer._cur_id = cur_id
    buffer._size = count
    buffer.eviction_count = int(payload.get("eviction_count", 0))
    buffer.skip_count = int(payload.get("skip_count", 0))


def _restore_strategy_buffer(buffer: HuStrategyBuffer, payload: dict[str, Any]) -> None:
    count, cur_id = _buffer_count(payload, buffer.capacity)
    if int(payload.get("state_dim", -1)) != int(buffer.state_dim):
        raise ValueError("HU checkpoint имеет несовместимый размер strategy-буфера")
    fields = {
        "states": (count, buffer.state_dim),
        "actor_ids": (count,),
        "policies": (count, NUM_ACTIONS),
        "masks": (count, NUM_ACTIONS),
        "iterations": (count,),
    }
    arrays = {name: _checkpoint_array(payload, name) for name in fields}
    if any(arrays[name].shape != shape for name, shape in fields.items()):
        raise ValueError("HU checkpoint имеет повреждённый strategy-буфер")
    if not np.all(np.isin(arrays["actor_ids"], (0, 1))):
        raise ValueError("HU checkpoint содержит недопустимый actor_id strategy-буфера")
    buffer._states[:count] = arrays["states"]
    buffer._actor_ids[:count] = arrays["actor_ids"]
    buffer._policies[:count] = arrays["policies"]
    buffer._masks[:count] = arrays["masks"]
    buffer._iterations[:count] = arrays["iterations"]
    buffer._cur_id = cur_id
    buffer.eviction_count = int(payload.get("eviction_count", 0))
    buffer.skip_count = int(payload.get("skip_count", 0))


def _optimizer_configuration(optimizer: torch.optim.Optimizer) -> list[dict[str, Any]]:
    """Фиксирует гиперпараметры optimizer, исключая внутренние ссылки на параметры."""
    return [
        {key: value for key, value in group.items() if key != "params"}
        for group in optimizer.param_groups
    ]


def _hu_trajectory_configuration(agent: DeepCFRAgent) -> dict[str, Any]:
    """Возвращает все параметры, меняющие траекторию HU обучения."""
    return {
        "advantage_accumulation": str(agent.advantage_accumulation),
        "discount_alpha": float(agent.discount_alpha),
        "discount_gamma": float(agent.discount_gamma),
        "advantage_regret_norm": str(agent.advantage_regret_norm),
        "advantage_regret_clip": agent.advantage_regret_clip,
        "advantage_reward_scale": float(agent.advantage_reward_scale),
        "advantage_loss": str(agent.advantage_loss),
        "advantage_huber_delta": float(agent.advantage_huber_delta),
        "advantage_batch_size": int(agent.advantage_batch_size),
        "strategy_batch_size": int(agent.strategy_batch_size),
        "advantage_epochs": int(agent.advantage_epochs),
        "strategy_epochs": int(agent.strategy_epochs),
        "advantage_train_steps": agent.advantage_train_steps,
        "strategy_train_steps": agent.strategy_train_steps,
        "advantage_buffer_reservoir": bool(agent.advantage_buffer_reservoir),
        "clear_strategy_buffer_each_iteration": bool(agent.clear_strategy_buffer_each_iteration),
        "hu_strategy_buffer_reservoir": True,
        "strategy_distillation_lambda": float(agent.strategy_distillation_lambda),
        "strategy_distillation_temperature": float(agent.strategy_distillation_temperature),
        "strategy_distillation_anneal_iterations": int(agent.strategy_distillation_anneal_iterations),
        "advantage_optimizers": [
            _optimizer_configuration(optimizer)
            for optimizer in agent.hu_advantage_optimizers
        ],
        "strategy_optimizer": _optimizer_configuration(agent.strategy_optimizer),
        "advantage_buffer_capacities": [
            int(buffer.capacity) for buffer in agent.hu_advantage_buffers
        ],
        "strategy_buffer_capacity": int(agent.hu_strategy_buffer.capacity),
        "training_error_mode": cfg_training_error_mode(),
        "training_max_failed_traversals_per_iteration": cfg_training_max_failed_traversals_per_iteration(),
    }


def _build_hu_checkpoint(agent: DeepCFRAgent, seed: int | None = None) -> dict[str, Any]:
    """Строит полный checkpoint только на границе завершённой HU-итерации."""
    if not bool(getattr(agent, "hu_current_policy_self_play", False)):
        raise ValueError("Полный HU checkpoint доступен только в hu_current_policy_self_play")
    advantage_nets = tuple(agent.hu_advantage_nets)
    advantage_targets = tuple(agent.hu_advantage_target_nets)
    advantage_optimizers = tuple(agent.hu_advantage_optimizers)
    advantage_buffers = tuple(agent.hu_advantage_buffers)
    if not all(len(items) == 2 for items in (advantage_nets, advantage_targets, advantage_optimizers, advantage_buffers)):
        raise ValueError("HU checkpoint требует две независимые advantage-ноги")
    mode = {
        "hu_current_policy_self_play": True,
        "num_players": int(agent.num_players),
        "num_trainable_players": int(agent.num_trainable_players),
        "use_multi_agent_advantage": bool(agent.use_multi_agent),
        "encoding_version": str(agent.encoding_version),
        "encoder_input_size": int(agent.input_size),
    }
    return {
        "checkpoint_format_version": CHECKPOINT_FORMAT_VERSION,
        "checkpoint_kind": _HU_CHECKPOINT_KIND,
        "hu_checkpoint_version": _HU_CHECKPOINT_VERSION,
        "action_space_version": ACTION_SPACE_VERSION,
        "action_labels": list(ACTION_LABELS),
        "num_actions": NUM_ACTIONS,
        "iteration": int(agent.iteration_count),
        "seed": seed,
        "mode": mode,
        "config": {**mode, **_hu_trajectory_configuration(agent)},
        "architecture": {
            "advantage": [_network_architecture(network) for network in advantage_nets],
            "advantage_target": [_network_architecture(network) for network in advantage_targets],
            "strategy": _network_architecture(agent.strategy_net),
        },
        "update_order": list(_HU_UPDATE_ORDER),
        "advantage_legs": [
            {
                "network": network.state_dict(),
                "target_network": target.state_dict(),
                "optimizer": optimizer.state_dict(),
                "buffer": _advantage_buffer_payload(buffer),
            }
            for network, target, optimizer, buffer in zip(
                advantage_nets,
                advantage_targets,
                advantage_optimizers,
                advantage_buffers,
                strict=True,
            )
        ],
        "strategy": {
            "network": agent.strategy_net.state_dict(),
            "optimizer": agent.strategy_optimizer.state_dict(),
            "buffer": _strategy_buffer_payload(agent.hu_strategy_buffer),
        },
        "rng": _capture_rng_state(),
    }


def _validate_hu_checkpoint(agent: DeepCFRAgent, checkpoint: object) -> dict[str, Any]:
    if not isinstance(checkpoint, dict) or checkpoint.get("checkpoint_kind") != _HU_CHECKPOINT_KIND:
        raise ValueError("Для HU resume требуется полный HU checkpoint")
    if checkpoint.get("hu_checkpoint_version") != _HU_CHECKPOINT_VERSION:
        raise ValueError("HU checkpoint имеет несовместимую версию")
    if checkpoint.get("checkpoint_format_version") != CHECKPOINT_FORMAT_VERSION:
        raise ValueError("HU checkpoint имеет несовместимый общий формат")
    if checkpoint.get("action_space_version") != ACTION_SPACE_VERSION or checkpoint.get("num_actions") != NUM_ACTIONS:
        raise ValueError("HU checkpoint имеет другое пространство действий")
    if checkpoint.get("action_labels") != list(ACTION_LABELS):
        raise ValueError("HU checkpoint имеет несовместимый action_labels контракт")
    iteration = checkpoint.get("iteration")
    if isinstance(iteration, bool) or not isinstance(iteration, int) or iteration < 0:
        raise ValueError("HU checkpoint имеет некорректный iteration")
    mode = checkpoint.get("mode")
    expected_mode = {
        "hu_current_policy_self_play": True,
        "num_players": int(agent.num_players),
        "num_trainable_players": int(agent.num_trainable_players),
        "use_multi_agent_advantage": bool(agent.use_multi_agent),
        "encoding_version": str(agent.encoding_version),
        "encoder_input_size": int(agent.input_size),
    }
    if not isinstance(mode, dict) or any(mode.get(key) != value for key, value in expected_mode.items()):
        raise ValueError("HU checkpoint имеет несовместимый режим или encoder")
    config = checkpoint.get("config")
    expected_config = {**expected_mode, **_hu_trajectory_configuration(agent)}
    if not isinstance(config, dict):
        raise ValueError("HU checkpoint не содержит полную конфигурацию")
    unsupported_config_keys = set(config) - set(expected_config) - _HU_RUNTIME_CONFIG_ALLOWLIST
    if unsupported_config_keys:
        raise ValueError(
            "HU checkpoint содержит неподдерживаемую конфигурацию: "
            + ", ".join(sorted(unsupported_config_keys))
        )
    for key, expected_value in expected_config.items():
        if config.get(key) != expected_value:
            raise ValueError(f"HU checkpoint имеет несовместимую конфигурацию: {key}")
    if int(agent.num_players) != 2 or int(agent.num_trainable_players) != 2:
        raise ValueError("HU resume требует ровно двух игроков и двух trainable players")
    if checkpoint.get("update_order") != _HU_UPDATE_ORDER:
        raise ValueError("HU checkpoint имеет неизвестный порядок обновления")
    advantage_legs = checkpoint.get("advantage_legs")
    strategy = checkpoint.get("strategy")
    if not isinstance(advantage_legs, list) or len(advantage_legs) != 2 or not isinstance(strategy, dict):
        raise ValueError("HU checkpoint не содержит полный набор training state")
    expected_architecture = {
        "advantage": [_network_architecture(network) for network in agent.hu_advantage_nets],
        "advantage_target": [_network_architecture(network) for network in agent.hu_advantage_target_nets],
        "strategy": _network_architecture(agent.strategy_net),
    }
    architecture = _normalize_hu_architecture(
        checkpoint.get("architecture"),
        expected_architecture,
    )
    if architecture != expected_architecture:
        raise ValueError("HU checkpoint имеет несовместимую архитектуру")
    for leg in advantage_legs:
        if not isinstance(leg, dict) or any(key not in leg for key in ("network", "target_network", "optimizer", "buffer")):
            raise ValueError("HU checkpoint содержит неполную advantage-ногу")
    if any(key not in strategy for key in ("network", "optimizer", "buffer")):
        raise ValueError("HU checkpoint содержит неполное strategy-состояние")
    if not isinstance(checkpoint.get("rng"), dict):
        raise ValueError("HU checkpoint не содержит состояние RNG")
    return checkpoint


def _load_hu_checkpoint(agent: DeepCFRAgent, path: str | Path) -> dict[str, Any]:
    """Строго восстанавливает HU training state до следующей итерации."""
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except (pickle.UnpicklingError, RuntimeError, ValueError) as error:
        raise ValueError("HU checkpoint не удалось безопасно прочитать") from error
    checkpoint = _validate_hu_checkpoint(agent, payload)
    for leg, network, target, optimizer, buffer in zip(
        checkpoint["advantage_legs"],
        agent.hu_advantage_nets,
        agent.hu_advantage_target_nets,
        agent.hu_advantage_optimizers,
        agent.hu_advantage_buffers,
        strict=True,
    ):
        try:
            network.load_state_dict(leg["network"], strict=True)
            target.load_state_dict(leg["target_network"], strict=True)
            optimizer.load_state_dict(leg["optimizer"])
        except (RuntimeError, ValueError, KeyError) as error:
            raise ValueError("HU checkpoint содержит несовместимые веса или optimizer") from error
        _restore_advantage_buffer(buffer, leg["buffer"])
    try:
        agent.strategy_net.load_state_dict(checkpoint["strategy"]["network"], strict=True)
        agent.strategy_optimizer.load_state_dict(checkpoint["strategy"]["optimizer"])
    except (RuntimeError, ValueError, KeyError) as error:
        raise ValueError("HU checkpoint содержит несовместимые strategy веса или optimizer") from error
    _restore_strategy_buffer(agent.hu_strategy_buffer, checkpoint["strategy"]["buffer"])
    agent.iteration_count = checkpoint["iteration"]
    _restore_rng_state(checkpoint["rng"])
    return checkpoint


def _save_hu_checkpoint(agent: DeepCFRAgent, path: str | Path, seed: int | None = None) -> Path:
    """Атомарно записывает полный HU checkpoint после завершённой итерации."""
    target = Path(path)
    _atomic_torch_save(_build_hu_checkpoint(agent, seed=seed), target)
    return target


def _save_iteration_checkpoint(
    agent: DeepCFRAgent,
    save_dir: str | Path,
    iteration: int,
    prefix: str = "multi_checkpoint_iter_",
    seed: int | None = None,
) -> Path:
    path = Path(save_dir) / f"{prefix}{int(iteration)}.pt"
    _atomic_torch_save(agent._build_checkpoint(seed=seed), path)
    if prefix == _HEAVY_CHECKPOINT_PREFIX:
        _prune_full_checkpoints(save_dir, int(cfg_get("checkpoint_keep_every", 50000)))
    return path


def _save_iteration_light_checkpoint(
    agent: DeepCFRAgent,
    save_dir: str | Path,
    iteration: int,
    prefix: str = _LIGHT_CHECKPOINT_PREFIX,
    seed: int | None = None,
) -> Path:
    """Сохраняет только усреднённую стратегию для инференса."""
    path = Path(save_dir) / f"{prefix}{int(iteration)}.pt"
    _atomic_torch_save(agent.build_light_checkpoint(seed=seed), path)
    _prune_light_checkpoints(save_dir)
    return path


def _get_process_rss_mb() -> float | None:
    """Возвращает RSS процесса, если psutil доступен."""
    try:
        import psutil

        return psutil.Process().memory_info().rss / (1024 * 1024)
    except ImportError:
        return None


def _apply_process_priority(priority: str) -> None:
    """Применяет приоритет Windows и явно сообщает результат в консоль."""
    if priority == "normal":
        print("Приоритет процесса: normal (не запрашивалось повышение).")
        return
    if priority != "above_normal":
        raise ValueError("process_priority должен быть normal или above_normal.")
    if os.name != "nt":
        print("Приоритет процесса: НЕ ПРИМЕНЁН (above_normal поддержан только в Windows).")
        return

    try:
        import psutil
    except ImportError:
        print("Приоритет процесса: НЕ ПРИМЕНЁН (не установлен psutil).")
        return

    try:
        psutil.Process().nice(psutil.ABOVE_NORMAL_PRIORITY_CLASS)
        print("Приоритет процесса: ПРИМЕНЁН (above_normal).")
    except psutil.Error as error:
        print(f"Приоритет процесса: НЕ ПРИМЕНЁН (Windows отклонил запрос: {error}).")


def _format_fanout_histogram(histogram: dict[int, int]) -> str:
    return ", ".join(f"{fanout}:{count}" for fanout, count in sorted(histogram.items())) or "—"


def _print_depth_histogram(depth_histogram: dict[int, int]) -> None:
    print("  Глубина: " + _format_fanout_histogram(depth_histogram))


def _print_opponent_checkpoints(
    paths: list[Path],
    random_opponents: int = 0,
) -> None:
    """Выводит фактически загруженные advantage-checkpoint оппонентов."""
    if random_opponents > 0:
        print(f"  Играет против: {random_opponents} RandomAgent")
        return
    if not paths:
        print("  Играет против: холодный старт (advantage checkpoint ещё не созданы)")
        return
    print("  Играет против:")
    for path in paths:
        print(f"    {path.name}")


def _print_current_strategy_opponents(count: int) -> None:
    print(f"  Играет против актуальной strategy: {count}")


def _format_iteration_summary(
    iteration_elapsed: float,
    traversal_elapsed: float,
    advantage_loss: float,
    strategy_loss: float,
    opponent_setup_elapsed: float = 0.0,
) -> str:
    return (
        f"Time/Iteration={iteration_elapsed:.1f}s | "
        f"Time/Traversal={traversal_elapsed:.1f}s | "
        f"Time/OpponentPoolSetup={opponent_setup_elapsed:.1f}s | "
        f"Loss/Advantage={advantage_loss:.6f} | "
        f"Loss/Strategy={strategy_loss:.6f}"
    )


@contextmanager
def _traversal_thread_limit(enabled: bool):
    """Ограничивает PyTorch одним потоком только на мелких forward обхода."""
    previous_threads = torch.get_num_threads()
    if enabled:
        torch.set_num_threads(1)
    try:
        yield
    finally:
        if enabled:
            torch.set_num_threads(previous_threads)


@contextmanager
def _training_thread_limit(num_threads: int | None):
    """Задаёт число intra-op потоков только для loss/backprop фазы."""
    if num_threads is None:
        yield
        return

    requested_threads = int(num_threads)
    if requested_threads <= 0:
        raise ValueError("training_torch_threads должен быть положительным.")

    previous_threads = torch.get_num_threads()
    torch.set_num_threads(requested_threads)
    try:
        yield
    finally:
        torch.set_num_threads(previous_threads)


def _log_multi_cfr_diagnostics(agent, writer, iteration, traversing_player, traversals_per_iteration):
    """Пишет только метрики обхода, независимые от представления действий."""
    del traversing_player, traversals_per_iteration
    stats = agent.get_traversal_stats()
    print(
        f"  Обходы: узлы={stats['nodes']}, макс_глубина={stats['max_depth']}, "
        f"попытки={stats.get('attempted', 0)}, успешные={stats.get('successful', 0)}, "
        f"ошибки={stats.get('failed', 0)}, отменённые_образцы={stats.get('cancelled_samples', 0)}, "
        f"лимит_глубины={stats.get('depth_limit_hits', 0)}"
    )
    if writer is None:
        return stats
    for key in (
        "nodes",
        "terminal_nodes",
        "max_depth",
        "recorded_nodes",
        "buffer_skip_ratio",
        "attempted",
        "successful",
        "failed",
        "cancelled_samples",
        "depth_limit_hits",
    ):
        writer.add_scalar(f"Traversal/{key}", stats[key], iteration)
    writer.add_scalar("Memory/Advantage", len(agent.advantage_buffer), iteration)
    writer.add_scalar("Memory/Strategy", len(agent.strategy_buffer), iteration)
    return stats


def _create_writer(log_dir: str | Path | None):
    """Создаёт TensorBoard writer только когда указан каталог логов."""
    if not log_dir:
        return None
    try:
        from torch.utils.tensorboard import SummaryWriter
    except ImportError:
        print("TensorBoard недоступен: метрики будут выведены только в консоль.")
        return None
    return SummaryWriter(log_dir=str(log_dir))


def _new_hand(num_players: int, seed: int) -> pkrs.State:
    return pkrs.State.from_seed(
        n_players=num_players,
        button=seed % num_players,
        sb=1.0,
        bb=2.0,
        stake=200.0,
        seed=seed,
    )


def evaluate_against_random(
    agent: DeepCFRAgent,
    num_games: int = 100,
    num_players: int = 6,
    return_stats: bool = False,
) -> float | dict[str, float]:
    """Оценивает agent против случайных оппонентов без смены смысла raise-слотов."""
    rewards: list[float] = []
    decisions = raises = 0
    opponents = [RandomAgent(player) for player in range(num_players)]
    for game in range(int(num_games)):
        state = _new_hand(num_players, game)
        while not state.final_state:
            player = int(state.current_player)
            action = agent.choose_action(state, player_id=player) if player == agent.player_id else opponents[player].choose_action(state)
            if player == agent.player_id:
                decisions += 1
                raises += int(action.action == pkrs.ActionEnum.Raise)
            state = state.apply_action(action)
            if state.status != pkrs.StateStatus.Ok:
                break
        rewards.append(float(state.players_state[agent.player_id].reward))
    mean_reward = float(np.mean(rewards)) if rewards else 0.0
    if not return_stats:
        return mean_reward
    return {
        "mean_reward": mean_reward,
        "std_reward": float(np.std(rewards)) if rewards else 0.0,
        "raise_frequency": raises / decisions if decisions else 0.0,
        "games": float(len(rewards)),
    }


class _PerspectiveAgentWrapper:
    """Совместимый адаптер для оценочного кода с разными позициями игрока."""

    def __init__(self, agent: DeepCFRAgent, player_id: int):
        self.agent = agent
        self.player_id = int(player_id)

    def choose_action(self, state):
        return self.agent.choose_action(state, player_id=self.player_id)


def evaluate_against_agent(agent: DeepCFRAgent, opponent_agent, num_games: int = 100) -> float:
    """Возвращает среднюю награду первого игрока в heads-up оценке."""
    rewards: list[float] = []
    for game in range(int(num_games)):
        state = _new_hand(2, game)
        while not state.final_state:
            current = int(state.current_player)
            policy = agent if current == agent.player_id else opponent_agent
            action = policy.choose_action(state)
            state = state.apply_action(action)
            if state.status != pkrs.StateStatus.Ok:
                break
        rewards.append(float(state.players_state[agent.player_id].reward))
    return float(np.mean(rewards)) if rewards else 0.0


def evaluate_against_checkpoint_agents(agent: DeepCFRAgent, opponent_agents, num_games: int = 100) -> float:
    if not opponent_agents:
        return evaluate_against_random(agent, num_games=num_games, num_players=agent.num_players)
    return evaluate_against_agent(agent, opponent_agents[0], num_games)


class _HuConditionedStrategyBuffer:
    """Представляет HU reservoir в форме, ожидаемой общим методом обучения strategy."""

    def __init__(self, buffer: HuStrategyBuffer):
        self._buffer = buffer

    def sample(self, num_samples: int = -1):
        return self._buffer.sample_conditioned(num_samples)

    def __len__(self) -> int:
        return len(self._buffer)


def _create_hu_current_policy_coordinator(
    agent: DeepCFRAgent,
) -> HuCurrentPolicySelfPlayCoordinator[pkrs.State]:
    """Создаёт изолированные P0/P1 advantage-ноги для HU режима."""
    advantage_nets = [agent.advantage_net, deepcopy(agent.advantage_net).to(agent.device)]
    advantage_target_nets = [
        agent.advantage_target_net,
        deepcopy(agent.advantage_target_net).to(agent.device),
    ]
    for target in advantage_target_nets:
        target.eval()
        for parameter in target.parameters():
            parameter.requires_grad_(False)
    advantage_optimizers = [
        agent.optimizer,
        torch.optim.AdamW(
            advantage_nets[1].parameters(),
            lr=float(cfg_get("advantage_lr", 1e-4)),
            weight_decay=float(cfg_get("advantage_weight_decay", 1e-5)),
        ),
    ]
    advantage_buffers = [
        agent.advantage_buffer,
        AdvantageBuffer(
            int(cfg_get("advantage_memory_size", 300000)),
            agent.input_size,
        ),
    ]

    hidden_size = int(_network_architecture(agent.strategy_net)["hidden_size"])
    strategy_net = PokerNetwork(
        agent.input_size + 2,
        hidden_size,
        architecture=agent.strategy_net.architecture,
    ).to(agent.device)
    strategy_optimizer = torch.optim.AdamW(
        strategy_net.parameters(),
        lr=float(cfg_get("strategy_lr", 5e-5)),
        weight_decay=float(cfg_get("strategy_weight_decay", 1e-5)),
    )
    strategy_buffer = HuStrategyBuffer(
        int(cfg_get("strategy_memory_size", 300000)), agent.input_size
    )
    agent.strategy_net = strategy_net
    agent.strategy_optimizer = strategy_optimizer
    agent.strategy_buffer = _HuConditionedStrategyBuffer(strategy_buffer)
    agent.hu_current_policy_self_play = True
    agent.hu_advantage_nets = tuple(advantage_nets)
    agent.hu_advantage_target_nets = tuple(advantage_target_nets)
    agent.hu_advantage_optimizers = tuple(advantage_optimizers)
    agent.hu_advantage_buffers = tuple(advantage_buffers)
    agent.hu_strategy_buffer = strategy_buffer

    def apply(state: pkrs.State, slot: int) -> pkrs.State:
        action = agent.action_type_to_pokers_action(slot, state)
        next_state = state.apply_action(action)
        if next_state.status != pkrs.StateStatus.Ok:
            raise ValueError(f"HU traversal получил некорректный status: {next_state.status}")
        return next_state

    advantage_losses = [0.0, 0.0]
    strategy_losses: list[float] = []

    def train_advantage(player_id: int, network, target_network, optimizer, buffer) -> float:
        previous = (
            agent.advantage_net,
            agent.advantage_target_net,
            agent.optimizer,
            agent.advantage_buffer,
        )
        try:
            agent.advantage_net = network
            agent.advantage_target_net = target_network
            agent.optimizer = optimizer
            agent.advantage_buffer = buffer
            with _training_thread_limit(cfg_get("training_torch_threads")):
                loss = agent.train_advantage_network_multi(player_id=player_id)
            advantage_losses[player_id] = float(loss)
            return float(loss)
        finally:
            (
                agent.advantage_net,
                agent.advantage_target_net,
                agent.optimizer,
                agent.advantage_buffer,
            ) = previous

    def train_strategy(_network, _optimizer, _buffer) -> float:
        with _training_thread_limit(cfg_get("training_torch_threads")):
            loss = agent.train_strategy_network()
        strategy_losses.append(float(loss))
        return float(loss)

    coordinator = HuCurrentPolicySelfPlayCoordinator(
        advantage_nets=advantage_nets,
        advantage_target_nets=advantage_target_nets,
        advantage_optimizers=advantage_optimizers,
        advantage_buffers=advantage_buffers,
        strategy_net=strategy_net,
        strategy_optimizer=strategy_optimizer,
        strategy_buffer=strategy_buffer,
        adapter=HuTraversalAdapter(
            current_player=lambda state: int(state.current_player),
            is_terminal=lambda state: bool(state.final_state),
            legal_mask=agent.get_legal_action_mask,
            encode=lambda state, player_id: agent._encode_state(state, player_id),
            apply=apply,
            terminal_value=lambda state, player_id: float(state.players_state[player_id].reward),
            normalize_regrets=lambda state, regrets, mask: agent._normalise_regrets(
                regrets,
                state,
                np.flatnonzero(mask).astype(int).tolist(),
            ),
        ),
        train_advantage=train_advantage,
        train_strategy=train_strategy,
    )
    coordinator.training_losses = (advantage_losses, strategy_losses)
    agent.hu_coordinator = coordinator
    return coordinator


def _prepare_hu_current_policy_iteration(agent: DeepCFRAgent) -> None:
    """Очищает только HU replay-буферы, не затрагивая legacy lifecycle."""
    if not agent.advantage_buffer_reservoir:
        for buffer in agent.hu_advantage_buffers:
            buffer.clear()
    if agent.clear_strategy_buffer_each_iteration:
        agent.hu_strategy_buffer.clear()
    agent.reset_traversal_stats()


def _create_hu_traversal_failure_handler(agent: DeepCFRAgent):
    """Возвращает policy skip/strict, совпадающую с legacy training lifecycle."""
    failed_traversals = 0

    def handle(error: TraversalFailure) -> bool:
        nonlocal failed_traversals
        agent.record_traversal_failure(error)
        if cfg_training_error_mode() != "skip_traversal":
            return False
        failed_traversals += 1
        if failed_traversals >= cfg_training_max_failed_traversals_per_iteration():
            raise RuntimeError(
                "Превышен training_max_failed_traversals_per_iteration"
            ) from error
        return True

    return handle


def _train_hu_current_policy_self_play(
    *,
    agent: DeepCFRAgent,
    num_iterations: int,
    traversals_per_iteration: int,
    save_dir: str | Path,
    evaluate_every: int,
    evaluation_games: int,
    num_players: int,
    seed: int | None,
    log_dir: str | Path | None,
    initial_checkpoint: str | Path | None = None,
) -> DeepCFRAgent:
    """Выполняет HU current-policy self-play без внешних opponent/checkpoint policy."""
    coordinator = _create_hu_current_policy_coordinator(agent)
    if initial_checkpoint is not None:
        checkpoint = _load_hu_checkpoint(agent, initial_checkpoint)
        checkpoint_seed = checkpoint.get("seed")
        if seed is not None and checkpoint_seed != seed:
            raise ValueError("HU resume требует тот же seed, что и в checkpoint")
        seed = checkpoint_seed
    start_iteration = agent.iteration_count + 1
    completed_iteration: int | None = None
    writer = _create_writer(log_dir)
    try:
        print(
            "Старт HU current-policy self-play: "
            f"итераций={num_iterations}, обходов/итерацию={traversals_per_iteration}, device={agent.device}"
        )
        print(
            "Порядок фаз HU: обе фазы обходов P0/P1 на frozen snapshots -> "
            "обучение advantage P0/P1 -> обучение shared strategy."
        )
        for iteration in range(start_iteration, start_iteration + int(num_iterations)):
            iteration_started = time.perf_counter()
            agent.iteration_count = iteration
            _prepare_hu_current_policy_iteration(agent)
            handle_traversal_failure = _create_hu_traversal_failure_handler(agent)
            print(f"\nИтерация {iteration} (HU current-policy self-play):")
            traversal_started = time.perf_counter()
            coordinator.run_iteration(
                iteration=iteration,
                traversals_per_player=traversals_per_iteration,
                new_initial_state=lambda player_id, traversal_index: _new_hand(
                    num_players,
                    (seed or 0)
                    + iteration * 2 * max(1, int(traversals_per_iteration))
                    + player_id * max(1, int(traversals_per_iteration))
                    + traversal_index,
                ),
                traversal_context=lambda: _traversal_thread_limit(
                    bool(cfg_get("traversal_single_thread", True))
                ),
                on_traversal_attempt=agent.record_traversal_attempt,
                on_traversal_success=agent.record_traversal_success,
                handle_traversal_failure=handle_traversal_failure,
            )
            traversal_elapsed = time.perf_counter() - traversal_started
            advantage_losses, strategy_losses = coordinator.training_losses
            advantage_loss = float(sum(advantage_losses) / len(advantage_losses))
            strategy_loss = float(strategy_losses[-1]) if strategy_losses else 0.0
            if writer is not None:
                writer.add_scalar("Loss/Advantage", advantage_loss, iteration)
                writer.add_scalar("Loss/Strategy", strategy_loss, iteration)
                writer.add_scalar("Time/Traversal", traversal_elapsed, iteration)
                writer.add_scalar("Train/AdvantageLearningRate", agent.optimizer.param_groups[0]["lr"], iteration)
            if evaluate_every and iteration % int(evaluate_every) == 0:
                evaluation = evaluate_against_random(
                    agent, evaluation_games, num_players, return_stats=True
                )
                print(
                    f"  Оценка против random: reward={evaluation['mean_reward']:.4f}, "
                    f"raise_freq={evaluation['raise_frequency']:.3f}, игр={int(evaluation['games'])}"
                )
            if _checkpoint_save_due(iteration, int(cfg_get("checkpoint_save_every", 1000))):
                checkpoint_path = _save_hu_checkpoint(
                    agent,
                    Path(save_dir) / f"hu_checkpoint_iter_{iteration}.pt",
                    seed=seed,
                )
                print(f"  HU checkpoint: {checkpoint_path}")
            completed_iteration = iteration
            iteration_elapsed = time.perf_counter() - iteration_started
            if writer is not None:
                writer.add_scalar("Time/Iteration", iteration_elapsed, iteration)
            print(_format_iteration_summary(
                iteration_elapsed, traversal_elapsed, advantage_loss, strategy_loss, 0.0
            ))
    finally:
        if writer is not None:
            writer.flush()
            writer.close()
    if completed_iteration is not None:
        final_checkpoint = _save_hu_checkpoint(
            agent,
            Path(save_dir) / "hu_checkpoint_final.pt",
            seed=seed,
        )
        print(f"Финальный HU checkpoint: {final_checkpoint}")
    return agent


def train_self_play_multi(
    num_iterations: int = 1_000,
    traversals_per_iteration: int = 100,
    save_dir: str | Path = "models",
    evaluate_every: int = 10,
    evaluation_games: int = 500,
    num_players: int = 6,
    device: str = "cpu",
    seed: int | None = None,
    initial_checkpoint: str | None = None,
    log_dir: str | Path | None = None,
    opponent_checkpoint_dir: str | Path | None = None,
    trainable_players: int | None = None,
    teacher_strategy_checkpoint: str | Path | None = None,
    teacher_transfer_enabled: bool | None = None,
    teacher_transfer_mode: str | None = None,
    teacher_transfer_checkpoint: str | Path | None = None,
    teacher_transfer_freeze_card_encoder: bool | None = None,
    teacher_hu_aux_distillation_enabled: bool | None = None,
    teacher_hu_aux_distillation_weight: float | None = None,
    teacher_transfer_auxiliary_enabled: bool | None = None,
    teacher_transfer_auxiliary_weight: float | None = None,
    hu_current_policy_self_play: bool | None = None,
    **_unused_options,
) -> DeepCFRAgent:
    """Обучает один общий action-only агент external-sampling Deep CFR."""
    if (
        teacher_transfer_auxiliary_enabled is not None
        or teacher_transfer_auxiliary_weight is not None
    ):
        raise ValueError(
            "Устаревшие auxiliary-параметры teacher_transfer не поддерживаются; "
            "используйте teacher_hu_aux_distillation_enabled и "
            "teacher_hu_aux_distillation_weight"
        )
    unexpected_teacher_transfer_options = sorted(
        key for key in _unused_options if key.startswith("teacher_")
    )
    if unexpected_teacher_transfer_options:
        raise ValueError(
            "Неизвестные teacher-параметры: "
            + ", ".join(unexpected_teacher_transfer_options)
        )
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

    if teacher_strategy_checkpoint is None:
        teacher_strategy_checkpoint = cfg_get("teacher_strategy_checkpoint", None)
    hu_current_policy_enabled = (
        bool(cfg_get("hu_current_policy_self_play", False))
        if hu_current_policy_self_play is None
        else bool(hu_current_policy_self_play)
    )
    teacher_transfer = _teacher_transfer_configuration(
        enabled=teacher_transfer_enabled,
        mode=teacher_transfer_mode,
        checkpoint=teacher_transfer_checkpoint,
        freeze=teacher_transfer_freeze_card_encoder,
        auxiliary_enabled=teacher_hu_aux_distillation_enabled,
        auxiliary_weight=teacher_hu_aux_distillation_weight,
    )
    _validate_teacher_transfer_runtime(
        teacher_transfer,
        num_players=num_players,
        hu_current_policy_self_play=hu_current_policy_enabled,
        teacher_strategy_checkpoint=teacher_strategy_checkpoint,
    )
    agent = DeepCFRAgent(
        player_id=0,
        num_players=num_players,
        device=device,
        network_architecture=(
            CARD_CONTEXT_ARCHITECTURE
            if teacher_transfer["enabled"]
            else None
        ),
    )
    if trainable_players is not None:
        agent.num_trainable_players = max(1, min(int(trainable_players), agent.num_players))
    if hu_current_policy_enabled:
        HuCurrentPolicySelfPlayCoordinator.validate_runtime_configuration(
            enabled=True,
            num_players=agent.num_players,
            num_trainable_players=agent.num_trainable_players,
            opponent_checkpoint_dir=opponent_checkpoint_dir,
            teacher_strategy_checkpoint=teacher_strategy_checkpoint,
        )
        return _train_hu_current_policy_self_play(
            agent=agent,
            num_iterations=num_iterations,
            traversals_per_iteration=traversals_per_iteration,
            save_dir=save_dir,
            evaluate_every=evaluate_every,
            evaluation_games=evaluation_games,
            num_players=num_players,
            seed=seed,
            log_dir=log_dir,
            initial_checkpoint=initial_checkpoint,
        )
    if teacher_strategy_checkpoint:
        agent.load_teacher_strategy_checkpoint(teacher_strategy_checkpoint)
    opponent_state_cache: dict[Path, dict[str, torch.Tensor]] = {}
    if initial_checkpoint:
        checkpoint = agent.load_model(initial_checkpoint)
        if isinstance(checkpoint, dict) and isinstance(checkpoint.get("strategy_net"), dict):
            opponent_state_cache[Path(initial_checkpoint).resolve()] = checkpoint["strategy_net"]
        if opponent_checkpoint_dir is None and not _heavy_checkpoints(save_dir):
            save_dir = Path(initial_checkpoint).parent
    elif teacher_transfer["enabled"]:
        provenance = agent.load_card_encoder_from_hu_checkpoint(
            teacher_transfer["checkpoint"],
            freeze=bool(teacher_transfer["freeze"]),
        )
        agent.teacher_transfer_provenance = _teacher_transfer_provenance_payload(provenance)
    opponent_checkpoint_dir = Path(opponent_checkpoint_dir) if opponent_checkpoint_dir is not None else Path(save_dir)

    checkpoint_every = int(cfg_get("checkpoint_save_every", 1000))
    opponent_pool_schedule = OpponentPoolSchedule(
        opponent_checkpoint_dir,
        checkpoint_every=checkpoint_every,
        historical_every=int(cfg_get("checkpoint_keep_every", 50000)),
    )

    start_iteration = agent.iteration_count + 1
    writer = _create_writer(log_dir)
    try:
        print(
            f"Старт обучения: итераций={num_iterations}, "
            f"обходов/итерацию={traversals_per_iteration}, device={device}"
        )
        for iteration in range(start_iteration, start_iteration + int(num_iterations)):
            iteration_started = time.perf_counter()
            agent.iteration_count = iteration
            print(f"\nИтерация {iteration}:")
            opponent_setup_started = time.perf_counter()
            trainable_player_count = max(1, int(getattr(agent, "num_trainable_players", 1)))
            traversing_players = range(trainable_player_count)
            opponent_setup_elapsed = time.perf_counter() - opponent_setup_started
            traversal_started = time.perf_counter()
            with _traversal_thread_limit(bool(cfg_get("traversal_single_thread", True))):
                failed_traversals = 0
                agent.reset_traversal_stats()
                for traversing_player in traversing_players:
                    max_opponents = max(0, agent.num_players - 1)
                    strategy_count = min(
                        _current_strategy_opponent_count(iteration, checkpoint_every),
                        max_opponents,
                    )
                    transition_iterations = _transition_checkpoint_iterations(iteration, checkpoint_every)
                    if transition_iterations:
                        available_checkpoints = _heavy_checkpoints(opponent_checkpoint_dir)
                        selected_paths = [
                            _resolve_checkpoint_path(number, available_checkpoints)
                            for number in transition_iterations[:max(0, max_opponents - strategy_count)]
                        ]
                    else:
                        checkpoint_opponent_count = max_opponents - strategy_count
                        selected_paths = (
                            opponent_pool_schedule.paths_for_iteration(
                                iteration,
                                num_opponents=checkpoint_opponent_count,
                            )
                            if checkpoint_opponent_count > 0
                            else []
                        )
                    strategy_positions, checkpoint_by_player = _assign_checkpoint_opponents(
                        traversing_player=traversing_player,
                        num_players=agent.num_players,
                        strategy_count=strategy_count,
                        checkpoint_paths=selected_paths,
                    )
                    opponent_checkpoints = _configure_strategy_opponent_pool(
                        agent,
                        checkpoint_by_player=checkpoint_by_player,
                        strategy_positions=strategy_positions,
                        current_strategy_path=None,
                        traversing_player=traversing_player,
                        state_cache=opponent_state_cache,
                    )
                    if strategy_count:
                        _print_current_strategy_opponents(strategy_count)
                    if opponent_checkpoints:
                        _print_opponent_checkpoints(opponent_checkpoints, traversing_player)
                    agent.prepare_iteration(iteration, traversing_player=traversing_player)
                    if trainable_player_count == 1:
                        print(f"  Запускаю {traversals_per_iteration} обходов...")
                    else:
                        print(
                            f"  Игрок {traversing_player}: запускаю "
                            f"{traversals_per_iteration} обходов..."
                        )
                    for traversal in range(int(traversals_per_iteration)):
                        state_seed = (
                            (seed or 0)
                            + iteration * max(1, int(traversals_per_iteration)) * trainable_player_count
                            + traversing_player * max(1, int(traversals_per_iteration))
                            + traversal
                        )
                        try:
                            agent.record_traversal_attempt()
                            agent.cfr_traverse_multi(
                                _new_hand(num_players, state_seed),
                                iteration,
                                traversing_player=traversing_player,
                                random_agent=None,
                                traversal_index=traversal,
                            )
                            agent.record_traversal_success()
                        except TraversalFailure as error:
                            agent.record_traversal_failure(error)
                            if cfg_training_error_mode() != "skip_traversal":
                                raise
                            failed_traversals += 1
                            if failed_traversals >= cfg_training_max_failed_traversals_per_iteration():
                                raise RuntimeError(
                                    "Превышен training_max_failed_traversals_per_iteration"
                                ) from error
                    _log_multi_cfr_diagnostics(agent, writer, iteration, traversing_player, traversals_per_iteration)
            traversal_elapsed = time.perf_counter() - traversal_started

            training_threads = cfg_get("training_torch_threads")
            with _training_thread_limit(training_threads):
                advantage_loss = agent.train_advantage_network_multi()
                strategy_loss = agent.train_strategy_network()
            if writer is not None:
                writer.add_scalar("Loss/Advantage", advantage_loss, iteration)
                writer.add_scalar("Loss/Strategy", strategy_loss, iteration)
                writer.add_scalar("Time/Traversal", traversal_elapsed, iteration)
                writer.add_scalar("Time/OpponentPoolSetup", opponent_setup_elapsed, iteration)
                writer.add_scalar("Train/AdvantageLearningRate", agent.optimizer.param_groups[0]["lr"], iteration)
            if evaluate_every and iteration % int(evaluate_every) == 0:
                evaluation = evaluate_against_random(
                    agent, evaluation_games, num_players, return_stats=True
                )
                score = evaluation["mean_reward"]
                raise_frequency = evaluation["raise_frequency"]
                print(
                    f"  Оценка против random: reward={score:.4f}, "
                    f"raise_freq={raise_frequency:.3f}, игр={int(evaluation['games'])}"
                )
                if writer is not None:
                    writer.add_scalar("Evaluation/RandomMeanReward", score, iteration)
                    writer.add_scalar("Evaluation/RandomRaiseFrequency", raise_frequency, iteration)
            if _checkpoint_save_due(iteration, int(cfg_get("checkpoint_save_every", 1000))):
                last_checkpoint = _save_iteration_checkpoint(
                    agent, save_dir, iteration, seed=seed
                )
                light_checkpoint = _save_iteration_light_checkpoint(
                    agent, save_dir, iteration, seed=seed
                )
                print(f"  Checkpoint: {last_checkpoint}")
                print(f"  Light checkpoint: {light_checkpoint}")
            iteration_elapsed = time.perf_counter() - iteration_started
            if writer is not None:
                writer.add_scalar("Time/Iteration", iteration_elapsed, iteration)
            print(_format_iteration_summary(
                iteration_elapsed, traversal_elapsed, advantage_loss, strategy_loss, opponent_setup_elapsed
            ))
    finally:
        if writer is not None:
            writer.flush()
            writer.close()
    return agent


def train_deep_cfr(*args, **kwargs) -> DeepCFRAgent:
    return train_self_play_multi(*args, **kwargs)


def continue_training(checkpoint_path: str, additional_iterations: int = 1_000, **kwargs) -> DeepCFRAgent:
    return train_self_play_multi(
        num_iterations=additional_iterations,
        initial_checkpoint=checkpoint_path,
        **kwargs,
    )


def train_against_checkpoint(checkpoint_path: str, additional_iterations: int = 1_000, **kwargs) -> DeepCFRAgent:
    """Совместимое имя: старый checkpoint будет отвергнут строгой проверкой формата."""
    return continue_training(checkpoint_path, additional_iterations, **kwargs)


def train_with_mixed_checkpoints(*_args, **_kwargs):
    raise NotImplementedError("Смешивание старых sizing/Q checkpoint удалено; используйте six_fixed_v2.")


def _add_optional_boolean_flag(
    parser: argparse.ArgumentParser,
    option: str,
    destination: str,
    help_text: str,
) -> None:
    """Добавляет Python-3.8-совместимую пару --flag/--no-flag с default None."""
    if not option.startswith("--"):
        raise ValueError("CLI boolean option должен начинаться с --")
    flags = parser.add_mutually_exclusive_group()
    flags.add_argument(
        option,
        dest=destination,
        action="store_true",
        default=None,
        help=help_text,
    )
    flags.add_argument(
        "--no-" + option[2:],
        dest=destination,
        action="store_false",
        default=None,
        help=f"Отключить: {help_text}",
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Action-only Deep CFR")
    parser.add_argument("--self-play-multi", action="store_true", help="Запустить self-play обучение")
    parser.add_argument("--iterations", type=int, default=1_000, help="Число итераций CFR")
    parser.add_argument("--traversals", type=int, default=100, help="Обходов на итерацию")
    parser.add_argument("--save-dir", default="models", help="Каталог checkpoint-файлов")
    parser.add_argument("--log-dir", default=None, help="Каталог TensorBoard-логов")
    parser.add_argument(
        "--opponent-checkpoint-dir",
        default=None,
        help="Каталог full checkpoint-файлов для opponent pool",
    )
    parser.add_argument(
        "--trainable-players",
        type=int,
        default=int(cfg_get("num_trainable_players", 1)),
        help="Сколько мест собирать как traversing players за итерацию",
    )
    parser.add_argument(
        "--teacher-strategy-checkpoint",
        default=cfg_get("teacher_strategy_checkpoint", None),
        help="Checkpoint strategy-сети teacher-а для policy distillation",
    )
    _add_optional_boolean_flag(
        parser,
        "--teacher-transfer-enabled",
        "teacher_transfer_enabled",
        "Включить Stage A перенос card_encoder из HU checkpoint",
    )
    parser.add_argument(
        "--teacher-transfer-mode",
        default=None,
        help="Режим Stage A teacher transfer",
    )
    parser.add_argument(
        "--teacher-transfer-checkpoint",
        default=None,
        help="Путь к полному версионированному HU checkpoint для Stage A",
    )
    _add_optional_boolean_flag(
        parser,
        "--teacher-transfer-freeze-card-encoder",
        "teacher_transfer_freeze_card_encoder",
        "Заморозить перенесённый card_encoder",
    )
    _add_optional_boolean_flag(
        parser,
        "--teacher-hu-aux-distillation-enabled",
        "teacher_hu_aux_distillation_enabled",
        "Флаг Stage B; пока намеренно отклоняется",
    )
    parser.add_argument(
        "--teacher-hu-aux-distillation-weight",
        type=float,
        default=None,
        help="Вес Stage B; пока намеренно отклоняется",
    )
    parser.add_argument("--evaluate-every", type=int, default=10)
    parser.add_argument("--evaluation-games", type=int, default=500)
    parser.add_argument("--num-players", type=int, default=int(cfg_get("num_players", 6)))
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--initial-checkpoint", default=None)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.iterations <= 0 or args.traversals <= 0:
        raise ValueError("--iterations и --traversals должны быть положительными")
    if args.num_players < 2:
        raise ValueError("--num-players должен быть не меньше 2")
    _apply_process_priority(str(cfg_get("process_priority", "normal")))
    if not args.self_play_multi:
        print("Режим не указан; запускаю self-play. Для явности используйте --self-play-multi.")
    train_self_play_multi(
        num_iterations=args.iterations,
        traversals_per_iteration=args.traversals,
        save_dir=args.save_dir,
        log_dir=args.log_dir,
        evaluate_every=args.evaluate_every,
        evaluation_games=args.evaluation_games,
        num_players=args.num_players,
        device=args.device,
        seed=args.seed,
        initial_checkpoint=args.initial_checkpoint,
        opponent_checkpoint_dir=args.opponent_checkpoint_dir,
        trainable_players=args.trainable_players,
        teacher_strategy_checkpoint=args.teacher_strategy_checkpoint,
        teacher_transfer_enabled=args.teacher_transfer_enabled,
        teacher_transfer_mode=args.teacher_transfer_mode,
        teacher_transfer_checkpoint=args.teacher_transfer_checkpoint,
        teacher_transfer_freeze_card_encoder=args.teacher_transfer_freeze_card_encoder,
        teacher_hu_aux_distillation_enabled=args.teacher_hu_aux_distillation_enabled,
        teacher_hu_aux_distillation_weight=args.teacher_hu_aux_distillation_weight,
    )


if __name__ == "__main__":
    main()
