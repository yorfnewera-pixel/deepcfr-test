"""Загрузка минимальной конфигурации action-only Deep CFR."""
from __future__ import annotations

import os
from collections.abc import Mapping
from copy import deepcopy

import yaml

from src.core.action_space import NUM_ACTIONS
from src.core.model import (
    HISTORY_SUMMARY_V3_ENCODING_VERSION,
    LEGACY_ENCODING_VERSION,
    NETWORK_ARCHITECTURES,
)


_DEFAULTS = {
    "advantage_lr": 1e-4,
    "advantage_weight_decay": 1e-5,
    "strategy_lr": 5e-5,
    "strategy_weight_decay": 1e-5,
    "hidden_size": 256,
    "num_actions": NUM_ACTIONS,
    "num_players": 6,
    "hu_current_policy_self_play": False,
    "encoding_version": "history_summary_v3",
    "network_architecture": "monolithic_v1",
    "use_multi_agent_advantage": False,
    "advantage_memory_size": 300000,
    "strategy_memory_size": 300000,
    "advantage_batch_size": 256,
    "strategy_batch_size": 128,
    "advantage_epochs": 1,
    "strategy_epochs": 1,
    "advantage_train_steps": None,
    "strategy_train_steps": None,
    "advantage_buffer_reservoir": False,
    "strategy_buffer_reservoir": False,
    "clear_strategy_buffer_each_iteration": False,
    "save_replay_buffers_in_checkpoint": False,
    "discount_alpha": 2.0,
    "discount_gamma": 1.0,
    "advantage_accumulation": "dcfr_plus",
    "advantage_regret_norm": "none",
    "advantage_regret_clip": None,
    "advantage_reward_scale": 1.0,
    "advantage_loss": "mse",
    "advantage_huber_delta": 1.0,
    "d2cfr_enabled": False,
    "d2cfr_regret_loss_weight": 1.0,
    "d2cfr_state_value_loss_weight": 1.0,
    "d2cfr_action_value_loss_weight": 1.0,
    "d2cfr_reinitialize_each_iteration": True,
    "d2cfr_iteration_weight_power": 1.0,
    "d2cfr_mc_correction_enabled": False,
    "policy_runtime_min_action_prob": 0.0,
    "checkpoint_save_every": 1000,
    "hu_checkpoint_save_every": 5000,
    "checkpoint_keep_every": 50000,
    "traversal_single_thread": True,
    "training_torch_threads": None,
    "training_preload_to_device": False,
    "num_trainable_players": 1,
    "teacher_strategy_checkpoint": None,
    "teacher_transfer_enabled": False,
    "teacher_transfer_mode": "card_encoder_warmstart",
    "teacher_transfer_checkpoint": None,
    "teacher_transfer_freeze_card_encoder": False,
    "teacher_hu_aux_distillation_enabled": False,
    "teacher_hu_aux_distillation_weight": 0.0,
    "strategy_distillation_lambda": 0.0,
    "strategy_distillation_temperature": 1.0,
    "strategy_distillation_anneal_iterations": 0,
    "process_priority": "normal",
    "training_error_mode": "strict",
    "training_validate_state_invariants": False,
    "training_max_failed_traversals_per_iteration": 3,
}

_config = None
_raw_config = {}


def _validate_d2cfr_configuration(config: Mapping[str, object]) -> None:
    """Отклоняет ещё не реализованные либо математически несогласованные D2 режимы."""
    for key in (
        "d2cfr_enabled",
        "d2cfr_reinitialize_each_iteration",
        "d2cfr_mc_correction_enabled",
    ):
        if not isinstance(config[key], bool):
            raise ValueError(f"{key} должен быть bool")
    if not config["d2cfr_enabled"]:
        return
    if config["d2cfr_mc_correction_enabled"]:
        raise ValueError("D2CFR MC correction пока не реализован")
    if config["advantage_regret_clip"] is not None:
        raise ValueError("D2CFR несовместим с advantage_regret_clip")

    weights = []
    for key in (
        "d2cfr_regret_loss_weight",
        "d2cfr_state_value_loss_weight",
        "d2cfr_action_value_loss_weight",
    ):
        try:
            value = float(config[key])
        except (TypeError, ValueError) as error:
            raise ValueError(f"{key} должен быть числом") from error
        if value < 0.0 or not value < float("inf"):
            raise ValueError(f"{key} должен быть конечным числом >= 0")
        weights.append(value)
    if not any(weight > 0.0 for weight in weights):
        raise ValueError("Для D2CFR хотя бы один loss weight должен быть > 0")

    try:
        iteration_weight_power = float(config["d2cfr_iteration_weight_power"])
    except (TypeError, ValueError) as error:
        raise ValueError("d2cfr_iteration_weight_power должен быть числом") from error
    if iteration_weight_power < 0.0 or not iteration_weight_power < float("inf"):
        raise ValueError("d2cfr_iteration_weight_power должен быть конечным числом >= 0")


def _deep_merge(base, override):
    result = deepcopy(base)
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def load_config(path=None):
    global _config, _raw_config
    if path is None:
        root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        for candidate in ("config.yaml", os.path.join(root, "config.yaml")):
            if os.path.isfile(candidate):
                path = candidate
                break
    if path and os.path.isfile(path):
        with open(path, "r", encoding="utf-8") as source:
            loaded = yaml.safe_load(source) or {}
        if not isinstance(loaded, dict):
            raise ValueError("config.yaml должен содержать YAML-словарь")
        candidate_config = _deep_merge(_DEFAULTS, loaded)
        _validate_d2cfr_configuration(candidate_config)
        _raw_config = loaded.copy()
        _config = candidate_config
        if int(_config["num_actions"]) != NUM_ACTIONS:
            raise ValueError(f"num_actions должен быть равен {NUM_ACTIONS}")
        if bool(_config["hu_current_policy_self_play"]) and (
            int(_config["num_players"]) != 2
            or int(_config["num_trainable_players"]) != 2
        ):
            raise ValueError(
                "hu_current_policy_self_play требует num_players == num_trainable_players == 2"
            )
        if bool(_config["hu_current_policy_self_play"]):
            forbidden_sources = [
                key
                for key, value in _raw_config.items()
                if value is not None
                and (
                    key == "teacher_strategy_checkpoint"
                    or "opponent_pool" in key.lower()
                    or key.lower() in {"opponent_checkpoint_dir", "opponent_checkpoints"}
                )
            ]
            if forbidden_sources:
                raise ValueError(
                    "hu_current_policy_self_play несовместим с teacher_strategy_checkpoint "
                    "и внешним opponent pool: "
                    + ", ".join(sorted(forbidden_sources))
                )
        if _config["teacher_transfer_mode"] != "card_encoder_warmstart":
            raise ValueError(
                "teacher_transfer_mode поддерживает только card_encoder_warmstart"
            )
        legacy_auxiliary_keys = {
            "teacher_transfer_auxiliary_enabled",
            "teacher_transfer_auxiliary_weight",
        }
        if legacy_auxiliary_keys.intersection(_raw_config):
            raise ValueError(
                "Устаревшие auxiliary-настройки teacher_transfer не поддерживаются; "
                "используйте teacher_hu_aux_distillation_enabled и "
                "teacher_hu_aux_distillation_weight"
            )
        try:
            auxiliary_weight = float(_config["teacher_hu_aux_distillation_weight"])
        except (TypeError, ValueError) as error:
            raise ValueError(
                "teacher_hu_aux_distillation_weight должен быть числом"
            ) from error
        if bool(_config["teacher_hu_aux_distillation_enabled"]) or auxiliary_weight != 0.0:
            raise ValueError("Stage B auxiliary transfer пока не реализован")
        if bool(_config["teacher_transfer_enabled"]):
            if not isinstance(_config["teacher_transfer_checkpoint"], str) or not _config[
                "teacher_transfer_checkpoint"
            ].strip():
                raise ValueError(
                    "teacher_transfer_checkpoint обязателен при teacher_transfer_enabled"
                )
            if bool(_config["hu_current_policy_self_play"]):
                raise ValueError(
                    "teacher_transfer_enabled несовместим с hu_current_policy_self_play"
                )
            if int(_config["num_players"]) != 6:
                raise ValueError("teacher_transfer_enabled поддержан только для six-max")
            if _config["teacher_strategy_checkpoint"]:
                raise ValueError(
                    "teacher_transfer_enabled несовместим с teacher_strategy_checkpoint"
                )
        if _config["encoding_version"] not in (
            LEGACY_ENCODING_VERSION,
            HISTORY_SUMMARY_V3_ENCODING_VERSION,
        ):
            raise ValueError("encoding_version имеет неподдерживаемое значение")
        if _config["network_architecture"] not in NETWORK_ARCHITECTURES:
            raise ValueError("network_architecture имеет неподдерживаемое значение")
        training_error_mode = _config["training_error_mode"]
        if not isinstance(training_error_mode, str) or training_error_mode not in (
            "strict",
            "skip_traversal",
        ):
            raise ValueError("training_error_mode должен быть strict или skip_traversal")
        max_failed_traversals = _config["training_max_failed_traversals_per_iteration"]
        if (
            isinstance(max_failed_traversals, bool)
            or not isinstance(max_failed_traversals, int)
            or max_failed_traversals < 0
        ):
            raise ValueError("training_max_failed_traversals_per_iteration должен быть целым числом >= 0")
        if _config["training_validate_state_invariants"]:
            raise ValueError(
                "training_validate_state_invariants пока не поддерживается: "
                "проверка инвариантов состояния не реализована"
            )
        print(f"[Config] Загружен: {path}")
    else:
        _raw_config = {}
        _config = deepcopy(_DEFAULTS)
        print("[Config] config.yaml не найден, используются дефолты")


def cfg_get(key, default=None):
    global _config
    if _config is None:
        load_config()
    return _config.get(key, default if default is not None else _DEFAULTS.get(key))


def cfg_all():
    global _config
    if _config is None:
        load_config()
    return deepcopy(_config)


def cfg_has_raw(key):
    global _config
    if _config is None:
        load_config()
    return key in _raw_config


def cfg_reservoir_flag(key):
    if key != "advantage_buffer_reservoir":
        raise KeyError(f"Неизвестный action-only replay buffer: {key}")
    return bool(cfg_get(key))


def cfg_clear_strategy_buffer_each_iteration():
    if cfg_has_raw("clear_strategy_buffer_each_iteration"):
        return bool(cfg_get("clear_strategy_buffer_each_iteration"))
    if cfg_has_raw("strategy_buffer_reservoir"):
        return not bool(cfg_get("strategy_buffer_reservoir"))
    return bool(cfg_get("clear_strategy_buffer_each_iteration"))


def cfg_training_error_mode() -> str:
    return str(cfg_get("training_error_mode"))


def cfg_training_validate_state_invariants() -> bool:
    return bool(cfg_get("training_validate_state_invariants"))


def cfg_training_max_failed_traversals_per_iteration() -> int:
    return int(cfg_get("training_max_failed_traversals_per_iteration"))
