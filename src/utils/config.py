"""Загрузка минимальной конфигурации action-only Deep CFR."""
from __future__ import annotations

import os
from copy import deepcopy

import yaml

from src.core.action_space import NUM_ACTIONS
from src.core.model import (
    HISTORY_SUMMARY_V3_ENCODING_VERSION,
    LEGACY_ENCODING_VERSION,
)


_DEFAULTS = {
    "advantage_lr": 1e-4,
    "advantage_weight_decay": 1e-5,
    "strategy_lr": 5e-5,
    "strategy_weight_decay": 1e-5,
    "hidden_size": 256,
    "num_actions": NUM_ACTIONS,
    "num_players": 6,
    "encoding_version": "history_summary_v3",
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
    "policy_runtime_min_action_prob": 0.0,
    "checkpoint_save_every": 1000,
    "checkpoint_keep_every": 50000,
    "traversal_single_thread": True,
    "training_torch_threads": None,
    "training_preload_to_device": False,
    "num_trainable_players": 1,
    "teacher_strategy_checkpoint": None,
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
        _raw_config = loaded.copy()
        _config = _deep_merge(_DEFAULTS, loaded)
        if int(_config["num_actions"]) != NUM_ACTIONS:
            raise ValueError(f"num_actions должен быть равен {NUM_ACTIONS}")
        if _config["encoding_version"] not in (
            LEGACY_ENCODING_VERSION,
            HISTORY_SUMMARY_V3_ENCODING_VERSION,
        ):
            raise ValueError("encoding_version имеет неподдерживаемое значение")
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
