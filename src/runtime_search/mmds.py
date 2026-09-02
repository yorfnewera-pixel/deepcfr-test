"""Чистая реализация магнитного mirror-descent update."""
from __future__ import annotations

import numpy as np


def mmds_update(
    pi: np.ndarray,
    mc_values: np.ndarray,
    mask: np.ndarray,
    eta: float,
    alpha: float,
    rho: np.ndarray | None = None,
    floor: float = 1e-3,
) -> np.ndarray:
    """Обновляет root policy только на носителе допустимых действий."""
    pi_array = np.asarray(pi, dtype=np.float64)
    values_array = np.asarray(mc_values, dtype=np.float64)
    mask_array = np.asarray(mask, dtype=bool)
    if pi_array.ndim != 1 or values_array.ndim != 1 or mask_array.ndim != 1:
        raise ValueError("pi, mc_values и mask должны быть одномерными")
    if pi_array.shape != values_array.shape or pi_array.shape != mask_array.shape:
        raise ValueError("pi, mc_values и mask должны иметь одинаковую форму")
    if not np.all(np.isfinite(pi_array)) or not np.all(np.isfinite(values_array)):
        raise ValueError("pi и mc_values должны содержать только конечные значения")
    if np.any(pi_array < 0.0):
        raise ValueError("pi не может содержать отрицательные вероятности")
    if not np.isfinite(eta) or eta < 0.0:
        raise ValueError("eta должен быть неотрицательным конечным числом")
    if not np.isfinite(alpha) or alpha < 0.0:
        raise ValueError("alpha должен быть неотрицательным конечным числом")
    if not np.isfinite(floor) or floor < 0.0:
        raise ValueError("floor должен быть неотрицательным конечным числом")

    legal_indices = np.flatnonzero(mask_array)
    legal_count = int(legal_indices.size)
    if legal_count == 0:
        raise ValueError("Нужен хотя бы один допустимый action")
    if floor * legal_count > 1.0:
        raise ValueError("floor слишком велик для числа допустимых действий")

    pi_legal = _apply_floor(_normalize_on_support(pi_array[legal_indices], "pi"), floor)
    if rho is None:
        rho_legal = np.full(legal_count, 1.0 / legal_count, dtype=np.float64)
    else:
        rho_array = np.asarray(rho, dtype=np.float64)
        if rho_array.shape != pi_array.shape or not np.all(np.isfinite(rho_array)):
            raise ValueError("rho должна иметь форму pi и содержать конечные значения")
        if np.any(rho_array < 0.0):
            raise ValueError("rho не может содержать отрицательные вероятности")
        rho_legal = _apply_floor(_normalize_on_support(rho_array[legal_indices], "rho"), floor)

    log_weights = (
        np.log(pi_legal)
        + eta * values_array[legal_indices]
        + eta * alpha * np.log(rho_legal)
    ) / (1.0 + alpha * eta)
    log_weights -= np.max(log_weights)
    updated_legal = np.exp(log_weights)
    updated_legal /= updated_legal.sum()
    updated_legal = (1.0 - floor * legal_count) * updated_legal + floor

    result = np.zeros_like(pi_array, dtype=np.float64)
    result[legal_indices] = updated_legal
    return result


def _normalize_on_support(probabilities: np.ndarray, name: str) -> np.ndarray:
    total = float(probabilities.sum())
    if total <= 0.0:
        raise ValueError(f"{name} должна иметь положительную массу на допустимых действиях")
    return probabilities / total


def _apply_floor(probabilities: np.ndarray, floor: float) -> np.ndarray:
    if floor == 0.0 and np.any(probabilities <= 0.0):
        raise ValueError("Нулевые вероятности требуют положительного floor")
    return (1.0 - floor * probabilities.size) * probabilities + floor
