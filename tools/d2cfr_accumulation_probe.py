"""Диагностика накопления D2CFR-сигнала по нескольким итерациям.

Утилита фиксирует небольшой набор samples из исторического D2CFR replay-buffer
и после каждой итерации обучения проверяет, насколько текущая advantage-сеть
восстанавливает их Q/V/R и regret-matching policy.
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.core.action_space import ACTION_LABELS, NUM_ACTIONS
from tools import d2cfr_signal_probe as signal_probe


__all__ = [
    "regret_matching_policy",
    "samples_from_dueling_buffer_for_iteration",
    "run_heldout_fit_probe",
    "evaluate_prediction_snapshot",
    "describe_buffer_age",
    "run_checkpoint_buffer_age_probe",
    "run_weight_mode_fit_ab",
    "run_replay_freshness_fit_ab",
    "run_checkpoint_weight_mode_ab_probe",
    "run_checkpoint_replay_freshness_probe",
    "describe_prediction_rows",
    "run_accumulation_probe",
    "run_live_accumulation_probe",
]


def _safe_float(value: Any) -> float:
    return float(torch.as_tensor(value).detach().cpu().item())


def _mean_abs(values: torch.Tensor, mask: torch.Tensor | None = None) -> float:
    if mask is None:
        return float(values.abs().mean().detach().cpu().item())
    weighted = values.abs() * mask
    denominator = max(float(mask.sum().detach().cpu().item()), 1.0)
    return float(weighted.sum().detach().cpu().item() / denominator)


def _max_abs(values: torch.Tensor, mask: torch.Tensor | None = None) -> float:
    if mask is not None:
        values = values * mask
    if values.numel() == 0:
        return 0.0
    return float(values.abs().max().detach().cpu().item())


def regret_matching_policy(regrets: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Строит RM-policy только по legal слотам; при отсутствии плюса даёт uniform."""
    legal_mask = mask.to(dtype=torch.float32)
    positive = torch.clamp(regrets.to(dtype=torch.float32), min=0.0) * legal_mask
    positive_sum = positive.sum()
    if float(positive_sum.detach().cpu().item()) > 0.0:
        return positive / positive_sum
    legal_count = legal_mask.sum()
    if float(legal_count.detach().cpu().item()) <= 0.0:
        return torch.zeros_like(legal_mask)
    return legal_mask / legal_count


def _batch_rm_policy(regrets: torch.Tensor, masks: torch.Tensor) -> torch.Tensor:
    rows = [
        regret_matching_policy(regret_row, mask_row)
        for regret_row, mask_row in zip(regrets, masks, strict=True)
    ]
    return torch.stack(rows, dim=0)


def _sign_margin_report(
    target_regrets: torch.Tensor,
    predicted_regrets: torch.Tensor,
    masks: torch.Tensor,
) -> dict[str, Any]:
    target = target_regrets.detach()
    predicted = predicted_regrets.detach()
    legal = masks.detach() > 0.0
    target_positive = target > 0.0
    predicted_positive = predicted > 0.0
    sign_flips = (target_positive != predicted_positive) & legal
    abs_target = target.abs()

    buckets = {
        "abs_le_0.01": legal & (abs_target <= 0.01),
        "abs_0.01_0.1": legal & (abs_target > 0.01) & (abs_target <= 0.1),
        "abs_gt_0.1": legal & (abs_target > 0.1),
        "overall": legal,
    }
    report: dict[str, Any] = {}
    for name, bucket_mask in buckets.items():
        count = int(bucket_mask.sum().detach().cpu().item())
        flips = int((sign_flips & bucket_mask).sum().detach().cpu().item())
        mean_abs_error = (
            float(((predicted - target).abs() * bucket_mask).sum().detach().cpu().item() / count)
            if count > 0
            else 0.0
        )
        report[name] = {
            "legal_slots": count,
            "sign_flips": flips,
            "sign_flip_rate": float(flips / count) if count > 0 else 0.0,
            "mean_abs_error": mean_abs_error,
        }
    return report


def evaluate_prediction_snapshot(network: Any, batch: signal_probe.ProbeBatch) -> dict[str, Any]:
    """Сравнивает prediction сети с фиксированным диагностическим batch."""
    was_training = bool(getattr(network, "training", False))
    network.eval()
    with torch.no_grad():
        components = network.forward_components(batch.states)
        predicted_q = components.action_values
        predicted_v = components.state_values.squeeze(1)
        predicted_regrets = components.regrets
        masks = batch.masks

        q_error = predicted_q - batch.action_targets
        v_error = predicted_v - batch.state_value_targets
        regret_error = predicted_regrets - batch.regret_targets
        regret_mse = ((regret_error.square() * masks).sum() / torch.clamp(masks.sum(), min=1.0)).detach()
        state_value_mse = v_error.square().mean().detach()
        target_rm = _batch_rm_policy(batch.regret_targets, masks)
        predicted_rm = _batch_rm_policy(predicted_regrets, masks)
        rm_l1 = ((target_rm - predicted_rm).abs() * masks).sum(dim=1)

        report = {
            "samples": int(batch.states.shape[0]),
            "legal_slots": int(masks.sum().detach().cpu().item()),
            "q_abs_error_mean": _mean_abs(q_error, masks),
            "q_abs_error_max": _max_abs(q_error, masks),
            "state_value_abs_error_mean": _mean_abs(v_error),
            "state_value_abs_error_max": _max_abs(v_error),
            "regret_abs_error_mean": _mean_abs(regret_error, masks),
            "regret_abs_error_max": _max_abs(regret_error, masks),
            "regret_mse": float(regret_mse.cpu().item()),
            "state_value_mse": float(state_value_mse.cpu().item()),
            "eval_loss_mse": float((regret_mse + 0.5 * state_value_mse).cpu().item()),
            "rm_policy_l1_mean": float(rm_l1.mean().detach().cpu().item()),
            "rm_policy_l1_max": float(rm_l1.max().detach().cpu().item()),
            "sign_margin": _sign_margin_report(batch.regret_targets, predicted_regrets, masks),
            "target_regret_abs_mean": _mean_abs(batch.regret_targets, masks),
            "predicted_regret_abs_mean": _mean_abs(predicted_regrets, masks),
            "target_state_value_abs_mean": _mean_abs(batch.state_value_targets),
            "predicted_state_value_abs_mean": _mean_abs(predicted_v),
        }
    if was_training:
        network.train()
    return report


def _float_list(values: torch.Tensor | np.ndarray | Iterable[float]) -> list[float]:
    array = torch.as_tensor(values).detach().cpu().to(dtype=torch.float32).reshape(-1)
    return [float(item) for item in array.tolist()]


def _legal_slots(mask: torch.Tensor) -> list[int]:
    legal = torch.nonzero(mask.detach().cpu() > 0.0, as_tuple=False).reshape(-1)
    return [int(item) for item in legal.tolist()]


def _top_abs_features(row: torch.Tensor, limit: int) -> list[dict[str, float | int]]:
    values = row.detach().cpu().to(dtype=torch.float32)
    nonzero = torch.nonzero(values != 0.0, as_tuple=False).reshape(-1)
    if nonzero.numel() == 0:
        return []
    order = torch.argsort(values[nonzero].abs(), descending=True)[: max(int(limit), 0)]
    return [
        {"index": int(nonzero[item].item()), "value": float(values[nonzero[item]].item())}
        for item in order
    ]


def describe_prediction_rows(
    network: Any,
    batch: signal_probe.ProbeBatch,
    *,
    sample_limit: int = 3,
    max_features: int = 16,
) -> dict[str, Any]:
    """Даёт компактный white-box срез target vs prediction по первым samples."""
    was_training = bool(getattr(network, "training", False))
    network.eval()
    with torch.no_grad():
        components = network.forward_components(batch.states)
        predicted_q = components.action_values
        predicted_v = components.state_values.squeeze(1)
        predicted_regrets = components.regrets
        target_rm = _batch_rm_policy(batch.regret_targets, batch.masks)
        predicted_rm = _batch_rm_policy(predicted_regrets, batch.masks)

        rows = []
        for index in range(min(int(sample_limit), int(batch.states.shape[0]))):
            legal = _legal_slots(batch.masks[index])
            rows.append(
                {
                    "sample_index": int(index),
                    "iteration": _safe_float(batch.iterations[index]),
                    "legal_slots": legal,
                    "input": {
                        "nonzero_count": int((batch.states[index] != 0.0).sum().detach().cpu().item()),
                        "top_abs_features": _top_abs_features(batch.states[index], max_features),
                    },
                    "target": {
                        "state_value": _safe_float(batch.state_value_targets[index]),
                        "q": _float_list(batch.action_targets[index]),
                        "regrets": _float_list(batch.regret_targets[index]),
                        "rm_policy": _float_list(target_rm[index]),
                    },
                    "predicted": {
                        "state_value": _safe_float(predicted_v[index]),
                        "q": _float_list(predicted_q[index]),
                        "regrets": _float_list(predicted_regrets[index]),
                        "rm_policy": _float_list(predicted_rm[index]),
                    },
                }
            )
    if was_training:
        network.train()
    return {
        "input_size": int(batch.states.shape[1]),
        "samples_returned": len(rows),
        "samples": rows,
    }


def _copy_diagnostic_samples(
    buffer: Any,
    *,
    sample_count: int,
    seed: int,
    device: str | torch.device,
) -> signal_probe.ProbeBatch:
    samples = signal_probe.samples_from_dueling_buffer(
        buffer,
        sample_count=sample_count,
        seed=seed,
    )
    return signal_probe.build_probe_batch(samples, device=device)


def _slice_probe_batch(batch: signal_probe.ProbeBatch, indices: torch.Tensor) -> signal_probe.ProbeBatch:
    return signal_probe.ProbeBatch(
        states=batch.states.index_select(0, indices),
        action_targets=batch.action_targets.index_select(0, indices),
        state_value_targets=batch.state_value_targets.index_select(0, indices),
        regret_targets=batch.regret_targets.index_select(0, indices),
        masks=batch.masks.index_select(0, indices),
        iterations=batch.iterations.index_select(0, indices),
    )


def _weighted_d2cfr_loss(
    network: Any,
    batch: signal_probe.ProbeBatch,
    weights: torch.Tensor,
    *,
    state_value_loss_weight: float,
) -> torch.Tensor:
    components = network.forward_components(batch.states)
    regret_error = (components.regrets - batch.regret_targets).square() * batch.masks
    regret_loss = (regret_error.sum(dim=1) * weights).mean()
    state_error = (components.state_values.squeeze(1) - batch.state_value_targets).square()
    state_loss = (state_error * weights).mean()
    return regret_loss + float(state_value_loss_weight) * state_loss


def _fit_d2cfr_arm(
    *,
    name: str,
    network_factory: Any,
    train_batch: signal_probe.ProbeBatch,
    evaluation_batch: signal_probe.ProbeBatch,
    index_batches: list[torch.Tensor],
    learning_rate: float,
    state_value_loss_weight: float,
    seed: int,
    max_grad_norm: float | None,
    iteration_weighted: bool,
) -> dict[str, Any]:
    """Обучает одну изолированную сеть на заранее выбранных minibatch-индексах."""
    device = train_batch.states.device
    source_iterations = train_batch.iterations.to(dtype=torch.float32)
    torch.manual_seed(int(seed))
    network = network_factory().to(device)
    optimizer = torch.optim.Adam(network.parameters(), lr=float(learning_rate))
    initial = evaluate_prediction_snapshot(network, evaluation_batch)
    initial_by_action = _action_prediction_metrics(network, evaluation_batch)
    batch_weight_means: list[float] = []
    network.train()
    for indices in index_batches:
        batch = _slice_probe_batch(train_batch, indices)
        if iteration_weighted:
            batch_weights = batch.iterations / batch.iterations.mean().clamp_min(1e-12)
        else:
            batch_weights = torch.ones_like(batch.iterations)
        batch_weight_means.append(float(batch_weights.mean().detach().cpu().item()))
        optimizer.zero_grad(set_to_none=True)
        loss = _weighted_d2cfr_loss(
            network,
            batch,
            batch_weights,
            state_value_loss_weight=state_value_loss_weight,
        )
        loss.backward()
        if max_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(network.parameters(), max_norm=float(max_grad_norm))
        optimizer.step()
    return {
        "name": name,
        "train_samples": int(train_batch.states.shape[0]),
        "source_iteration_minimum": float(source_iterations.min().detach().cpu().item()),
        "source_iteration_maximum": float(source_iterations.max().detach().cpu().item()),
        "weight_summary": {
            "normalization": "per_minibatch_mean_1" if iteration_weighted else "uniform",
            "batch_mean_minimum": min(batch_weight_means, default=1.0),
            "batch_mean_maximum": max(batch_weight_means, default=1.0),
        },
        "initial": initial,
        "initial_by_action": initial_by_action,
        "final": evaluate_prediction_snapshot(network, evaluation_batch),
        "final_by_action": _action_prediction_metrics(network, evaluation_batch),
    }


def _random_index_batches(
    *, sample_count: int, batch_size: int, steps: int, seed: int, device: torch.device
) -> list[torch.Tensor]:
    draw_count = min(int(batch_size), int(sample_count))
    rng = np.random.default_rng(seed)
    return [
        torch.as_tensor(rng.integers(0, sample_count, size=draw_count), dtype=torch.long, device=device)
        for _ in range(int(steps))
    ]


def run_weight_mode_fit_ab(
    *,
    network_factory: Any,
    train_batch: signal_probe.ProbeBatch,
    evaluation_batch: signal_probe.ProbeBatch,
    steps: int,
    batch_size: int,
    learning_rate: float,
    state_value_loss_weight: float,
    seed: int,
    max_grad_norm: float | None = 1.0,
) -> dict[str, Any]:
    """Сравнивает одинаковый offline fit с временными и равномерными весами."""
    sample_count = int(train_batch.states.shape[0])
    if sample_count <= 0 or int(evaluation_batch.states.shape[0]) <= 0:
        raise ValueError("Train и evaluation batch не должны быть пустыми")
    if steps < 0 or batch_size < 1 or learning_rate <= 0.0:
        raise ValueError("steps, batch_size и learning_rate должны быть положительными")
    if max_grad_norm is not None and max_grad_norm <= 0.0:
        raise ValueError("max_grad_norm должен быть положительным или None")

    device = train_batch.states.device
    draw_count = min(int(batch_size), sample_count)
    index_batches = _random_index_batches(
        sample_count=sample_count, batch_size=draw_count, steps=int(steps), seed=int(seed), device=device,
    )

    return {
        "train_samples": sample_count,
        "evaluation_samples": int(evaluation_batch.states.shape[0]),
        "steps": int(steps),
        "batch_size": draw_count,
        "learning_rate": float(learning_rate),
        "state_value_loss_weight": float(state_value_loss_weight),
        "max_grad_norm": max_grad_norm,
        "arms": {
            "iteration_weighted": _fit_d2cfr_arm(
                name="iteration_weighted", network_factory=network_factory, train_batch=train_batch,
                evaluation_batch=evaluation_batch, index_batches=index_batches, learning_rate=learning_rate,
                state_value_loss_weight=state_value_loss_weight, seed=seed, max_grad_norm=max_grad_norm,
                iteration_weighted=True,
            ),
            "uniform": _fit_d2cfr_arm(
                name="uniform", network_factory=network_factory, train_batch=train_batch,
                evaluation_batch=evaluation_batch, index_batches=index_batches, learning_rate=learning_rate,
                state_value_loss_weight=state_value_loss_weight, seed=seed, max_grad_norm=max_grad_norm,
                iteration_weighted=False,
            ),
        },
    }


def run_replay_freshness_fit_ab(
    *,
    network_factory: Any,
    full_batch: signal_probe.ProbeBatch,
    recent_batch: signal_probe.ProbeBatch,
    evaluation_batch: signal_probe.ProbeBatch,
    steps: int,
    batch_size: int,
    learning_rate: float,
    state_value_loss_weight: float,
    seed: int,
    max_grad_norm: float | None = 1.0,
) -> dict[str, Any]:
    """Сравнивает full replay с recent window при неизменной D2CFR loss-функции."""
    if min(int(full_batch.states.shape[0]), int(recent_batch.states.shape[0]), int(evaluation_batch.states.shape[0])) < 1:
        raise ValueError("Full, recent и evaluation batch не должны быть пустыми")
    if steps < 0 or batch_size < 1 or learning_rate <= 0.0:
        raise ValueError("steps, batch_size и learning_rate должны быть положительными")
    device = full_batch.states.device
    if recent_batch.states.device != device or evaluation_batch.states.device != device:
        raise ValueError("Все batch должны быть на одном device")
    return {
        "steps": int(steps),
        "batch_size": int(batch_size),
        "learning_rate": float(learning_rate),
        "state_value_loss_weight": float(state_value_loss_weight),
        "max_grad_norm": max_grad_norm,
        "arms": {
            "full_replay": _fit_d2cfr_arm(
                name="full_replay", network_factory=network_factory, train_batch=full_batch,
                evaluation_batch=evaluation_batch,
                index_batches=_random_index_batches(
                    sample_count=int(full_batch.states.shape[0]), batch_size=batch_size,
                    steps=steps, seed=seed, device=device,
                ),
                learning_rate=learning_rate, state_value_loss_weight=state_value_loss_weight,
                seed=seed, max_grad_norm=max_grad_norm, iteration_weighted=True,
            ),
            "recent_window": _fit_d2cfr_arm(
                name="recent_window", network_factory=network_factory, train_batch=recent_batch,
                evaluation_batch=evaluation_batch,
                index_batches=_random_index_batches(
                    sample_count=int(recent_batch.states.shape[0]), batch_size=batch_size,
                    steps=steps, seed=seed, device=device,
                ),
                learning_rate=learning_rate, state_value_loss_weight=state_value_loss_weight,
                seed=seed, max_grad_norm=max_grad_norm, iteration_weighted=True,
            ),
        },
    }


def _action_prediction_metrics(network: Any, batch: signal_probe.ProbeBatch) -> dict[str, dict[str, Any]]:
    """Возвращает Q/R-ошибки по action slot без смешивания legal масок."""
    was_training = bool(getattr(network, "training", False))
    network.eval()
    with torch.no_grad():
        components = network.forward_components(batch.states)
        q_error = (components.action_values - batch.action_targets).abs()
        regret_error = (components.regrets - batch.regret_targets).abs()
        target_positive = batch.regret_targets > 0.0
        predicted_positive = components.regrets > 0.0
        report: dict[str, dict[str, Any]] = {}
        for slot, label in enumerate(ACTION_LABELS):
            legal = batch.masks[:, slot] > 0.0
            count = int(legal.sum().detach().cpu().item())
            if count == 0:
                continue
            report[label] = {
                "legal_slots": count,
                "q_abs_error_mean": float(q_error[legal, slot].mean().detach().cpu().item()),
                "regret_abs_error_mean": float(regret_error[legal, slot].mean().detach().cpu().item()),
                "regret_sign_flip_rate": float(
                    (target_positive[legal, slot] != predicted_positive[legal, slot])
                    .to(dtype=torch.float32).mean().detach().cpu().item()
                ),
                "sign_margin": _sign_margin_report(
                    batch.regret_targets[:, slot : slot + 1],
                    components.regrets[:, slot : slot + 1],
                    batch.masks[:, slot : slot + 1],
                ),
            }
    if was_training:
        network.train()
    return report


def describe_buffer_age(
    network: Any,
    buffer: Any,
    *,
    checkpoint_iteration: int,
    age_bucket_width: int = 25,
    sample_limit_per_bucket: int | None = None,
    seed: int = 0,
    device: str | torch.device = "cpu",
) -> dict[str, Any]:
    """Сравнивает текущую сеть со stored labels, разложенными по возрасту replay."""
    count = len(buffer)
    if count <= 0:
        raise ValueError("D2CFR replay buffer пуст")
    if checkpoint_iteration < 1 or age_bucket_width < 1:
        raise ValueError("checkpoint_iteration и age_bucket_width должны быть положительными")
    if sample_limit_per_bucket is not None and sample_limit_per_bucket < 1:
        raise ValueError("sample_limit_per_bucket должен быть положительным или None")
    required_fields = ("_states", "_action_values", "_state_values", "_regrets", "_masks", "_iterations")
    if not all(hasattr(buffer, field) for field in required_fields):
        raise TypeError("Нужен DuelingAdvantageBuffer с Q/V/R samples")

    source_iterations = np.asarray(buffer._iterations[:count], dtype=np.int64)
    ages = int(checkpoint_iteration) - source_iterations
    if np.any(ages < 0):
        raise ValueError("Replay содержит samples из будущей iteration")
    rng = np.random.default_rng(seed)
    groups: dict[str, np.ndarray] = {}
    for lower_bound in np.unique((ages // int(age_bucket_width)) * int(age_bucket_width)):
        upper_bound = int(lower_bound) + int(age_bucket_width) - 1
        groups[f"age_{int(lower_bound)}_{upper_bound}"] = np.flatnonzero(
            (ages >= int(lower_bound)) & (ages <= upper_bound)
        )

    by_age: dict[str, dict[str, Any]] = {}
    for label, indices in sorted(groups.items(), key=lambda item: int(item[0].split("_")[1])):
        if sample_limit_per_bucket is not None and indices.size > sample_limit_per_bucket:
            indices = np.sort(rng.choice(indices, size=sample_limit_per_bucket, replace=False))
        samples = [
            (
                buffer._states[index].copy(),
                buffer._action_values[index].copy(),
                np.float32(buffer._state_values[index]),
                buffer._regrets[index].copy(),
                buffer._masks[index].copy(),
                np.float32(buffer._iterations[index]),
            )
            for index in indices
        ]
        batch = signal_probe.build_probe_batch(samples, device=device)
        bucket_ages = ages[indices]
        bucket_iterations = source_iterations[indices]
        by_age[label] = {
            "samples": int(indices.size),
            "source_iteration_min": int(bucket_iterations.min()),
            "source_iteration_max": int(bucket_iterations.max()),
            "age_min": int(bucket_ages.min()),
            "age_max": int(bucket_ages.max()),
            "prediction": evaluate_prediction_snapshot(network, batch),
            "by_action": _action_prediction_metrics(network, batch),
        }
    return {
        "checkpoint_iteration": int(checkpoint_iteration),
        "age_bucket_width": int(age_bucket_width),
        "sample_limit_per_bucket": sample_limit_per_bucket,
        "buffer_samples": int(count),
        "by_age": by_age,
    }


def run_checkpoint_buffer_age_probe(
    checkpoint_path: str | Path,
    *,
    config_path: str | Path,
    output_path: str | Path,
    player_id: int = 0,
    age_bucket_width: int = 25,
    sample_limit_per_bucket: int | None = 2048,
    seed: int = 0,
    device: str | torch.device = "cpu",
) -> dict[str, Any]:
    """Загружает HU checkpoint и сохраняет read-only age-срез его D2CFR buffer."""
    from src.core.deep_cfr import DeepCFRAgent
    from src.training import train as train_mod
    from src.utils import config as config_mod

    config_mod.load_config(config_path)
    agent = DeepCFRAgent(player_id=0, num_players=2, device=str(device))
    train_mod._create_hu_current_policy_coordinator(agent)
    checkpoint = train_mod._load_hu_checkpoint(agent, checkpoint_path)
    if not agent.d2cfr_enabled:
        raise ValueError("Нужен D2CFR HU checkpoint")
    actor_id = int(player_id)
    if actor_id not in (0, 1):
        raise ValueError("player_id должен быть 0 или 1 для HU")
    report = describe_buffer_age(
        agent.hu_advantage_nets[actor_id],
        agent.hu_advantage_buffers[actor_id],
        checkpoint_iteration=int(checkpoint["iteration"]),
        age_bucket_width=age_bucket_width,
        sample_limit_per_bucket=sample_limit_per_bucket,
        seed=seed,
        device=device,
    )
    report.update({
        "schema_version": 1,
        "checkpoint": str(checkpoint_path),
        "config": str(config_path),
        "player_id": actor_id,
        "read_only": True,
    })
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(target)
    return report


def _samples_from_dueling_buffer_window(
    buffer: Any, *, minimum_iteration: int, sample_count: int, seed: int
) -> list[tuple]:
    """Копирует случайный срез replay, не старше заданной CFR-итерации."""
    size = len(buffer)
    if size <= 0:
        raise ValueError("D2CFR advantage_buffer пуст")
    required_arrays = ("_states", "_action_values", "_state_values", "_regrets", "_masks", "_iterations")
    try:
        arrays = [getattr(buffer, name) for name in required_arrays]
    except AttributeError as error:
        raise ValueError("live buffer не похож на DuelingAdvantageBuffer") from error
    matching = np.flatnonzero(np.asarray(arrays[5][:size], dtype=np.int64) >= int(minimum_iteration))
    if matching.size <= 0:
        raise ValueError(f"D2CFR buffer не содержит samples с iteration >= {minimum_iteration}")
    take = min(int(sample_count), int(matching.size))
    indices = np.random.default_rng(seed).choice(matching, size=take, replace=False)
    return [
        (
            arrays[0][index].copy(), arrays[1][index].copy(), np.float32(arrays[2][index]),
            arrays[3][index].copy(), arrays[4][index].copy(), np.float32(arrays[5][index]),
        )
        for index in indices
    ]


def _collect_frozen_postflop_evaluation_batch(
    agent: Any,
    coordinator: Any,
    *,
    player_id: int,
    iteration: int,
    states: int,
    repeats: int,
    roots: int,
    seed: int,
    device: str | torch.device,
) -> tuple[signal_probe.ProbeBatch, int]:
    """Собирает усреднённые ES-target'ы для общего postflop evaluation набора."""
    from src.core.action_space import legal_action_mask
    from tools import d2cfr_allin_bias_probe as allin_probe

    captured = allin_probe._collect_candidate_states(
        coordinator, roots=roots, seed=seed, target_count=states, traverser=player_id, iteration=iteration,
    )
    if len(captured) != states:
        raise RuntimeError(f"Собрано только {len(captured)} из {states} postflop-состояний; увеличьте --frozen-roots")
    samples: list[tuple] = []
    for state in captured:
        target_samples = allin_probe._sample_state_targets(
            coordinator, state, repeats=repeats, traverser=player_id, iteration=iteration,
        )
        samples.append((
            agent._encode_state(state, player_id),
            np.stack([sample[0] for sample in target_samples], axis=0).mean(axis=0).astype(np.float32),
            np.float32(np.mean([sample[1] for sample in target_samples])),
            np.stack([sample[2] for sample in target_samples], axis=0).mean(axis=0).astype(np.float32),
            legal_action_mask(state).astype(np.float32), np.float32(iteration),
        ))
    return signal_probe.build_probe_batch(samples, device=device), len(captured)


def run_checkpoint_weight_mode_ab_probe(
    checkpoint_path: str | Path,
    *,
    config_path: str | Path,
    output_path: str | Path,
    player_id: int = 0,
    fit_samples: int = 20_000,
    fit_steps: int = 2_000,
    fit_batch_size: int = 256,
    frozen_states: int = 64,
    frozen_repeats: int = 64,
    frozen_roots: int = 4_096,
    seed: int = 0,
    device: str | torch.device = "cpu",
) -> dict[str, Any]:
    """Проверяет вклад iteration weights без изменения checkpoint или его replay."""
    from src.core.deep_cfr import DeepCFRAgent
    from src.training import train as train_mod
    from src.utils import config as config_mod

    actor_id = int(player_id)
    if actor_id not in (0, 1):
        raise ValueError("player_id должен быть 0 или 1 для HU")
    if min(fit_samples, fit_steps, fit_batch_size, frozen_states, frozen_repeats, frozen_roots) < 1:
        raise ValueError("Все размеры A/B-проверки должны быть положительными")

    config_mod.load_config(config_path)
    agent = DeepCFRAgent(player_id=0, num_players=2, device=str(device))
    coordinator = train_mod._create_hu_current_policy_coordinator(agent)
    checkpoint = train_mod._load_hu_checkpoint(agent, checkpoint_path)
    if not agent.d2cfr_enabled:
        raise ValueError("Нужен D2CFR HU checkpoint")

    buffer_size_before_targets = int(len(agent.hu_advantage_buffers[actor_id]))
    training_samples = signal_probe.samples_from_dueling_buffer(
        agent.hu_advantage_buffers[actor_id], sample_count=int(fit_samples), seed=int(seed),
    )
    training_batch = signal_probe.build_probe_batch(training_samples, device=device)

    evaluation_batch, states_collected = _collect_frozen_postflop_evaluation_batch(
        agent, coordinator, player_id=actor_id, iteration=int(checkpoint["iteration"]),
        states=int(frozen_states), repeats=int(frozen_repeats), roots=int(frozen_roots), seed=int(seed), device=device,
    )
    comparison = run_weight_mode_fit_ab(
        network_factory=agent._new_advantage_network,
        train_batch=training_batch,
        evaluation_batch=evaluation_batch,
        steps=int(fit_steps),
        batch_size=int(fit_batch_size),
        learning_rate=float(agent.advantage_lr),
        state_value_loss_weight=float(agent.d2cfr_state_value_loss_weight),
        seed=int(seed),
        max_grad_norm=1.0 if agent.d2cfr_loss_mode == "anchored" else None,
    )
    report = {
        "schema_version": 1,
        "read_only": True,
        "checkpoint": str(checkpoint_path),
        "checkpoint_iteration": int(checkpoint["iteration"]),
        "config": str(config_path),
        "player_id": actor_id,
        "seed": int(seed),
        "training_buffer": {
            "available_samples_before_targets": buffer_size_before_targets,
            "sampled_without_replacement": int(len(training_samples)),
        },
        "frozen_postflop": {
            "states_requested": int(frozen_states),
            "states_collected": states_collected,
            "repeats_per_state": int(frozen_repeats),
            "roots_examined": int(frozen_roots),
            "targets": "mean of repeated frozen-policy ES traversals",
        },
        "fit": {
            "loss_function": str(agent.d2cfr_loss_function),
            "loss_mode": str(agent.d2cfr_loss_mode),
            "gradient_clip_norm": 1.0 if agent.d2cfr_loss_mode == "anchored" else None,
            "comparison": comparison,
        },
    }
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(target)
    return report


def run_checkpoint_replay_freshness_probe(
    checkpoint_path: str | Path,
    *,
    config_path: str | Path,
    output_path: str | Path,
    recent_iterations: int = 25,
    player_id: int = 0,
    fit_samples: int = 20_000,
    fit_steps: int = 2_000,
    fit_batch_size: int = 256,
    frozen_states: int = 64,
    frozen_repeats: int = 64,
    frozen_roots: int = 4_096,
    seed: int = 0,
    device: str | torch.device = "cpu",
) -> dict[str, Any]:
    """Сравнивает полный replay с окном последних итераций на frozen postflop target'ах."""
    from src.core.deep_cfr import DeepCFRAgent
    from src.training import train as train_mod
    from src.utils import config as config_mod

    actor_id = int(player_id)
    if actor_id not in (0, 1):
        raise ValueError("player_id должен быть 0 или 1 для HU")
    if min(recent_iterations, fit_samples, fit_steps, fit_batch_size, frozen_states, frozen_repeats, frozen_roots) < 1:
        raise ValueError("Все размеры freshness-проверки должны быть положительными")
    config_mod.load_config(config_path)
    agent = DeepCFRAgent(player_id=0, num_players=2, device=str(device))
    coordinator = train_mod._create_hu_current_policy_coordinator(agent)
    checkpoint = train_mod._load_hu_checkpoint(agent, checkpoint_path)
    if not agent.d2cfr_enabled:
        raise ValueError("Нужен D2CFR HU checkpoint")

    latest_iteration = int(checkpoint["iteration"])
    minimum_iteration = max(1, latest_iteration - int(recent_iterations) + 1)
    buffer = agent.hu_advantage_buffers[actor_id]
    buffer_size_before_targets = int(len(buffer))
    recent_available = int(np.count_nonzero(np.asarray(buffer._iterations[:len(buffer)], dtype=np.int64) >= minimum_iteration))
    common_samples = min(int(fit_samples), buffer_size_before_targets, recent_available)
    if common_samples <= 0:
        raise ValueError("В recent window нет D2CFR samples")
    full_batch = signal_probe.build_probe_batch(
        signal_probe.samples_from_dueling_buffer(buffer, sample_count=common_samples, seed=int(seed)), device=device,
    )
    recent_batch = signal_probe.build_probe_batch(
        _samples_from_dueling_buffer_window(
            buffer, minimum_iteration=minimum_iteration, sample_count=common_samples, seed=int(seed),
        ),
        device=device,
    )
    evaluation_batch, states_collected = _collect_frozen_postflop_evaluation_batch(
        agent, coordinator, player_id=actor_id, iteration=latest_iteration, states=int(frozen_states),
        repeats=int(frozen_repeats), roots=int(frozen_roots), seed=int(seed), device=device,
    )
    comparison = run_replay_freshness_fit_ab(
        network_factory=agent._new_advantage_network,
        full_batch=full_batch,
        recent_batch=recent_batch,
        evaluation_batch=evaluation_batch,
        steps=int(fit_steps), batch_size=int(fit_batch_size), learning_rate=float(agent.advantage_lr),
        state_value_loss_weight=float(agent.d2cfr_state_value_loss_weight), seed=int(seed),
        max_grad_norm=1.0 if agent.d2cfr_loss_mode == "anchored" else None,
    )
    report = {
        "schema_version": 1,
        "read_only": True,
        "checkpoint": str(checkpoint_path),
        "checkpoint_iteration": latest_iteration,
        "config": str(config_path),
        "player_id": actor_id,
        "seed": int(seed),
        "replay_window": {
            "recent_iterations": int(recent_iterations),
            "minimum_source_iteration": minimum_iteration,
            "full_available_samples": buffer_size_before_targets,
            "recent_available_samples": recent_available,
            "equal_samples_per_arm": common_samples,
        },
        "frozen_postflop": {
            "states_requested": int(frozen_states), "states_collected": states_collected,
            "repeats_per_state": int(frozen_repeats), "roots_examined": int(frozen_roots),
            "targets": "mean of repeated frozen-policy ES traversals",
        },
        "fit": {
            "loss_function": str(agent.d2cfr_loss_function), "loss_mode": str(agent.d2cfr_loss_mode),
            "gradient_clip_norm": 1.0 if agent.d2cfr_loss_mode == "anchored" else None,
            "comparison": comparison,
        },
    }
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(target)
    return report


def run_heldout_fit_probe(
    network: Any,
    batch: signal_probe.ProbeBatch,
    *,
    train_fraction: float = 0.8,
    seed: int = 7,
    steps: int = 100,
    learning_rate: float = 1e-3,
    state_value_loss_weight: float = 0.5,
) -> dict[str, Any]:
    """Обучает clone на train split и оценивает train-vs-validation gap."""
    sample_count = int(batch.states.shape[0])
    if sample_count < 2:
        raise ValueError("held-out probe требует минимум два sample")
    fraction = min(max(float(train_fraction), 0.1), 0.9)
    generator = torch.Generator(device=batch.states.device)
    generator.manual_seed(int(seed))
    permutation = torch.randperm(sample_count, generator=generator, device=batch.states.device)
    train_count = min(max(int(round(sample_count * fraction)), 1), sample_count - 1)
    train_batch = _slice_probe_batch(batch, permutation[:train_count])
    validation_batch = _slice_probe_batch(batch, permutation[train_count:])

    probe_net = copy.deepcopy(network)
    probe_net.to(batch.states.device)
    optimizer = torch.optim.Adam(probe_net.parameters(), lr=float(learning_rate))

    def mse_loss(target_batch: signal_probe.ProbeBatch) -> torch.Tensor:
        components = probe_net.forward_components(target_batch.states)
        regret_error = (components.regrets - target_batch.regret_targets).square() * target_batch.masks
        regret_loss = regret_error.sum() / torch.clamp(target_batch.masks.sum(), min=1.0)
        state_error = (components.state_values.squeeze(1) - target_batch.state_value_targets).square().mean()
        return regret_loss + float(state_value_loss_weight) * state_error

    history = []
    checkpoints = {0, max(int(steps), 0)}
    for step in range(0, max(int(steps), 0) + 1):
        if step in checkpoints:
            history.append(
                {
                    "step": int(step),
                    "train": evaluate_prediction_snapshot(probe_net, train_batch),
                    "validation": evaluate_prediction_snapshot(probe_net, validation_batch),
                }
            )
        if step == int(steps):
            break
        probe_net.train()
        optimizer.zero_grad()
        loss = mse_loss(train_batch)
        loss.backward()
        optimizer.step()

    train_final = evaluate_prediction_snapshot(probe_net, train_batch)
    validation_final = evaluate_prediction_snapshot(probe_net, validation_batch)
    return {
        "mode": "heldout_fit",
        "steps": int(steps),
        "learning_rate": float(learning_rate),
        "train_fraction": float(fraction),
        "train": train_final,
        "validation": validation_final,
        "generalization_gap_eval_loss_mse": float(
            validation_final["eval_loss_mse"] - train_final["eval_loss_mse"]
        ),
        "generalization_gap_rm_l1": float(
            validation_final["rm_policy_l1_mean"] - train_final["rm_policy_l1_mean"]
        ),
        "history": history,
    }


def samples_from_dueling_buffer_for_iteration(
    buffer: Any,
    *,
    iteration: int,
    sample_count: int,
    seed: int,
) -> list[tuple]:
    """Копирует samples только заданной CFR-итерации без изменения RNG буфера."""
    size = len(buffer)
    if size <= 0:
        raise ValueError("D2CFR advantage_buffer пуст")
    required_arrays = (
        "_states",
        "_action_values",
        "_state_values",
        "_regrets",
        "_masks",
        "_iterations",
    )
    try:
        arrays = [getattr(buffer, name) for name in required_arrays]
    except AttributeError as error:
        raise ValueError("live buffer не похож на DuelingAdvantageBuffer") from error

    iteration_array = np.asarray(arrays[5][:size], dtype=np.float32)
    matching = np.flatnonzero(iteration_array == np.float32(iteration))
    if matching.size <= 0:
        raise ValueError(f"D2CFR buffer не содержит samples iteration={iteration}")
    take = min(int(sample_count), int(matching.size))
    rng = np.random.default_rng(int(seed))
    selected = rng.choice(matching, take, replace=False)
    return [
        (
            arrays[0][index].copy(),
            arrays[1][index].copy(),
            np.float32(arrays[2][index]),
            arrays[3][index].copy(),
            arrays[4][index].copy(),
            np.float32(arrays[5][index]),
        )
        for index in selected
    ]


def _iteration_root_bank(*, roots: int, seed: int, iteration: int, num_players: int) -> list[Any]:
    from tools.benchmark_traversal import generate_root_bank

    return generate_root_bank(
        int(roots),
        int(seed) + int(iteration) * 100_003,
        int(num_players),
    )


def run_accumulation_probe(
    agent: Any,
    *,
    iterations: int,
    roots_per_iteration: int,
    diagnostic_samples: int,
    seed: int,
    traversing_player: int,
    num_players: int,
    device: str | torch.device,
    state_factory: Any,
    batch_size: int | None = None,
    epochs: int | None = None,
    sample_dump_limit: int = 3,
    max_features: int = 16,
    heldout_steps: int = 0,
    heldout_train_fraction: float = 0.8,
    heldout_learning_rate: float = 1e-3,
) -> dict[str, Any]:
    """Запускает live traversal+training и меряет восстановление fixed samples."""
    if iterations <= 0:
        raise ValueError("iterations должно быть положительным")
    if roots_per_iteration <= 0:
        raise ValueError("roots-per-iteration должно быть положительным")
    if diagnostic_samples <= 0:
        raise ValueError("diagnostic-samples должно быть положительным")
    if not bool(getattr(agent, "d2cfr_enabled", False)):
        raise ValueError("accumulation probe требует d2cfr_enabled=true")
    buffer = getattr(agent, "d2cfr_buffer", None)
    if buffer is None:
        raise ValueError("agent не содержит d2cfr_buffer")

    from tools.benchmark_traversal import seed_everything

    diagnostic_batch: signal_probe.ProbeBatch | None = None
    history: list[dict[str, Any]] = []
    seed_everything(int(seed))

    for iteration in range(1, int(iterations) + 1):
        agent.iteration_count = int(iteration)
        if hasattr(agent, "prepare_iteration"):
            agent.prepare_iteration(int(iteration), int(traversing_player))
        if hasattr(agent, "reset_traversal_stats"):
            agent.reset_traversal_stats()

        roots = _iteration_root_bank(
            roots=int(roots_per_iteration),
            seed=int(seed),
            iteration=int(iteration),
            num_players=int(num_players),
        )
        for root in roots:
            agent.cfr_traverse_multi(
                state_factory(root),
                int(iteration),
                int(traversing_player),
            )

        if diagnostic_batch is None:
            diagnostic_batch = _copy_diagnostic_samples(
                buffer,
                sample_count=int(diagnostic_samples),
                seed=int(seed),
                device=device,
            )
        current_batch = signal_probe.build_probe_batch(
            samples_from_dueling_buffer_for_iteration(
                buffer,
                iteration=int(iteration),
                sample_count=int(diagnostic_samples),
                seed=int(seed) + int(iteration) * 997,
            ),
            device=device,
        )

        train_loss = float(
            agent.train_advantage_network_multi(
                batch_size=batch_size,
                epochs=epochs,
                player_id=int(traversing_player),
            )
        )
        old_prediction = evaluate_prediction_snapshot(agent.advantage_net, diagnostic_batch)
        current_prediction = evaluate_prediction_snapshot(agent.advantage_net, current_batch)
        entry = {
            "iteration": int(iteration),
            "buffer_size": int(len(buffer)),
            "train_loss": train_loss,
            "train_steps": int(getattr(agent, "last_advantage_train_steps", 0) or 0),
            "effective_batch_size": int(getattr(agent, "last_advantage_effective_batch_size", 0) or 0),
            "target_stats": getattr(agent, "last_advantage_target_stats", None),
            "traversal_stats": agent.get_traversal_stats() if hasattr(agent, "get_traversal_stats") else {},
            "old": {
                "label": "old_fixed",
                "source_iterations": sorted(
                    {float(item) for item in diagnostic_batch.iterations.detach().cpu().tolist()}
                ),
                "prediction": old_prediction,
            },
            "current": {
                "label": "current_iteration",
                "source_iterations": sorted(
                    {float(item) for item in current_batch.iterations.detach().cpu().tolist()}
                ),
                "prediction": current_prediction,
            },
            "prediction": old_prediction,
        }
        if sample_dump_limit > 0:
            old_dump = describe_prediction_rows(
                agent.advantage_net,
                diagnostic_batch,
                sample_limit=sample_dump_limit,
                max_features=max_features,
            )
            current_dump = describe_prediction_rows(
                agent.advantage_net,
                current_batch,
                sample_limit=sample_dump_limit,
                max_features=max_features,
            )
            entry["old"]["sample_dump"] = old_dump
            entry["current"]["sample_dump"] = current_dump
            entry["sample_dump"] = old_dump
        history.append(entry)

    assert diagnostic_batch is not None
    report = {
        "mode": "live_accumulation",
        "iterations": int(iterations),
        "roots_per_iteration": int(roots_per_iteration),
        "traversing_player": int(traversing_player),
        "num_players": int(num_players),
        "training_request": {
            "batch_size": int(batch_size) if batch_size is not None else None,
            "epochs": int(epochs) if epochs is not None else None,
        },
        "diagnostic": {
            "samples": int(diagnostic_batch.states.shape[0]),
            "input_size": int(diagnostic_batch.states.shape[1]),
            "legal_slots": int(diagnostic_batch.masks.sum().detach().cpu().item()),
            "source_iterations": sorted(
                {float(item) for item in diagnostic_batch.iterations.detach().cpu().tolist()}
            ),
            "targets": signal_probe.analyse_targets(diagnostic_batch),
        },
        "history": history,
    }
    if int(heldout_steps) > 0:
        report["heldout_fit"] = run_heldout_fit_probe(
            agent.advantage_net,
            diagnostic_batch,
            train_fraction=float(heldout_train_fraction),
            seed=int(seed),
            steps=int(heldout_steps),
            learning_rate=float(heldout_learning_rate),
        )
    return report


def run_live_accumulation_probe(
    config_path: str | Path | None,
    *,
    iterations: int,
    roots_per_iteration: int,
    diagnostic_samples: int,
    seed: int,
    traversing_player: int,
    num_players: int | None,
    device: str | torch.device,
    batch_size: int | None,
    epochs: int | None,
    sample_dump_limit: int,
    max_features: int,
    heldout_steps: int,
    heldout_train_fraction: float,
    heldout_learning_rate: float,
) -> dict[str, Any]:
    from src.core.deep_cfr import DeepCFRAgent
    from src.utils import config as config_mod
    from tools.benchmark_traversal import make_state

    if config_path is not None:
        config_mod.load_config(config_path)
    configured_players = config_mod.cfg_get("num_players", 6)
    player_count = int(num_players) if num_players is not None else int(configured_players or 6)
    agent = DeepCFRAgent(
        player_id=int(traversing_player),
        num_players=player_count,
        device=str(device),
    )
    report = run_accumulation_probe(
        agent,
        iterations=iterations,
        roots_per_iteration=roots_per_iteration,
        diagnostic_samples=diagnostic_samples,
        seed=seed,
        traversing_player=traversing_player,
        num_players=player_count,
        device=device,
        state_factory=lambda root: make_state(root, player_count),
        batch_size=batch_size,
        epochs=epochs,
        sample_dump_limit=sample_dump_limit,
        max_features=max_features,
        heldout_steps=heldout_steps,
        heldout_train_fraction=heldout_train_fraction,
        heldout_learning_rate=heldout_learning_rate,
    )
    report["config"] = {
        "path": str(config_path) if config_path is not None else None,
        "d2cfr_state_value_loss_weight": float(config_mod.cfg_get("d2cfr_state_value_loss_weight", 0.5)),
        "d2cfr_loss_function": str(config_mod.cfg_get("d2cfr_loss_function", "mse")),
        "d2cfr_reinitialize_each_iteration": bool(
            config_mod.cfg_get("d2cfr_reinitialize_each_iteration", False)
        ),
        "advantage_batch_size": int(getattr(agent, "advantage_batch_size", 0) or 0),
        "advantage_epochs": int(getattr(agent, "advantage_epochs", 0) or 0),
        "advantage_train_steps": getattr(agent, "advantage_train_steps", None),
    }
    return report


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("config.yaml"))
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--roots-per-iteration", type=int, default=10)
    parser.add_argument("--diagnostic-samples", type=int, default=64)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--traversing-player", type=int, default=0)
    parser.add_argument("--num-players", type=int, default=None)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--dump-samples", type=int, default=3)
    parser.add_argument("--max-features", type=int, default=16)
    parser.add_argument("--heldout-steps", type=int, default=0)
    parser.add_argument("--heldout-train-fraction", type=float, default=0.8)
    parser.add_argument("--heldout-learning-rate", type=float, default=1e-3)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--age-bucket-width", type=int, default=25)
    parser.add_argument("--age-samples", type=int, default=2048)
    parser.add_argument(
        "--weight-mode-ab",
        action="store_true",
        help="Сравнить iteration-weighted и uniform offline fit на frozen postflop targets",
    )
    parser.add_argument(
        "--replay-freshness-ab",
        action="store_true",
        help="Сравнить full replay и recent window на frozen postflop targets",
    )
    parser.add_argument("--recent-iterations", type=int, default=25)
    parser.add_argument("--fit-samples", type=int, default=20_000)
    parser.add_argument("--fit-steps", type=int, default=2_000)
    parser.add_argument("--fit-batch-size", type=int, default=256)
    parser.add_argument("--frozen-states", type=int, default=64)
    parser.add_argument("--frozen-repeats", type=int, default=64)
    parser.add_argument("--frozen-roots", type=int, default=4_096)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.checkpoint is not None:
        if args.output is None:
            raise ValueError("Для checkpoint-анализа нужен --output")
        if args.weight_mode_ab and args.replay_freshness_ab:
            raise ValueError("Выберите только один режим: --weight-mode-ab или --replay-freshness-ab")
        if args.replay_freshness_ab:
            report = run_checkpoint_replay_freshness_probe(
                args.checkpoint,
                config_path=args.config,
                output_path=args.output,
                recent_iterations=args.recent_iterations,
                player_id=args.traversing_player,
                fit_samples=args.fit_samples,
                fit_steps=args.fit_steps,
                fit_batch_size=args.fit_batch_size,
                frozen_states=args.frozen_states,
                frozen_repeats=args.frozen_repeats,
                frozen_roots=args.frozen_roots,
                seed=args.seed,
                device=args.device,
            )
        elif args.weight_mode_ab:
            report = run_checkpoint_weight_mode_ab_probe(
                args.checkpoint,
                config_path=args.config,
                output_path=args.output,
                player_id=args.traversing_player,
                fit_samples=args.fit_samples,
                fit_steps=args.fit_steps,
                fit_batch_size=args.fit_batch_size,
                frozen_states=args.frozen_states,
                frozen_repeats=args.frozen_repeats,
                frozen_roots=args.frozen_roots,
                seed=args.seed,
                device=args.device,
            )
        else:
            report = run_checkpoint_buffer_age_probe(
                args.checkpoint,
                config_path=args.config,
                output_path=args.output,
                player_id=args.traversing_player,
                age_bucket_width=args.age_bucket_width,
                sample_limit_per_bucket=args.age_samples,
                seed=args.seed,
                device=args.device,
            )
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0
    report = run_live_accumulation_probe(
        args.config,
        iterations=args.iterations,
        roots_per_iteration=args.roots_per_iteration,
        diagnostic_samples=args.diagnostic_samples,
        seed=args.seed,
        traversing_player=args.traversing_player,
        num_players=args.num_players,
        device=args.device,
        batch_size=args.batch_size,
        epochs=args.epochs,
        sample_dump_limit=args.dump_samples,
        max_features=args.max_features,
        heldout_steps=args.heldout_steps,
        heldout_train_fraction=args.heldout_train_fraction,
        heldout_learning_rate=args.heldout_learning_rate,
    )
    text = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    else:
        print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
