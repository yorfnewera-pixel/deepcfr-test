"""Read-only проверка checkpoint-сети относительно historical objective replay reservoir."""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.core.deep_cfr import DeepCFRAgent
from src.training import train as train_mod
from src.utils import config as config_mod
from tools.d2cfr_allin_bias_probe import (
    _apply_checkpoint_training_contract,
    _checkpoint_provenance_enabled,
)

_ALL_IN_INDEX = 5
_GROUP_THRESHOLDS = (2, 5, 10, 20)


def weighted_target_statistics(
    values: np.ndarray, iterations: np.ndarray, weights: np.ndarray,
) -> dict[str, float]:
    """Возвращает weighted mean и разложение variance по iteration для одного label."""
    values = np.asarray(values, dtype=np.float64)
    iterations = np.asarray(iterations, dtype=np.int64)
    weights = np.asarray(weights, dtype=np.float64)
    if values.ndim != 1 or iterations.shape != values.shape or weights.shape != values.shape:
        raise ValueError("values, iterations и weights должны быть одномерными одинаковой длины")
    if len(values) == 0 or np.any(weights < 0.0) or float(weights.sum()) <= 0.0:
        raise ValueError("Для target statistics нужны samples с положительным суммарным весом")
    total_weight = float(weights.sum())
    mean = float(np.average(values, weights=weights))
    variance = float(np.average((values - mean) ** 2, weights=weights))
    within_variance = 0.0
    between_variance = 0.0
    for iteration in np.unique(iterations):
        selection = iterations == iteration
        iteration_weights = weights[selection]
        iteration_weight = float(iteration_weights.sum())
        if iteration_weight <= 0.0:
            continue
        iteration_mean = float(np.average(values[selection], weights=iteration_weights))
        proportion = iteration_weight / total_weight
        within_variance += proportion * float(
            np.average((values[selection] - iteration_mean) ** 2, weights=iteration_weights)
        )
        between_variance += proportion * (iteration_mean - mean) ** 2
    return {
        "mean": mean,
        "variance": variance,
        "within_iteration_variance": float(within_variance),
        "between_iteration_variance": float(between_variance),
    }


def action_target_statistics(
    regrets: np.ndarray,
    masks: np.ndarray,
    iterations: np.ndarray,
    weights: np.ndarray,
    *,
    action_index: int,
) -> dict[str, float | int]:
    """Агрегирует regret действия только по samples, где действие было legal."""
    legal = np.asarray(masks[:, action_index], dtype=bool)
    if not np.any(legal):
        return {"samples": 0}
    result = weighted_target_statistics(regrets[legal, action_index], iterations[legal], weights[legal])
    return {"samples": int(legal.sum()), **result}


def margin_flip_counts(
    historical: np.ndarray, predicted: np.ndarray, *, margin: float,
) -> dict[str, int]:
    """Считает только flips, удалённые от нуля с обеих сторон заданного margin."""
    historical = np.asarray(historical, dtype=np.float64)
    predicted = np.asarray(predicted, dtype=np.float64)
    if historical.shape != predicted.shape:
        raise ValueError("historical и predicted должны иметь одинаковую форму")
    return {
        "historical_negative_predicted_positive": int(
            np.count_nonzero((historical < -margin) & (predicted > margin))
        ),
        "historical_positive_predicted_negative": int(
            np.count_nonzero((historical > margin) & (predicted < -margin))
        ),
    }


def _batch_mean_effective_weights(
    iterations: np.ndarray, *, batch_size: int, batches: int, seed: int,
) -> np.ndarray:
    """Monte-Carlo коэффициенты samples в фактическом batch_mean_1 SGD objective."""
    iterations = np.asarray(iterations, dtype=np.float64)
    if iterations.ndim != 1 or len(iterations) == 0 or np.any(iterations <= 0.0):
        raise ValueError("iterations должны быть положительным одномерным массивом")
    if batch_size < 1 or batches < 1:
        raise ValueError("batch_size и batches должны быть положительными")
    rng = np.random.default_rng(seed)
    effective = np.zeros(len(iterations), dtype=np.float64)
    sample_size = min(int(batch_size), len(iterations))
    for _ in range(int(batches)):
        selected = rng.choice(len(iterations), size=sample_size, replace=False)
        selected_iterations = iterations[selected]
        np.add.at(effective, selected, selected_iterations / float(selected_iterations.mean()))
    if float(effective.sum()) <= 0.0:
        raise RuntimeError("Не удалось оценить effective batch weights")
    return effective


def _restore_agent(checkpoint_path: str | Path, *, config_path: str | Path, device: str) -> tuple[DeepCFRAgent, dict[str, Any]]:
    config_mod.load_config(config_path)
    contract = torch.load(Path(checkpoint_path), map_location="cpu", weights_only=True)
    if not isinstance(contract, dict):
        raise ValueError("Не удалось прочитать HU checkpoint")
    agent = DeepCFRAgent(player_id=0, num_players=2, device=device)
    _apply_checkpoint_training_contract(agent, contract)
    agent.d2cfr_replay_provenance_audit = _checkpoint_provenance_enabled(checkpoint_path)
    train_mod._create_hu_current_policy_coordinator(agent)
    checkpoint = train_mod._load_hu_checkpoint(agent, checkpoint_path)
    if not agent.d2cfr_enabled or agent.d2cfr_loss_function != "mse":
        raise ValueError("Probe поддерживает только HU D2CFR checkpoint с MSE loss")
    return agent, checkpoint


def _regret_matching(regrets: np.ndarray, mask: np.ndarray) -> np.ndarray:
    positive = np.maximum(np.asarray(regrets, dtype=np.float64), 0.0) * np.asarray(mask, dtype=np.float64)
    if float(positive.sum()) > 0.0:
        return positive / float(positive.sum())
    legal = np.asarray(mask, dtype=bool)
    result = np.zeros_like(positive)
    result[legal] = 1.0 / float(np.count_nonzero(legal))
    return result


def _group_rows(states: np.ndarray) -> dict[bytes, list[int]]:
    groups: dict[bytes, list[int]] = defaultdict(list)
    for index, state in enumerate(states):
        groups[np.asarray(state, dtype=np.float32).tobytes()].append(index)
    return groups


def _summarize_leg(
    *,
    network: torch.nn.Module,
    buffer: Any,
    batch_size: int,
    mc_batches: int,
    seed: int,
    device: str,
) -> dict[str, Any]:
    count = len(buffer)
    states = buffer._states[:count]
    values = buffer._state_values[:count]
    regrets = buffer._regrets[:count]
    masks = buffer._masks[:count]
    iterations = buffer._iterations[:count].astype(np.int64)
    groups = _group_rows(states)
    # Одиночные samples не входят ни в один отчётный порог и не дают estimate target variance.
    group_items = [rows for rows in groups.values() if len(rows) >= min(_GROUP_THRESHOLDS)]
    paper_weights = iterations.astype(np.float64)
    effective_weights = _batch_mean_effective_weights(
        iterations, batch_size=batch_size, batches=mc_batches, seed=seed,
    )
    mask_variants = sum(len({masks[index].tobytes() for index in rows}) > 1 for rows in groups.values())
    device_object = torch.device(device)
    network.eval()
    representatives = np.stack([states[rows[0]] for rows in group_items]).astype(np.float32)
    predicted_values: list[np.ndarray] = []
    predicted_regrets: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(representatives), 4096):
            batch = torch.as_tensor(representatives[start:start + 4096], dtype=torch.float32, device=device_object)
            components = network.forward_components(batch)
            predicted_values.append(components.state_values.squeeze(-1).detach().cpu().numpy())
            predicted_regrets.append(components.regrets.detach().cpu().numpy())
    all_predicted_values = np.concatenate(predicted_values).astype(np.float64)
    all_predicted_regrets = np.concatenate(predicted_regrets).astype(np.float64)
    records: list[dict[str, Any]] = []
    for group_index, rows in enumerate(group_items):
        indices = np.asarray(rows, dtype=np.intp)
        predicted_v = float(all_predicted_values[group_index])
        predicted_regrets = all_predicted_regrets[group_index]
        paper_v = weighted_target_statistics(values[indices], iterations[indices], paper_weights[indices])
        effective_v = weighted_target_statistics(values[indices], iterations[indices], effective_weights[indices])
        paper_actions = [
            action_target_statistics(regrets[indices], masks[indices], iterations[indices], paper_weights[indices], action_index=action)
            for action in range(regrets.shape[1])
        ]
        effective_actions = [
            action_target_statistics(regrets[indices], masks[indices], iterations[indices], effective_weights[indices], action_index=action)
            for action in range(regrets.shape[1])
        ]
        records.append({
            "samples": int(len(indices)), "masks": masks[indices].copy(),
            "predicted_v": predicted_v, "predicted_regrets": predicted_regrets,
            "paper_v": paper_v, "effective_v": effective_v,
            "paper_actions": paper_actions, "effective_actions": effective_actions,
        })

    by_threshold: dict[str, Any] = {}
    for minimum in _GROUP_THRESHOLDS:
        selected = [record for record in records if record["samples"] >= minimum]
        allin = [record for record in selected if int(record["paper_actions"][_ALL_IN_INDEX]["samples"]) > 0]
        v_errors = [abs(record["predicted_v"] - float(record["paper_v"]["mean"])) for record in selected]
        effective_v_errors = [abs(record["predicted_v"] - float(record["effective_v"]["mean"])) for record in selected]
        r_errors: list[float] = []
        effective_r_errors: list[float] = []
        policy_l1: list[float] = []
        allin_hist: list[float] = []
        allin_pred: list[float] = []
        within_variances: list[float] = []
        between_variances: list[float] = []
        effective_deltas: list[float] = []
        for record in selected:
            historical = np.zeros(regrets.shape[1], dtype=np.float64)
            present = np.zeros(regrets.shape[1], dtype=bool)
            for action, stats in enumerate(record["paper_actions"]):
                if int(stats["samples"]) > 0:
                    historical[action] = float(stats["mean"])
                    present[action] = True
                    r_errors.append(abs(record["predicted_regrets"][action] - historical[action]))
                    effective_r_errors.append(abs(
                        record["predicted_regrets"][action]
                        - float(record["effective_actions"][action]["mean"])
                    ))
                    within_variances.append(float(stats["within_iteration_variance"]))
                    between_variances.append(float(stats["between_iteration_variance"]))
                    effective_deltas.append(abs(float(record["effective_actions"][action]["mean"]) - historical[action]))
            for mask in {row.tobytes(): row for row in record["masks"]}.values():
                legal = np.asarray(mask, dtype=bool)
                if np.any(legal & ~present):
                    continue
                policy_l1.append(float(np.abs(
                    _regret_matching(record["predicted_regrets"], legal) - _regret_matching(historical, legal)
                ).sum()))
        for record in allin:
            allin_hist.append(float(record["paper_actions"][_ALL_IN_INDEX]["mean"]))
            allin_pred.append(float(record["predicted_regrets"][_ALL_IN_INDEX]))
        flips = margin_flip_counts(np.asarray(allin_hist), np.asarray(allin_pred), margin=0.1) if allin else {
            "historical_negative_predicted_positive": 0, "historical_positive_predicted_negative": 0,
        }
        by_threshold[str(minimum)] = {
            "groups": int(len(selected)),
            "replay_samples": int(sum(record["samples"] for record in selected)),
            "groups_with_legal_all_in": int(len(allin)),
            "v_mae_paper_weighted": float(np.mean(v_errors)) if v_errors else 0.0,
            "v_mae_batch_mean_1_effective": float(np.mean(effective_v_errors)) if effective_v_errors else 0.0,
            "regret_mae_paper_weighted": float(np.mean(r_errors)) if r_errors else 0.0,
            "regret_mae_batch_mean_1_effective": float(np.mean(effective_r_errors)) if effective_r_errors else 0.0,
            "rm_policy_l1": float(np.mean(policy_l1)) if policy_l1 else 0.0,
            "action_within_iteration_variance": float(np.mean(within_variances)) if within_variances else 0.0,
            "action_between_iteration_variance": float(np.mean(between_variances)) if between_variances else 0.0,
            "paper_vs_batch_mean_1_target_mae": float(np.mean(effective_deltas)) if effective_deltas else 0.0,
            "all_in_margin_flips": flips,
        }
    return {
        "replay_samples": int(count),
        "unique_exact_encodings": int(len(groups)),
        "encodings_with_multiple_legal_masks": int(mask_variants),
        "group_size_thresholds": by_threshold,
    }


def run_probe(
    checkpoint_path: str | Path, *, config_path: str | Path, output_path: str | Path,
    mc_batches: int = 4096, seed: int = 20261001, device: str = "cpu",
) -> dict[str, Any]:
    """Сравнивает финальную сеть с paper-weighted target сохранённого reservoir."""
    agent, checkpoint = _restore_agent(checkpoint_path, config_path=config_path, device=device)
    report = {
        "schema_version": 1,
        "read_only": True,
        "checkpoint": str(checkpoint_path),
        "checkpoint_iteration": int(checkpoint["iteration"]),
        "objective_scope": "historical optimum for samples retained in checkpoint replay reservoir",
        "loss_contract": {
            "loss_function": str(agent.d2cfr_loss_function),
            "iteration_weight_mode": str(agent.d2cfr_iteration_weight_mode),
            "batch_size": int(agent.advantage_batch_size),
            "batch_mean_1_mc_batches": int(mc_batches),
        },
        "grouping": "player plus exact encoded state; action targets condition on that action being legal",
        "players": {
            f"P{player}": _summarize_leg(
                network=network, buffer=buffer, batch_size=int(agent.advantage_batch_size),
                mc_batches=mc_batches, seed=seed + player, device=device,
            )
            for player, (network, buffer) in enumerate(zip(agent.hu_advantage_nets, agent.hu_advantage_buffers, strict=True))
        },
    }
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(target)
    return report


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--output", required=True)
    parser.add_argument("--mc-batches", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=20261001)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    report = run_probe(
        args.checkpoint, config_path=args.config, output_path=args.output,
        mc_batches=args.mc_batches, seed=args.seed, device=args.device,
    )
    print(f"Отчёт сохранён: {args.output}")
    for player, result in report["players"].items():
        print(f"{player}: replay={result['replay_samples']}, infosets={result['unique_exact_encodings']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
