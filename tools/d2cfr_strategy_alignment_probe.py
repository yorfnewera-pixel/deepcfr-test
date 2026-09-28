"""Read-only аудит: historical strategy buffer против masked softmax strategy_net."""
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

from tools.d2cfr_objective_alignment_probe import _restore_agent

_THRESHOLDS = (2, 5, 10, 20)
_ALL_IN = 5


def historical_policy_target(policies: np.ndarray, masks: np.ndarray, iterations: np.ndarray) -> np.ndarray:
    """MSE-optimum для policy targets, action-wise только среди legal samples."""
    result = np.zeros(policies.shape[1], dtype=np.float64)
    weights = np.asarray(iterations, dtype=np.float64)
    for action in range(policies.shape[1]):
        legal = np.asarray(masks[:, action], dtype=bool)
        if legal.any():
            result[action] = float(np.average(policies[legal, action], weights=weights[legal]))
    return result


def all_in_extreme_summary(historical: np.ndarray, predicted: np.ndarray) -> dict[str, float | int]:
    """Считает только большие и практически значимые расхождения all-in probability."""
    historical = np.asarray(historical, dtype=np.float64)
    predicted = np.asarray(predicted, dtype=np.float64)
    low = historical < 0.1
    high = historical > 0.5
    low_to_high = int(np.count_nonzero(low & (predicted > 0.5)))
    high_to_low = int(np.count_nonzero(high & (predicted < 0.1)))
    return {
        "historical_low_count": int(low.sum()),
        "historical_high_count": int(high.sum()),
        "historical_low_predicted_high": low_to_high,
        "historical_high_predicted_low": high_to_low,
        "historical_low_predicted_high_rate": low_to_high / int(low.sum()) if low.any() else 0.0,
        "historical_high_predicted_low_rate": high_to_low / int(high.sum()) if high.any() else 0.0,
    }


def _groups(states: np.ndarray, actors: np.ndarray) -> dict[bytes, list[int]]:
    result: dict[bytes, list[int]] = defaultdict(list)
    for index, (state, actor) in enumerate(zip(states, actors, strict=True)):
        result[np.asarray(state, dtype=np.float32).tobytes() + bytes([int(actor)])].append(index)
    return result


def _masked_softmax(network: torch.nn.Module, states: np.ndarray, masks: np.ndarray, device: str) -> np.ndarray:
    with torch.inference_mode():
        logits = network(torch.as_tensor(states, dtype=torch.float32, device=device))
        masked = torch.where(torch.as_tensor(masks, device=device) > 0.0, logits, torch.full_like(logits, -1e20))
        return torch.softmax(masked, dim=1).cpu().numpy().astype(np.float64)


def _summarize(agent: Any) -> dict[str, Any]:
    buffer = agent.hu_strategy_buffer
    count = len(buffer)
    states, actors, policies, masks, iterations = (
        buffer._states[:count], buffer._actor_ids[:count], buffer._policies[:count],
        buffer._masks[:count], buffer._iterations[:count].astype(np.int64),
    )
    groups = _groups(states, actors)
    items = [rows for rows in groups.values() if len(rows) >= 2]
    representatives = np.stack([
        np.concatenate((states[rows[0]], np.eye(2, dtype=np.float32)[actors[rows[0]]])) for rows in items
    ])
    predictions: list[np.ndarray] = []
    # Выход сети зависит от legal mask, поэтому ниже logits будут вызываться для каждого mask варианта.
    by_threshold: dict[str, Any] = {}
    for minimum in _THRESHOLDS:
        selected = [rows for rows in items if len(rows) >= minimum]
        l1, mae, allin_hist, allin_pred = [], [], [], []
        for rows in selected:
            indices = np.asarray(rows, dtype=np.intp)
            target = historical_policy_target(policies[indices], masks[indices], iterations[indices])
            unique_masks = {masks[index].tobytes(): masks[index] for index in indices}
            conditioned = np.concatenate((states[indices[0]], np.eye(2, dtype=np.float32)[actors[indices[0]]]))
            for mask in unique_masks.values():
                prediction = _masked_softmax(agent.strategy_net, conditioned[None, :], mask[None, :], str(agent.device))[0]
                legal = np.asarray(mask, dtype=bool)
                if not np.isclose(target[legal].sum(), 1.0, atol=1e-5):
                    continue
                l1.append(float(np.abs(prediction - target).sum()))
                mae.append(float(np.abs(prediction[legal] - target[legal]).mean()))
                if legal[_ALL_IN]:
                    allin_hist.append(float(target[_ALL_IN]))
                    allin_pred.append(float(prediction[_ALL_IN]))
        extremes = all_in_extreme_summary(np.asarray(allin_hist), np.asarray(allin_pred)) if allin_hist else all_in_extreme_summary(np.array([]), np.array([]))
        by_threshold[str(minimum)] = {
            "groups": len(selected),
            "replay_samples": int(sum(len(rows) for rows in selected)),
            "policy_l1": float(np.mean(l1)) if l1 else 0.0,
            "policy_mae": float(np.mean(mae)) if mae else 0.0,
            "all_in_probability_bias": float(np.mean(np.asarray(allin_pred) - np.asarray(allin_hist))) if allin_hist else 0.0,
            "groups_with_legal_all_in": len(allin_hist),
            "all_in_extremes": extremes,
        }
    return {
        "replay_samples": int(count), "unique_actor_conditioned_encodings": len(groups),
        "group_size_thresholds": by_threshold,
        "street_breakdown": "unavailable: strategy replay stores encoded states, not raw pokers.State",
    }


def run_probe(checkpoint_path: str | Path, *, config_path: str | Path, output_path: str | Path, device: str = "cpu") -> dict[str, Any]:
    agent, checkpoint = _restore_agent(checkpoint_path, config_path=config_path, device=device)
    report = {
        "schema_version": 1, "read_only": True, "checkpoint": str(checkpoint_path),
        "checkpoint_iteration": int(checkpoint["iteration"]),
        "objective_scope": "iteration-weighted historical strategy policies retained in checkpoint reservoir",
        "loss_contract": {"loss": "masked policy MSE", "iteration_weight_mode": str(agent.d2cfr_iteration_weight_mode)},
        "players": _summarize(agent),
    }
    target = Path(output_path); target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"); temporary.replace(target)
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True); parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--output", required=True); parser.add_argument("--device", default="cpu")
    args = parser.parse_args(argv)
    run_probe(args.checkpoint, config_path=args.config, output_path=args.output, device=args.device)
    print(f"Отчёт сохранён: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
