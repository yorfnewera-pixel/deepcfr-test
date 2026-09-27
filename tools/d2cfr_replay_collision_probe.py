"""Read-only аудит точных коллизий encoded infoset в D2CFR replay."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch


def _legal_range(values: np.ndarray, mask: np.ndarray) -> float:
    legal = np.asarray(mask, dtype=bool)
    if not np.any(legal) or len(values) < 2:
        return 0.0
    return float(np.ptp(values[:, legal], axis=0).mean())


def _sign_disagrees(values: np.ndarray) -> bool:
    return bool(np.any(values > 0.0) and np.any(values < 0.0))


def _validate_buffer(buffer: dict[str, Any]) -> tuple[np.ndarray, ...]:
    required = ("states", "action_values", "state_values", "regrets", "masks", "iterations")
    if not isinstance(buffer, dict) or any(key not in buffer for key in required):
        raise ValueError("Checkpoint не содержит полный D2CFR replay buffer")
    arrays = tuple(np.asarray(buffer[key]) for key in required)
    states, action_values, state_values, regrets, masks, iterations = arrays
    count = len(states)
    if (
        states.ndim != 2
        or action_values.shape != regrets.shape
        or action_values.shape != masks.shape
        or action_values.shape != (count, 6)
        or state_values.shape != (count,)
        or iterations.shape != (count,)
        or not all(np.all(np.isfinite(array)) for array in arrays)
    ):
        raise ValueError("D2CFR replay buffer имеет некорректные формы или значения")
    return arrays


def _group_key(state: np.ndarray, mask: np.ndarray) -> bytes:
    return state.tobytes() + mask.tobytes()


def _group_id(key: bytes) -> str:
    return hashlib.sha256(key).hexdigest()[:16]


def audit_leg_buffer(buffer: dict[str, Any], *, example_limit: int = 20) -> dict[str, Any]:
    """Группирует строго идентичные ``encoded state + mask`` и измеряет target conflict."""
    states, action_values, state_values, regrets, masks, iterations = _validate_buffer(buffer)
    groups: dict[bytes, list[int]] = defaultdict(list)
    for index, (state, mask) in enumerate(zip(states, masks, strict=True)):
        groups[_group_key(state, mask)].append(index)

    collision_groups = [(key, rows) for key, rows in groups.items() if len(rows) > 1]
    within_v_ranges: list[float] = []
    within_r_ranges: list[float] = []
    within_allin_disagreements = 0
    within_duplicate_groups = 0
    within_duplicate_samples = 0
    between_v_ranges: list[float] = []
    between_r_ranges: list[float] = []
    between_allin_disagreements = 0
    between_groups = 0
    examples: list[dict[str, Any]] = []

    for key, rows in collision_groups:
        row_indices = np.asarray(rows, dtype=np.intp)
        group_iterations = iterations[row_indices].astype(np.int64)
        unique_iterations = np.unique(group_iterations)
        mask = masks[row_indices[0]].astype(bool)
        allin_legal = bool(mask[5])
        per_iteration_rows = [row_indices[group_iterations == iteration] for iteration in unique_iterations]

        for iteration_rows in per_iteration_rows:
            if len(iteration_rows) < 2:
                continue
            within_duplicate_groups += 1
            within_duplicate_samples += int(len(iteration_rows))
            within_v_ranges.append(float(np.ptp(state_values[iteration_rows])))
            within_r_ranges.append(_legal_range(regrets[iteration_rows], mask))
            if allin_legal and _sign_disagrees(regrets[iteration_rows, 5]):
                within_allin_disagreements += 1

        if len(unique_iterations) > 1:
            between_groups += 1
            mean_q = np.stack([action_values[index_set].mean(axis=0) for index_set in per_iteration_rows])
            mean_v = np.asarray([state_values[index_set].mean() for index_set in per_iteration_rows])
            mean_r = np.stack([regrets[index_set].mean(axis=0) for index_set in per_iteration_rows])
            between_v_ranges.append(float(np.ptp(mean_v)))
            between_r_ranges.append(_legal_range(mean_r, mask))
            if allin_legal and _sign_disagrees(mean_r[:, 5]):
                between_allin_disagreements += 1

        examples.append({
            "encoding_hash": _group_id(key),
            "samples": int(len(row_indices)),
            "iterations": [int(value) for value in unique_iterations.tolist()],
            "state_value_range": float(np.ptp(state_values[row_indices])),
            "regret_legal_range_mean": _legal_range(regrets[row_indices], mask),
            "all_in_legal": allin_legal,
            "all_in_target_signs": sorted({int(np.sign(value)) for value in regrets[row_indices, 5]}) if allin_legal else [],
        })

    examples.sort(key=lambda row: (-int(row["samples"]), -float(row["regret_legal_range_mean"])))
    return {
        "samples": int(len(states)),
        "unique_exact_encodings": int(len(groups)),
        "exact_collision_groups": int(len(collision_groups)),
        "collision_samples": int(sum(len(rows) for _, rows in collision_groups)),
        "within_iteration": {
            "duplicate_groups": int(within_duplicate_groups),
            "duplicate_samples": int(within_duplicate_samples),
            "state_value_range_mean": float(np.mean(within_v_ranges)) if within_v_ranges else 0.0,
            "regret_legal_range_mean": float(np.mean(within_r_ranges)) if within_r_ranges else 0.0,
            "groups_with_all_in_sign_disagreement": int(within_allin_disagreements),
        },
        "between_iterations": {
            "groups_with_multiple_iterations": int(between_groups),
            "state_value_range_mean": float(np.mean(between_v_ranges)) if between_v_ranges else 0.0,
            "regret_legal_range_mean": float(np.mean(between_r_ranges)) if between_r_ranges else 0.0,
            "all_in_mean_sign_disagreement_groups": int(between_allin_disagreements),
        },
        "collision_examples": examples[:max(0, int(example_limit))],
    }


def audit_checkpoint(checkpoint_path: str | Path, *, example_limit: int = 20) -> dict[str, Any]:
    """Читает оба HU D2CFR replay buffer из checkpoint без создания агента."""
    checkpoint = torch.load(Path(checkpoint_path), map_location="cpu", weights_only=False)
    legs = checkpoint.get("advantage_legs")
    if not isinstance(legs, list) or len(legs) != 2:
        raise ValueError("Нужен HU D2CFR checkpoint с двумя advantage legs")
    return {
        "schema_version": 1,
        "read_only": True,
        "checkpoint": str(checkpoint_path),
        "checkpoint_iteration": int(checkpoint.get("iteration", 0)),
        "method": "exact encoded-state plus legal-mask equality; no approximate matching",
        "players": {
            f"P{player_id}": audit_leg_buffer(leg.get("buffer"), example_limit=example_limit)
            for player_id, leg in enumerate(legs)
        },
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--example-limit", type=int, default=20)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    report = audit_checkpoint(args.checkpoint, example_limit=args.example_limit)
    target = Path(args.output)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(target)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
