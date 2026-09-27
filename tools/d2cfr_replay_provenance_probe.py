"""Read-only аудит: доказывает или опровергает aliasing полной history в D2CFR replay."""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

from tools.d2cfr_replay_collision_probe import _group_key, _validate_buffer


def _provenance_key(value: np.ndarray) -> bytes:
    return np.asarray(value, dtype=np.uint8).tobytes()


def _history_id(value: np.ndarray) -> str:
    return hashlib.sha256(_provenance_key(value)).hexdigest()[:16]


def audit_leg_buffer(buffer: dict[str, Any], *, example_limit: int = 20) -> dict[str, Any]:
    """Ищет разные полные infoset внутри одинакового encoded state и одной итерации."""
    states, action_values, state_values, regrets, masks, iterations = _validate_buffer(buffer)
    raw_provenances = buffer.get("provenances") if isinstance(buffer, dict) else None
    provenance_available = bool(buffer.get("provenance_enabled", False)) and raw_provenances is not None
    if not provenance_available:
        return {
            "samples": int(len(states)),
            "provenance_available": False,
            "reason": "checkpoint создан без d2cfr_replay_provenance_audit",
        }
    provenances = np.asarray(raw_provenances)
    if provenances.shape != (len(states), 16) or provenances.dtype != np.uint8:
        raise ValueError("Checkpoint содержит повреждённый D2CFR replay provenance")

    by_encoding: dict[bytes, list[int]] = defaultdict(list)
    for index, (state, mask) in enumerate(zip(states, masks, strict=True)):
        by_encoding[_group_key(state, mask)].append(index)

    different_history_groups = 0
    different_history_samples = 0
    same_history_duplicate_groups = 0
    same_history_duplicate_samples = 0
    same_history_v_ranges: list[float] = []
    same_history_r_ranges: list[float] = []
    examples: list[dict[str, Any]] = []

    for encoding, rows in by_encoding.items():
        if len(rows) < 2:
            continue
        row_indices = np.asarray(rows, dtype=np.intp)
        for iteration in np.unique(iterations[row_indices].astype(np.int64)):
            iteration_rows = row_indices[iterations[row_indices].astype(np.int64) == iteration]
            if len(iteration_rows) < 2:
                continue
            by_history: dict[bytes, list[int]] = defaultdict(list)
            for index in iteration_rows:
                by_history[_provenance_key(provenances[index])].append(int(index))
            if len(by_history) > 1:
                different_history_groups += 1
                different_history_samples += int(len(iteration_rows))
                examples.append({
                    "encoding_hash": hashlib.sha256(encoding).hexdigest()[:16],
                    "iteration": int(iteration),
                    "samples": int(len(iteration_rows)),
                    "different_history_fingerprints": sorted(
                        _history_id(provenances[index]) for index in iteration_rows
                    ),
                })
            for history_rows in by_history.values():
                if len(history_rows) < 2:
                    continue
                indices = np.asarray(history_rows, dtype=np.intp)
                same_history_duplicate_groups += 1
                same_history_duplicate_samples += int(len(indices))
                same_history_v_ranges.append(float(np.ptp(state_values[indices])))
                legal = masks[indices[0]].astype(bool)
                same_history_r_ranges.append(
                    float(np.ptp(regrets[indices][:, legal], axis=0).mean()) if np.any(legal) else 0.0
                )

    examples.sort(key=lambda item: (-int(item["samples"]), item["encoding_hash"]))
    return {
        "samples": int(len(states)),
        "provenance_available": True,
        "method": "same encoded state + mask + iteration; fingerprints differ only when full infoset differs",
        "within_iteration": {
            "different_history_groups": int(different_history_groups),
            "different_history_samples": int(different_history_samples),
            "same_history_duplicate_groups": int(same_history_duplicate_groups),
            "same_history_duplicate_samples": int(same_history_duplicate_samples),
            "same_history_state_value_range_mean": float(np.mean(same_history_v_ranges)) if same_history_v_ranges else 0.0,
            "same_history_regret_legal_range_mean": float(np.mean(same_history_r_ranges)) if same_history_r_ranges else 0.0,
        },
        "different_history_examples": examples[:max(0, int(example_limit))],
    }


def audit_checkpoint(checkpoint_path: str | Path, *, example_limit: int = 20) -> dict[str, Any]:
    checkpoint = torch.load(Path(checkpoint_path), map_location="cpu", weights_only=False)
    legs = checkpoint.get("advantage_legs")
    if not isinstance(legs, list) or len(legs) != 2:
        raise ValueError("Нужен HU D2CFR checkpoint с двумя advantage legs")
    return {
        "schema_version": 1,
        "read_only": True,
        "checkpoint": str(checkpoint_path),
        "checkpoint_iteration": int(checkpoint.get("iteration", 0)),
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
