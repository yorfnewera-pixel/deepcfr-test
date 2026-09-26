"""Изолированный HU probe: измеряет drift D2CFR targets между iterations."""

from __future__ import annotations

import argparse
import json
import random
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.core.action_space import ActionSlot, legal_action_mask
from tools import d2cfr_allin_bias_probe as allin_probe


_ORDER_PAIRS = (
    ("all_in_vs_raise_1pot", int(ActionSlot.ALL_IN), int(ActionSlot.RAISE_POT)),
    ("raise_1pot_vs_raise_0.5pot", int(ActionSlot.RAISE_POT), int(ActionSlot.RAISE_HALF_POT)),
)


@contextmanager
def _preserve_rng_state() -> Iterator[None]:
    """Не позволяет anchor measurement сдвинуть RNG короткого training run."""
    python_state = random.getstate()
    numpy_state = np.random.get_state()
    torch_state = torch.get_rng_state()
    try:
        yield
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)
        torch.set_rng_state(torch_state)


@contextmanager
def _without_probe_recording(coordinator: Any) -> Iterator[None]:
    """Не даёт anchor traversal добавлять labels в training replay buffers."""
    old_advantage = coordinator._record_d2cfr_advantage
    old_strategy = coordinator._record_strategy
    coordinator._record_d2cfr_advantage = lambda *args, **kwargs: None
    coordinator._record_strategy = lambda *args, **kwargs: None
    try:
        yield
    finally:
        coordinator._record_d2cfr_advantage = old_advantage
        coordinator._record_strategy = old_strategy


def _mean_targets_for_anchors(
    coordinator: Any,
    anchors: list[Any],
    *,
    repeats: int,
    iteration: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with _preserve_rng_state():
        coordinator.begin_iteration()
        with _without_probe_recording(coordinator):
            for state in anchors:
                samples = allin_probe._sample_state_targets(
                    coordinator, state, repeats=int(repeats), traverser=0, iteration=int(iteration),
                )
                q = np.stack([sample[0] for sample in samples], axis=0)
                v = np.asarray([sample[1] for sample in samples], dtype=np.float64)
                r = np.stack([sample[2] for sample in samples], axis=0)
                rows.append({
                    "q": q.mean(axis=0),
                    "v": float(v.mean()),
                    "r": r.mean(axis=0),
                    "mask": legal_action_mask(state).astype(np.float64),
                    "q_std": q.std(axis=0, ddof=0),
                })
    return rows


def _ordering_drift(reference: dict[str, Any], current: dict[str, Any]) -> dict[str, dict[str, float | int | None]]:
    report: dict[str, dict[str, float | int | None]] = {}
    for label, first, second in _ORDER_PAIRS:
        legal = bool(reference["mask"][first] and reference["mask"][second])
        reference_gap = float(reference["q"][first] - reference["q"][second])
        current_gap = float(current["q"][first] - current["q"][second])
        comparable = legal and abs(reference_gap) > 1e-8
        report[label] = {
            "legal_pairs": int(legal),
            "comparable_pairs": int(comparable),
            "ordering_preserved": (
                bool((reference_gap > 0.0) == (current_gap > 0.0)) if comparable else None
            ),
            "q_gap_abs_drift": abs(current_gap - reference_gap) if comparable else None,
        }
    return report


def _snapshot_drift(reference: list[dict[str, Any]], current: list[dict[str, Any]]) -> dict[str, Any]:
    if len(reference) != len(current):
        raise ValueError("Reference и current anchor наборы имеют разную длину")
    q_errors: list[float] = []
    v_errors: list[float] = []
    regret_errors: list[float] = []
    sign_changes = 0
    legal_slots = 0
    q_stds: list[float] = []
    pair_rows: dict[str, list[dict[str, float | int | None]]] = {pair[0]: [] for pair in _ORDER_PAIRS}
    for reference_row, current_row in zip(reference, current, strict=True):
        mask = reference_row["mask"] > 0.0
        if not np.array_equal(mask, current_row["mask"] > 0.0):
            raise ValueError("Legal action mask anchor-state изменился")
        q_errors.extend(np.abs(current_row["q"][mask] - reference_row["q"][mask]).tolist())
        v_errors.append(abs(float(current_row["v"]) - float(reference_row["v"])))
        regret_errors.extend(np.abs(current_row["r"][mask] - reference_row["r"][mask]).tolist())
        sign_changes += int(np.count_nonzero((current_row["r"][mask] > 0.0) != (reference_row["r"][mask] > 0.0)))
        legal_slots += int(np.count_nonzero(mask))
        q_stds.extend(current_row["q_std"][mask].tolist())
        ordering = _ordering_drift(reference_row, current_row)
        for label, row in ordering.items():
            pair_rows[label].append(row)
    ordering_summary: dict[str, dict[str, float | int | None]] = {}
    for label, rows in pair_rows.items():
        comparable = [row for row in rows if row["comparable_pairs"]]
        ordering_summary[label] = {
            "legal_pairs": int(sum(int(row["legal_pairs"]) for row in rows)),
            "comparable_pairs": len(comparable),
            "ordering_preserved_rate": (
                float(np.mean([bool(row["ordering_preserved"]) for row in comparable])) if comparable else None
            ),
            "q_gap_abs_drift_mean": (
                float(np.mean([float(row["q_gap_abs_drift"]) for row in comparable])) if comparable else None
            ),
        }
    return {
        "q_mae": float(np.mean(q_errors)) if q_errors else 0.0,
        "v_mae": float(np.mean(v_errors)) if v_errors else 0.0,
        "regret_mae": float(np.mean(regret_errors)) if regret_errors else 0.0,
        "regret_sign_change_rate": float(sign_changes / legal_slots) if legal_slots else 0.0,
        "target_q_std_mean": float(np.mean(q_stds)) if q_stds else 0.0,
        "pairwise_ordering": ordering_summary,
    }


def run_target_drift_probe(
    *,
    agent: Any,
    coordinator: Any,
    iterations: int,
    traversals_per_player: int,
    anchor_states: int,
    anchor_repeats: int,
    anchor_roots: int,
    anchor_every: int,
    seed: int,
) -> dict[str, Any]:
    """Запускает короткое HU обучение и снимает target drift на fixed anchor states."""
    if min(iterations, traversals_per_player, anchor_states, anchor_repeats, anchor_roots, anchor_every) < 1:
        raise ValueError("Все размеры target drift probe должны быть положительными")
    from src.training import train as train_mod
    from tools.benchmark_traversal import seed_everything

    seed_everything(int(seed))
    before_anchor_collection = sum(len(buffer) for buffer in agent.hu_advantage_buffers)
    with _preserve_rng_state():
        with _without_probe_recording(coordinator):
            anchors = allin_probe._collect_candidate_states(
                coordinator, roots=int(anchor_roots), seed=int(seed), target_count=int(anchor_states),
                traverser=0, iteration=1,
            )
    if len(anchors) != int(anchor_states):
        raise RuntimeError(f"Собрано только {len(anchors)} из {anchor_states} anchor states; увеличьте --anchor-roots")
    baseline = _mean_targets_for_anchors(coordinator, anchors, repeats=int(anchor_repeats), iteration=1)
    after_anchor_collection = sum(len(buffer) for buffer in agent.hu_advantage_buffers)
    snapshots = [{
        "iteration": 0,
        "drift_from_initial": _snapshot_drift(baseline, baseline),
    }]
    for iteration in range(1, int(iterations) + 1):
        agent.iteration_count = int(iteration)
        train_mod._prepare_hu_current_policy_iteration(agent)
        coordinator.run_iteration(
            iteration=int(iteration),
            traversals_per_player=int(traversals_per_player),
            new_initial_state=lambda player_id, traversal_index: train_mod._new_hand(
                2,
                int(seed) + int(iteration) * 2 * int(traversals_per_player)
                + int(player_id) * int(traversals_per_player) + int(traversal_index),
            ),
            on_traversal_attempt=agent.record_traversal_attempt,
            on_traversal_success=agent.record_traversal_success,
            train_strategy_due=True,
        )
        if iteration % int(anchor_every) == 0 or iteration == int(iterations):
            current = _mean_targets_for_anchors(
                coordinator, anchors, repeats=int(anchor_repeats), iteration=int(iteration),
            )
            snapshots.append({
                "iteration": int(iteration),
                "drift_from_initial": _snapshot_drift(baseline, current),
            })
    return {
        "schema_version": 1,
        "anchors": {
            "states_requested": int(anchor_states),
            "states_collected": len(anchors),
            "repeats_per_snapshot": int(anchor_repeats),
            "roots_examined": int(anchor_roots),
            "probe_recorded_samples": int(after_anchor_collection - before_anchor_collection),
        },
        "training": {
            "iterations": int(iterations),
            "traversals_per_player": int(traversals_per_player),
            "anchor_every": int(anchor_every),
            "seed": int(seed),
        },
        "snapshots": snapshots,
    }


def run_config_probe(
    *, config_path: str | Path, output_path: str | Path, iterations: int,
    traversals_per_player: int, anchor_states: int, anchor_repeats: int,
    anchor_roots: int, anchor_every: int, seed: int, device: str = "cpu",
) -> dict[str, Any]:
    from src.core.deep_cfr import DeepCFRAgent
    from src.training import train as train_mod
    from src.utils import config as config_mod

    config_mod.load_config(config_path)
    agent = DeepCFRAgent(player_id=0, num_players=2, device=device)
    coordinator = train_mod._create_hu_current_policy_coordinator(agent)
    report = run_target_drift_probe(
        agent=agent, coordinator=coordinator, iterations=iterations,
        traversals_per_player=traversals_per_player, anchor_states=anchor_states,
        anchor_repeats=anchor_repeats, anchor_roots=anchor_roots,
        anchor_every=anchor_every, seed=seed,
    )
    report.update({"config": str(config_path), "read_only": False})
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(target)
    return report


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--output", required=True)
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--traversals-per-player", type=int, default=64)
    parser.add_argument("--anchor-states", type=int, default=16)
    parser.add_argument("--anchor-repeats", type=int, default=32)
    parser.add_argument("--anchor-roots", type=int, default=1024)
    parser.add_argument("--anchor-every", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20261001)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    report = run_config_probe(
        config_path=args.config, output_path=args.output, iterations=args.iterations,
        traversals_per_player=args.traversals_per_player, anchor_states=args.anchor_states,
        anchor_repeats=args.anchor_repeats, anchor_roots=args.anchor_roots,
        anchor_every=args.anchor_every, seed=args.seed, device=args.device,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
