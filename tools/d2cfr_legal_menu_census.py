"""Read-only census legal action menus в on-policy HU rollout checkpoint-а."""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pokers as pkrs

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.core.action_space import ACTION_LABELS, ActionSlot, legal_action_mask
from tools.d2cfr_allin_bias_probe import _new_hand
from tools.d2cfr_objective_alignment_probe import _restore_agent


_HALF_POT = int(ActionSlot.RAISE_HALF_POT)
_POT = int(ActionSlot.RAISE_POT)
_ALL_IN = int(ActionSlot.ALL_IN)
_MENU_LABELS = (
    "all_three_raise_sizes",
    "half_pot_and_all_in_without_pot",
    "all_in_only_raise",
    "other",
)


def classify_raise_menu(mask: np.ndarray) -> str:
    """Классифицирует доступность трёх raise slots без интерпретации policy."""
    legal = np.asarray(mask, dtype=bool)
    if legal.shape != (len(ACTION_LABELS),):
        raise ValueError("legal mask должна содержать все action slots")
    half_pot, pot, all_in = legal[_HALF_POT], legal[_POT], legal[_ALL_IN]
    if half_pot and pot and all_in:
        return "all_three_raise_sizes"
    if half_pot and not pot and all_in:
        return "half_pot_and_all_in_without_pot"
    if not half_pot and not pot and all_in:
        return "all_in_only_raise"
    return "other"


def _empty_menu_counts() -> dict[str, int]:
    return {label: 0 for label in _MENU_LABELS}


def _empty_policy_mass() -> dict[str, float]:
    return {label: 0.0 for label in ACTION_LABELS}


def _summarise_selection(visits: Iterable[dict[str, Any]]) -> dict[str, Any]:
    rows = list(visits)
    menu_counts = _empty_menu_counts()
    sampled_actions: Counter[str] = Counter()
    policy_sum = np.zeros(len(ACTION_LABELS), dtype=np.float64)
    half_pot_legal = 0
    legacy_all_in_only = 0
    for row in rows:
        mask = np.asarray(row["mask"], dtype=bool)
        policy = np.asarray(row["policy"], dtype=np.float64)
        sampled_slot = int(row["sampled_slot"])
        if mask.shape != (len(ACTION_LABELS),) or policy.shape != mask.shape:
            raise ValueError("visit содержит несовместимые mask и policy")
        if not np.all(np.isfinite(policy)) or np.any(policy < 0.0):
            raise ValueError("visit policy должна быть конечной и неотрицательной")
        if not mask.any() or not np.isclose(policy[mask].sum(), 1.0, atol=1e-6):
            raise ValueError("visit policy должна быть нормирована по legal actions")
        if sampled_slot < 0 or sampled_slot >= len(ACTION_LABELS) or not mask[sampled_slot]:
            raise ValueError("sampled slot должен быть legal")
        menu = classify_raise_menu(mask)
        menu_counts[menu] += 1
        sampled_actions[ACTION_LABELS[sampled_slot]] += 1
        policy_sum += policy
        if mask[_HALF_POT]:
            half_pot_legal += 1
        if menu == "half_pot_and_all_in_without_pot":
            legacy_all_in_only += 1
    count = len(rows)
    return {
        "visits": count,
        "menu_counts": menu_counts,
        "half_pot_legal_visits": half_pot_legal,
        "legacy_all_in_only_raise_visits": legacy_all_in_only,
        "policy_mass_mean": {
            label: float(policy_sum[index] / count) if count else 0.0
            for index, label in enumerate(ACTION_LABELS)
        },
        "sampled_action_counts": {
            label: int(sampled_actions[label]) for label in ACTION_LABELS
        },
    }


def summarise_visits(visits: Iterable[dict[str, Any]]) -> dict[str, Any]:
    """Сводит все посещения, после рейза и по улицам.

    ``legacy_all_in_only_raise_visits`` — точный counterfactual старого
    запрета: half-pot был legal после рейза, pot уже нет, all-in legal.
    """
    rows = list(visits)
    after_raise = [row for row in rows if bool(row["after_raise"])]
    streets = sorted({str(row["street"]) for row in rows})
    return {
        "overall": _summarise_selection(rows),
        "after_raise": _summarise_selection(after_raise),
        "by_street": {
            street: _summarise_selection(row for row in rows if str(row["street"]) == street)
            for street in streets
        },
    }


def _after_raise_on_current_street(state: Any) -> bool:
    return any(
        getattr(player, "last_stage_action", None) == pkrs.ActionEnum.Raise
        for player in state.players_state
    )


def _street_name(state: Any) -> str:
    return str(state.stage).split(".")[-1]


def _collect_visits(coordinator: Any, *, hands: int, seed: int, max_decisions_per_hand: int) -> list[dict[str, Any]]:
    if hands <= 0 or max_decisions_per_hand <= 0:
        raise ValueError("hands и max_decisions_per_hand должны быть положительными")
    rng = np.random.default_rng(seed)
    visits: list[dict[str, Any]] = []
    coordinator.begin_iteration()
    for hand_index in range(hands):
        state = _new_hand(seed + hand_index)
        for _ in range(max_decisions_per_hand):
            if coordinator.adapter.is_terminal(state):
                break
            actor = int(coordinator.adapter.current_player(state))
            mask = np.asarray(legal_action_mask(state), dtype=np.float32)
            encoded = coordinator.adapter.encode(state, actor)
            policy = coordinator._policy_from_snapshot(actor, encoded, mask)
            legal_slots = np.flatnonzero(mask).astype(int)
            sampled_slot = int(rng.choice(legal_slots, p=policy[legal_slots]))
            visits.append({
                "street": _street_name(state),
                "after_raise": _after_raise_on_current_street(state),
                "mask": mask,
                "policy": policy,
                "sampled_slot": sampled_slot,
            })
            state = coordinator.adapter.apply(state, sampled_slot)
        else:
            raise RuntimeError(f"Rollout раздачи {hand_index} превысил max_decisions_per_hand")
    return visits


def run_probe(
    checkpoint_path: str | Path,
    *,
    config_path: str | Path,
    output_path: str | Path,
    hands: int,
    seed: int,
    device: str = "cpu",
    max_decisions_per_hand: int = 128,
) -> dict[str, Any]:
    """Строит отчёт без optimizer, traversal targets и изменений checkpoint-а."""
    agent, checkpoint = _restore_agent(checkpoint_path, config_path=config_path, device=device)
    coordinator = agent.hu_coordinator
    visits = _collect_visits(
        coordinator,
        hands=hands,
        seed=seed,
        max_decisions_per_hand=max_decisions_per_hand,
    )
    report = {
        "schema_version": 1,
        "read_only": True,
        "checkpoint": str(checkpoint_path),
        "checkpoint_iteration": int(checkpoint["iteration"]),
        "policy_source": "checkpoint HU advantage nets with regret matching",
        "hands": int(hands),
        "seed": int(seed),
        "max_decisions_per_hand": int(max_decisions_per_hand),
        "summary": summarise_visits(visits),
    }
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(target)
    return report


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, help="Путь к полному HU checkpoint")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--output", required=True)
    parser.add_argument("--hands", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20260927)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--max-decisions-per-hand", type=int, default=128)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    run_probe(
        args.checkpoint,
        config_path=args.config,
        output_path=args.output,
        hands=args.hands,
        seed=args.seed,
        device=args.device,
        max_decisions_per_hand=args.max_decisions_per_hand,
    )
    print(f"Отчёт сохранён: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
