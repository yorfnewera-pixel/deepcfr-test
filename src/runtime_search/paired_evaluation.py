"""Paired EV runner для frozen blueprint и S6 runtime policy."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Protocol, Sequence

import numpy as np
import pokers as pkrs

from src.evaluation.paired_harness import evaluate_paired
from src.runtime_search.history import BlueprintActionPolicy, RuntimeSearchPolicy
from src.runtime_search.policy import RuntimeSearchConfig


class RuntimeEvaluationBlueprint(BlueprintActionPolicy, Protocol):
    """Полный blueprint contract для paired runtime evaluation."""


@dataclass(frozen=True)
class RuntimePairedSeedResult:
    """Paired EV и факт использования solver-а для одного seed."""

    seed: int
    samples: int
    mean_difference: float
    std_difference: float
    bb_per_100: float
    search_decisions: int
    reach_weighted_decisions: int
    blueprint_fallback_decisions: int
    action_comparisons: int
    action_changes: int
    search_decisions_by_stage: dict[str, int]
    action_changes_by_stage: dict[str, int]
    policy_l1_sum: float
    policy_l1_max: float
    policy_argmax_changes: int
    policy_l1_sum_by_stage: dict[str, float]
    policy_argmax_changes_by_stage: dict[str, int]
    root_diagnostics: tuple[dict[str, object], ...] = ()


@dataclass(frozen=True)
class RuntimePairedEvaluation:
    """Результаты одинакового paired schedule на нескольких независимых seed."""

    runs: tuple[RuntimePairedSeedResult, ...]

    @property
    def total_samples(self) -> int:
        return sum(run.samples for run in self.runs)

    @property
    def pooled_statistics(self) -> dict[str, float | str]:
        """Возвращает pooled paired uncertainty по всем независимым seed."""
        total_samples = self.total_samples
        if total_samples < 2:
            raise ValueError("Для pooled statistics нужны минимум два samples")
        mean = sum(run.mean_difference * run.samples for run in self.runs) / total_samples
        variance = sum(
            (run.samples - 1) * run.std_difference**2
            + run.samples * (run.mean_difference - mean) ** 2
            for run in self.runs
        ) / (total_samples - 1)
        standard_error = math.sqrt(variance / total_samples)
        bb_scale = 100.0 / 2.0
        t_statistic = mean / standard_error if standard_error > 0.0 else 0.0
        return {
            "mean_difference": mean,
            "std_difference": math.sqrt(variance),
            "standard_error": standard_error,
            "bb_per_100": mean * bb_scale,
            "standard_error_bb_per_100": standard_error * bb_scale,
            "t_statistic": t_statistic,
            "ci95_low_bb_per_100": (mean - 1.96 * standard_error) * bb_scale,
            "ci95_high_bb_per_100": (mean + 1.96 * standard_error) * bb_scale,
            "power_status": "reported" if total_samples >= 10_000 else "insufficient_power",
        }

    def to_dict(self) -> dict:
        return {
            "runs": [asdict(run) for run in self.runs],
            "total_samples": self.total_samples,
            "pooled_statistics": self.pooled_statistics,
        }


def run_runtime_paired_evaluation(
    blueprint: RuntimeEvaluationBlueprint,
    *,
    config: RuntimeSearchConfig,
    seeds: Sequence[int],
    num_deals: int,
    num_players: int = 6,
    rotate_seats: bool = True,
    bb: float = 2.0,
) -> RuntimePairedEvaluation:
    """Сравнивает runtime policy с тем же frozen blueprint на фиксированных seeds."""
    if not seeds:
        raise ValueError("Нужен хотя бы один seed")
    if len(set(seeds)) != len(seeds):
        raise ValueError("Seeds должны быть уникальными")

    runs: list[RuntimePairedSeedResult] = []
    for seed in seeds:
        candidate = RuntimeSearchPolicy(hero_id=0, blueprint=blueprint, config=config)
        evaluation = evaluate_paired(
            blueprint,
            candidate,
            num_deals=num_deals,
            seed=int(seed),
            num_players=num_players,
            rotate_seats=rotate_seats,
            bb=bb,
        )
        runs.append(RuntimePairedSeedResult(
            seed=int(seed),
            samples=evaluation.samples,
            mean_difference=evaluation.mean_difference,
            std_difference=evaluation.std_difference,
            bb_per_100=evaluation.bb_per_100,
            search_decisions=candidate.search_decision_count,
            reach_weighted_decisions=candidate.reach_weighted_decision_count,
            blueprint_fallback_decisions=candidate.blueprint_fallback_decision_count,
            action_comparisons=candidate.action_comparison_count,
            action_changes=candidate.action_change_count,
            search_decisions_by_stage=dict(candidate.search_decisions_by_stage),
            action_changes_by_stage=dict(candidate.action_changes_by_stage),
            policy_l1_sum=candidate.policy_l1_sum,
            policy_l1_max=candidate.policy_l1_max,
            policy_argmax_changes=candidate.policy_argmax_change_count,
            policy_l1_sum_by_stage=dict(candidate.policy_l1_sum_by_stage),
            policy_argmax_changes_by_stage=dict(candidate.policy_argmax_changes_by_stage),
            root_diagnostics=tuple(_root_diagnostic(decision) for decision in candidate.search_decisions),
        ))
    return RuntimePairedEvaluation(runs=tuple(runs))


def _root_diagnostic(decision) -> dict[str, object]:
    prior = decision.root_blueprint_policy
    positive = prior[prior > 0.0]
    raw_values = decision.raw_ev[np.isfinite(decision.raw_ev)]
    return {
        "action_label": decision.action_label,
        "prior_log_ratio": float(np.log(positive.max() / positive.min())) if positive.size > 1 else 0.0,
        "raw_ev_spread": float(raw_values.max() - raw_values.min()) if raw_values.size else None,
        "policy_l1": float(np.abs(decision.root_search_policy - prior).sum()),
        "belief_ess": _finite_or_none(decision.belief_ess),
        "belief_ess_ratio": _finite_or_none(decision.belief_ess_ratio),
        "belief_resample_count": decision.belief_resample_count,
        "best_gap": _finite_or_none(decision.best_gap),
        "best_gap_se": _finite_or_none(decision.best_gap_se),
        "best_gap_zscore": _finite_or_none(decision.best_gap_zscore),
        "value_scale": _finite_or_none(decision.value_scale),
        "eta_effective": _finite_or_none(decision.eta_effective),
        "flags": list(decision.diagnostic_flags),
    }


def _finite_or_none(value: float | None) -> float | None:
    """Не допускает NaN и inf в строго JSON-совместимый отчёт."""
    if value is None or not math.isfinite(float(value)):
        return None
    return float(value)


__all__ = [
    "RuntimePairedEvaluation",
    "RuntimePairedSeedResult",
    "run_runtime_paired_evaluation",
]
