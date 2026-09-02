"""Запускает paired EV: frozen blueprint против S6 runtime search."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.evaluation.blueprint_policy import FrozenBlueprintPolicy
from src.runtime_search.paired_evaluation import run_runtime_paired_evaluation
from src.runtime_search.policy import RuntimeSearchConfig


def parse_args() -> argparse.Namespace:
    """Разбирает параметры воспроизводимого paired EV прогона."""
    parser = argparse.ArgumentParser(description="Paired EV: blueprint против runtime search")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Путь к light checkpoint")
    parser.add_argument("--output", type=Path, required=True, help="Путь JSON-отчёта")
    parser.add_argument("--deals", type=int, default=8, help="Число deals на seed")
    parser.add_argument("--seeds", type=int, nargs="+", default=(107, 109), help="Независимые seeds")
    parser.add_argument("--particles", type=int, default=2, help="Particles на solver-root")
    parser.add_argument("--proposals", type=int, default=256, help="Reach-weighted belief proposals")
    parser.add_argument("--eta", type=float, default=10.0, help="MMDS eta")
    parser.add_argument("--min-ess-ratio", type=float, default=0.25, help="Минимальный ESS ratio")
    parser.add_argument("--min-ess", type=float, default=8.0, help="Минимальный абсолютный ESS")
    parser.add_argument(
        "--resample-ess-ratio",
        type=float,
        default=0.5,
        help="ESS ratio, ниже которого B1 делает resampling между решениями",
    )
    parser.add_argument("--min-root-gap-zscore", type=float, default=1.0, help="Минимальный z-score root signal")
    return parser.parse_args()


def main() -> None:
    """Загружает checkpoint только для чтения и сохраняет paired EV report."""
    args = parse_args()
    if not args.checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint не найден: {args.checkpoint}")

    blueprint = FrozenBlueprintPolicy.from_checkpoint(args.checkpoint, device="cpu")
    evaluation = run_runtime_paired_evaluation(
        blueprint,
        config=RuntimeSearchConfig(
            belief_proposal_count=args.proposals,
            belief_particles=args.particles,
            eta=args.eta,
            belief_min_ess=args.min_ess,
            belief_min_ess_ratio=args.min_ess_ratio,
            belief_resample_ess_ratio=args.resample_ess_ratio,
            min_root_gap_zscore=args.min_root_gap_zscore,
        ),
        seeds=tuple(args.seeds),
        num_deals=args.deals,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(evaluation.to_dict(), ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(f"OK: отчёт сохранён в {args.output}")
    for run in evaluation.runs:
        print(
            f"seed={run.seed}; samples={run.samples}; bb/100={run.bb_per_100:.6f}; "
            f"search={run.search_decisions}; reach_weighted={run.reach_weighted_decisions}; "
            f"fallback={run.blueprint_fallback_decisions}; "
            f"changes={run.action_changes}/{run.action_comparisons}; "
            f"l1_mean={run.policy_l1_sum / run.search_decisions if run.search_decisions else 0.0:.6f}; "
            f"l1_max={run.policy_l1_max:.6f}; argmax_changes={run.policy_argmax_changes}; "
            f"search_by_stage={run.search_decisions_by_stage}; "
            f"changes_by_stage={run.action_changes_by_stage}; "
            f"l1_by_stage={run.policy_l1_sum_by_stage}; "
            f"argmax_by_stage={run.policy_argmax_changes_by_stage}"
        )
    pooled = evaluation.pooled_statistics
    print(
        f"pooled: bb/100={pooled['bb_per_100']:.6f}; "
        f"se={pooled['standard_error_bb_per_100']:.6f}; "
        f"t={pooled['t_statistic']:.6f}; "
        f"ci95=[{pooled['ci95_low_bb_per_100']:.6f}, {pooled['ci95_high_bb_per_100']:.6f}]; "
        f"power={pooled['power_status']}"
    )


if __name__ == "__main__":
    main()
