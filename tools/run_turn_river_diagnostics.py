"""CLI для S5 диагностики turn/river light checkpoint."""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.evaluation.blueprint_policy import FrozenBlueprintPolicy
from src.runtime_search.diagnostics import collect_turn_river_diagnostics


def parse_args() -> argparse.Namespace:
    """Разбирает параметры короткой воспроизводимой диагностики."""
    parser = argparse.ArgumentParser(description="Диагностика turn/river для light checkpoint")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Путь к light checkpoint")
    parser.add_argument("--output", type=Path, required=True, help="Путь JSON-отчёта")
    parser.add_argument("--deals", type=int, default=16, help="Число естественных раздач")
    parser.add_argument("--particles", type=int, default=2, help="Число particles для S4 probe")
    parser.add_argument("--seed", type=int, default=107, help="Фиксированный seed")
    return parser.parse_args()


def main() -> None:
    """Загружает checkpoint только для чтения и сохраняет диагностику."""
    args = parse_args()
    if not args.checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint не найден: {args.checkpoint}")

    blueprint = FrozenBlueprintPolicy.from_checkpoint(args.checkpoint, device="cpu")
    report = collect_turn_river_diagnostics(
        blueprint,
        num_deals=args.deals,
        belief_particles=args.particles,
        seed=args.seed,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report.to_dict(), ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    logging.info("Отчёт сохранён: %s", args.output)
    print(f"OK: отчёт сохранён в {args.output}")
    print(
        "Natural turn/river decisions: "
        f"{report.natural.streets['turn'].decisions}/"
        f"{report.natural.streets['river'].decisions}; "
        f"completed deals={report.natural.completed_deals}; "
        f"failed deals={report.natural.failed_deals}"
    )
    print(
        f"Search probe: action={report.search_probe.action_label}; "
        f"legal={report.search_probe.action_is_legal}; "
        f"latency_ms={report.search_probe.latency_milliseconds:.3f}"
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    main()
