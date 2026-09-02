#!/usr/bin/env python3
"""Построчный агрегатор логов обучения (без загрузки файла в память).

Зачем: логи прогонов на десятки итераций весят десятки-сотни КБ и целиком в контекст не
нужны. Скрипт читает поток построчно, собирает per-iteration метрики и печатает таблицу.

Ключевая метрика — **мкс/узел**, а не секунды/итерацию: #107 §2.8 п.3 показал разброс числа
узлов 6.6x при фиксированных traversals, поэтому сравнение по времени итерации бессмысленно.
Первые итерации отбрасываются как разогрев (#107 §2.8 п.1).

Использование:
    python tools/aggregate_run_log.py models/test107v8/123-3.txt
    python tools/aggregate_run_log.py log.txt --warmup 2 --csv out.csv
    python tools/aggregate_run_log.py a.txt b.txt --compare
"""
from __future__ import annotations

import argparse
import csv
import re
import statistics
import sys
from pathlib import Path
from typing import Any, Iterator

PATTERNS: dict[str, re.Pattern[str]] = {
    "traversals": re.compile(r"(\d+) traversals/iter"),
    "iteration": re.compile(r"Iteration (\d+)/"),
    "nodes": re.compile(
        r"nodes=(\d+).*?traversing_nodes=(\d+), opponent_nodes=(\d+)"
    ),
    "depth": re.compile(r"max_depth=(\d+), depth_hits=(\d+)"),
    "elapsed": re.compile(r"завершена за ([\d.]+)"),
    "budget": re.compile(
        r"iteration=([\d.]+)s traversal=([\d.]+)s profiled=([\d.]+)s "
        r"stages=([\d.]+)s unaccounted=([\d.]+)"
    ),
    "eval": re.compile(r"Средний профит: ([-\d.]+), raise_freq: ([\d.]+)"),
    "adv_loss": re.compile(r"Advantage loss: ([\d.]+)"),
    "sizing_loss": re.compile(r"Sizing anchor loss: ([\d.]+) \((\d+) steps\)"),
    "strategy_loss": re.compile(r"Strategy loss: ([\d.]+)"),
    "seed": re.compile(r"Seed: (\S+)"),
}


def parse_log(path: Path) -> tuple[dict[int, dict[str, Any]], dict[str, Any]]:
    """Читает лог построчно, возвращает (per_iteration, meta)."""
    rows: dict[int, dict[str, Any]] = {}
    meta: dict[str, Any] = {"path": str(path)}
    current: int | None = None

    with path.open(encoding="utf-8", errors="replace") as handle:
        for line in handle:
            match = PATTERNS["seed"].search(line)
            if match and "seed" not in meta:
                meta["seed"] = match.group(1)
            match = PATTERNS["traversals"].search(line)
            if match and "traversals" not in meta:
                meta["traversals"] = int(match.group(1))

            match = PATTERNS["iteration"].search(line)
            if match:
                current = int(match.group(1))
                rows.setdefault(current, {})
                continue
            if current is None:
                continue
            row = rows[current]

            match = PATTERNS["nodes"].search(line)
            if match:
                row["nodes"] = int(match.group(1))
                row["traversing_nodes"] = int(match.group(2))
                row["opponent_nodes"] = int(match.group(3))
            match = PATTERNS["depth"].search(line)
            if match:
                row["max_depth"] = int(match.group(1))
                row["depth_hits"] = int(match.group(2))
            match = PATTERNS["elapsed"].search(line)
            if match:
                row["elapsed_s"] = float(match.group(1))
            match = PATTERNS["budget"].search(line)
            if match:
                values = [float(x) for x in match.groups()]
                row["iteration_s"], row["traversal_s"] = values[0], values[1]
                row["profiled_s"], row["stages_s"] = values[2], values[3]
                row["unaccounted_s"] = values[4]
            match = PATTERNS["eval"].search(line)
            if match:
                row["profit"] = float(match.group(1))
                row["raise_freq"] = float(match.group(2))
            match = PATTERNS["adv_loss"].search(line)
            if match:
                row["adv_loss"] = float(match.group(1))
            match = PATTERNS["sizing_loss"].search(line)
            if match:
                row["sizing_loss"] = float(match.group(1))
                row["sizing_steps"] = int(match.group(2))
            match = PATTERNS["strategy_loss"].search(line)
            if match:
                row["strategy_loss"] = float(match.group(1))

    cumulative = 0
    for key in sorted(rows):
        row = rows[key]
        nodes = row.get("nodes")
        traversal = row.get("traversal_s") or row.get("elapsed_s")
        if nodes and traversal:
            row["us_per_node"] = traversal * 1e6 / nodes
        if row.get("traversal_s") and row.get("unaccounted_s") is not None:
            row["unaccounted_pct"] = 100.0 * row["unaccounted_s"] / row["traversal_s"]
        # Кумулятивные узлы — единственная ось, сравнимая между прогонами с разным
        # traversals_per_iteration (#107: iter 30 при K=250 ~ iter 8 при K=1000).
        if nodes:
            cumulative += nodes
            row["cumulative_nodes"] = cumulative
    return rows, meta


def _stat(values: list[float]) -> str:
    if not values:
        return "n/a"
    if len(values) == 1:
        return f"{values[0]:.1f}"
    return f"{statistics.mean(values):.1f} +/- {statistics.pstdev(values):.1f}"


def summarize(rows: dict[int, dict[str, Any]], meta: dict[str, Any], warmup: int) -> None:
    keys = sorted(rows)
    if not keys:
        print(f"{meta['path']}: итерации не найдены")
        return
    stable = [k for k in keys if k > warmup]

    print(f"файл       : {meta['path']}")
    print(f"seed       : {meta.get('seed', 'n/a')}")
    print(f"traversals : {meta.get('traversals', 'n/a')}  <- ось сравнения: кумулятивные узлы, не iter")
    print(f"итераций: {len(keys)} ({min(keys)}..{max(keys)}), разогрев отброшен: 1..{warmup}")
    print()
    header = (
        f"{'it':>3} {'nodes':>8} {'us/node':>8} {'trav_s':>7} {'unacc%':>7} "
        f"{'adv':>8} {'sizing':>8} {'stp':>4} {'profit':>7} {'rf':>6} {'depth':>5}"
    )
    print(header)
    print("-" * len(header))
    for key in keys:
        row = rows[key]
        marker = " " if key > warmup else "~"
        print(
            f"{key:>3}{marker}{row.get('nodes', 0):>7} "
            f"{row.get('us_per_node', float('nan')):>8.1f} "
            f"{row.get('traversal_s', float('nan')):>7.1f} "
            f"{row.get('unaccounted_pct', float('nan')):>7.1f} "
            f"{row.get('adv_loss', float('nan')):>8.4f} "
            f"{row.get('sizing_loss', float('nan')):>8.4f} "
            f"{row.get('sizing_steps', 0):>4} "
            f"{row.get('profit', float('nan')):>7.2f} "
            f"{row.get('raise_freq', float('nan')):>6.3f} "
            f"{row.get('max_depth', 0):>5}"
        )

    def collect(field: str) -> list[float]:
        return [rows[k][field] for k in stable if field in rows[k]]

    print()
    print("СТАБИЛЬНЫЕ ИТЕРАЦИИ (разогрев отброшен):")
    print(f"  us/node        : {_stat(collect('us_per_node'))}")
    print(f"  unaccounted %  : {_stat(collect('unaccounted_pct'))}")
    nodes = collect("nodes")
    if nodes:
        print(
            f"  nodes          : min={min(nodes):.0f} max={max(nodes):.0f} "
            f"разброс={max(nodes) / max(min(nodes), 1):.1f}x"
        )
    traversal = collect("traversal_s")
    if traversal:
        print(f"  traversal сумма: {sum(traversal):.1f} с")
    rf = collect("raise_freq")
    if len(rf) >= 2:
        print(f"  raise_freq     : {rf[0]:.3f} -> {rf[-1]:.3f}")
    profit = collect("profit")
    if profit:
        print(f"  profit         : {_stat(profit)}  (min {min(profit):.2f}, max {max(profit):.2f})")
    sizing = collect("sizing_loss")
    if len(sizing) >= 2:
        print(f"  sizing loss    : {sizing[0]:.4f} -> {sizing[-1]:.4f}")
        print(
            "    ВНИМАНИЕ: полный sizing-лосс = fresh_share x fresh_loss (#107 §2.8 п.17)."
            " Падение отражает падение доли свежих записей, не улучшение подгонки."
        )


def compare(reports: list[tuple[dict[int, dict[str, Any]], dict[str, Any]]], warmup: int) -> None:
    print()
    traversals = {meta.get("traversals") for _, meta in reports}
    if len(traversals) > 1:
        print("ОТКАЗ СРАВНИВАТЬ ПО НОМЕРУ ИТЕРАЦИИ: разное traversals_per_iteration "
              f"({sorted(str(t) for t in traversals)}).")
        print("  Причина: action-буфер чистится каждую итерацию, значит число шагов обучения")
        print("  пропорционально traversals. iter 30 при K=250 ~ iter 8 при K=1000 по данным.")
        print("  Сравнивайте на равных КУМУЛЯТИВНЫХ УЗЛАХ:")
        print()
        print(f"  {'прогон':<24}{'iter':>6}{'кум.узлов':>14}{'us/node':>10}")
        for rows, meta in reports:
            name = Path(meta["path"]).parent.name or Path(meta["path"]).stem
            for key in sorted(k for k in rows if k > warmup):
                row = rows[key]
                if "cumulative_nodes" in row:
                    print(f"  {name:<24}{key:>6}{row['cumulative_nodes']:>14}"
                          f"{row.get('us_per_node', float('nan')):>10.1f}")
        return
    print("СРАВНЕНИЕ (мкс/узел на общих итерациях, разогрев отброшен)")
    common = set.intersection(*[{k for k in rows if k > warmup} for rows, _ in reports])
    if not common:
        print("  общих стабильных итераций нет")
        return
    names = [Path(meta["path"]).parent.name or Path(meta["path"]).stem for _, meta in reports]
    print("  " + "it".rjust(4) + "".join(name.rjust(14) for name in names) + "   delta")
    for key in sorted(common):
        values = [rows[key].get("us_per_node") for rows, _ in reports]
        if any(v is None for v in values):
            continue
        delta = 100.0 * (values[-1] / values[0] - 1.0)
        print("  " + f"{key:>4}" + "".join(f"{v:>14.1f}" for v in values) + f"   {delta:>+6.1f}%")
    means = [
        statistics.mean([rows[k]["us_per_node"] for k in common if "us_per_node" in rows[k]])
        for rows, _ in reports
    ]
    print("  " + "mean".rjust(4) + "".join(f"{m:>14.1f}" for m in means)
          + f"   {100.0 * (means[-1] / means[0] - 1.0):>+6.1f}%")
    identical = all(
        rows[k].get("nodes") == reports[0][0][k].get("nodes")
        for rows, _ in reports
        for k in common
    )
    print(f"  число узлов идентично: {identical}"
          + ("" if identical else "  <- прогоны НЕ сравнимы напрямую"))


def write_csv(rows: dict[int, dict[str, Any]], path: Path) -> None:
    fields = [
        "iteration", "nodes", "cumulative_nodes", "traversing_nodes", "opponent_nodes",
        "us_per_node", "traversal_s", "unaccounted_s", "unaccounted_pct", "adv_loss",
        "sizing_loss", "sizing_steps", "strategy_loss", "profit", "raise_freq",
        "max_depth", "depth_hits",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for key in sorted(rows):
            writer.writerow({"iteration": key, **rows[key]})
    print(f"CSV записан: {path}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("logs", nargs="+", help="файлы логов прогонов")
    parser.add_argument("--warmup", type=int, default=2,
                        help="сколько первых итераций отбросить как разогрев (по умолчанию 2)")
    parser.add_argument("--csv", help="записать per-iteration метрики в CSV")
    parser.add_argument("--compare", action="store_true",
                        help="сравнить прогоны по мкс/узел на общих итерациях")
    args = parser.parse_args()

    reports = []
    for index, raw in enumerate(args.logs):
        path = Path(raw)
        if not path.is_file():
            print(f"файл не найден: {path}", file=sys.stderr)
            return 1
        if index:
            print()
            print("=" * 78)
        rows, meta = parse_log(path)
        summarize(rows, meta, args.warmup)
        reports.append((rows, meta))
        if args.csv and len(args.logs) == 1:
            write_csv(rows, Path(args.csv))

    if args.compare and len(reports) >= 2:
        compare(reports, args.warmup)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
