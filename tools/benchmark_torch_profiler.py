"""Короткий CPU trace PyTorch-операторов внутри production traversal.

Этот инструмент диагностический: его wall-clock нельзя использовать как
production baseline из-за overhead torch.profiler.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from tools.benchmark_traversal import (
    _load_agent,
    load_root_bank,
    make_state,
    seed_everything,
)
from src.utils.traversal_profiler import TRAVERSAL_PROFILER, set_traversal_profiler_level


def profile_options(mode: str) -> dict[str, bool]:
    if mode not in {"time", "shapes", "memory", "stacks"}:
        raise ValueError(f"Неизвестный режим torch.profiler: {mode}")
    return {
        "record_shapes": mode == "shapes",
        "profile_memory": mode == "memory",
        "with_stack": mode == "stacks",
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--root-bank", required=True, type=Path)
    parser.add_argument("--roots", type=int, default=10)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--iteration", type=int, default=1)
    parser.add_argument("--mode", choices=("time", "shapes", "memory", "stacks"), default="time")
    parser.add_argument("--skip-first", type=int, default=3)
    parser.add_argument("--wait", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--active", type=int, default=1)
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    bank = load_root_bank(args.root_bank)
    if len(bank) < args.roots:
        raise ValueError("В root bank меньше roots, чем запрошено.")
    if args.roots < args.skip_first + args.wait + args.warmup + args.active:
        raise ValueError("Для schedule требуется больше roots.")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    options = profile_options(args.mode)
    agent = _load_agent(args.checkpoint, "cpu", 6)
    agent.prepare_iteration(args.iteration, 0)
    agent.reset_traversal_stats()
    set_traversal_profiler_level("off")
    TRAVERSAL_PROFILER.reset()
    seed_everything(args.seed)

    trace_path = args.output_dir / f"torch_traversal_{args.mode}_trace.json"

    def on_trace_ready(profiler: torch.profiler.profile) -> None:
        profiler.export_chrome_trace(str(trace_path))

    threads_before = torch.get_num_threads()
    try:
        torch.set_num_threads(1)
        with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU],
            schedule=torch.profiler.schedule(
                skip_first=args.skip_first,
                wait=args.wait,
                warmup=args.warmup,
                active=args.active,
                repeat=1,
            ),
            on_trace_ready=on_trace_ready,
            **options,
        ) as profiler:
            for root in bank[:args.roots]:
                with torch.profiler.record_function("traversal/root"):
                    state = make_state(root)
                    agent.cfr_traverse_multi(state, args.iteration, 0)
                profiler.step()
    finally:
        torch.set_num_threads(threads_before)

    summary = {
        "mode": args.mode,
        "roots": args.roots,
        "schedule": {
            "skip_first": args.skip_first,
            "wait": args.wait,
            "warmup": args.warmup,
            "active": args.active,
        },
        "trace": str(trace_path),
        "self_cpu_time_total": profiler.key_averages().table(
            sort_by="self_cpu_time_total", row_limit=30),
        "cpu_time_total": profiler.key_averages().table(
            sort_by="cpu_time_total", row_limit=30),
    }
    if args.mode == "shapes":
        summary["by_input_shape"] = profiler.key_averages(group_by_input_shape=True).table(
            sort_by="self_cpu_time_total", row_limit=30)
    summary_path = args.output_dir / f"torch_traversal_{args.mode}_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(summary_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
