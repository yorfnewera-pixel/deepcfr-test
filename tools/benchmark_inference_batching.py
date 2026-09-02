"""Offline сравнение batch=1 и batched inference на captured traversal inputs."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from tools.benchmark_traversal import _load_agent, current_rss_mb


def parse_batches(value: str) -> list[int]:
    batches = [int(item) for item in value.split(",")]
    if not batches or any(batch <= 0 for batch in batches):
        raise argparse.ArgumentTypeError("batches должны быть положительными integer.")
    return batches


def infer(model: torch.nn.Module, inputs: np.ndarray, batch_size: int) -> tuple[np.ndarray, float]:
    outputs = []
    started_at = time.perf_counter()
    with torch.inference_mode():
        for offset in range(0, len(inputs), batch_size):
            batch = torch.from_numpy(inputs[offset:offset + batch_size])
            outputs.append(model(batch).cpu().numpy())
    return np.concatenate(outputs, axis=0), time.perf_counter() - started_at


def benchmark_model(model: torch.nn.Module, inputs: np.ndarray, batches: list[int]) -> list[dict[str, float | int]]:
    if len(inputs) == 0:
        return []
    # Не учитываем запуск allocator/dispatcher в измерении.
    infer(model, inputs[:min(len(inputs), max(batches))], min(max(batches), len(inputs)))
    reference, _ = infer(model, inputs, 1)
    results = []
    for batch_size in batches:
        rss_before = current_rss_mb()
        outputs, elapsed = infer(model, inputs, batch_size)
        samples = len(inputs)
        results.append({
            "batch": batch_size,
            "samples": samples,
            "wall_seconds": elapsed,
            "samples_per_second": samples / elapsed if elapsed else 0.0,
            "microseconds_per_sample": elapsed * 1_000_000 / samples if samples else 0.0,
            "peak_rss_mb": max(rss_before, current_rss_mb()),
            "max_abs_diff_vs_batch1": float(np.max(np.abs(outputs - reference))),
        })
    batch1 = next((item for item in results if item["batch"] == 1), None)
    if batch1 is not None:
        for item in results:
            item["speedup_vs_batch1"] = (
                float(batch1["microseconds_per_sample"]) / float(item["microseconds_per_sample"])
                if item["microseconds_per_sample"] else 0.0
            )
    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--states", required=True, type=Path)
    parser.add_argument("--batches", type=parse_batches, default=[1, 8, 32, 128, 512, 1024])
    parser.add_argument("--torch-threads", type=int, default=1)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.torch_threads <= 0:
        raise ValueError("torch-threads должно быть положительным.")
    captured = np.load(args.states)
    agent = _load_agent(args.checkpoint, "cpu", 6)
    models = {
        "advantage": (agent.advantage_net, captured["advantage"]),
        "strategy": (agent.strategy_net, captured["advantage"]),
    }
    previous_threads = torch.get_num_threads()
    try:
        torch.set_num_threads(args.torch_threads)
        results = {
            name: benchmark_model(model, inputs.astype(np.float32, copy=False), args.batches)
            for name, (model, inputs) in models.items()
        }
    finally:
        torch.set_num_threads(previous_threads)
    payload = {
        "batches": args.batches,
        "torch_threads": args.torch_threads,
        "networks": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
