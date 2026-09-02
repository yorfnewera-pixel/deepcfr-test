"""Сравнивает задержку одного production-батча advantage обучения на CPU."""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import torch

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.core.action_space import NUM_ACTIONS
from src.core.deep_cfr import DeepCFRAgent
from src.utils.config import load_config


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--threads", type=int, nargs="+", default=[6, 12])
    parser.add_argument("--batches", type=int, default=30)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20260824)
    return parser.parse_args()


def _build_agent(batch_size: int, seed: int) -> DeepCFRAgent:
    rng = np.random.default_rng(seed)
    agent = DeepCFRAgent(num_players=6, memory_size=batch_size, device="cpu")
    agent.iteration_count = 1
    states = rng.standard_normal((batch_size, agent.input_size), dtype=np.float32)
    regrets = rng.standard_normal((batch_size, NUM_ACTIONS), dtype=np.float32)
    masks = np.ones((batch_size, NUM_ACTIONS), dtype=np.float32)
    for state, regret, mask in zip(states, regrets, masks, strict=True):
        agent.advantage_buffer.add(state, regret, mask, iteration=1)
    return agent


def _measure(threads: int, batch_size: int, warmup: int, batches: int, seed: int) -> list[float]:
    if threads <= 0:
        raise ValueError("Число потоков должно быть положительным.")
    previous_threads = torch.get_num_threads()
    try:
        torch.set_num_threads(threads)
        torch.manual_seed(seed)
        agent = _build_agent(batch_size, seed)
        for _ in range(warmup):
            agent.train_advantage_network_multi(batch_size=batch_size, epochs=1)

        delays_ms = []
        for _ in range(batches):
            started = time.perf_counter()
            agent.train_advantage_network_multi(batch_size=batch_size, epochs=1)
            delays_ms.append((time.perf_counter() - started) * 1000.0)
        return delays_ms
    finally:
        torch.set_num_threads(previous_threads)


def main() -> int:
    args = _parse_args()
    if args.batches < 3 or args.warmup < 0:
        raise ValueError("batches должен быть не меньше 3, warmup — неотрицательным.")
    load_config()
    batch_size = int(DeepCFRAgent(num_players=6, memory_size=1).advantage_batch_size)
    print(f"Production advantage batch_size={batch_size}; warmup={args.warmup}; samples={args.batches}")

    results = []
    for threads in args.threads:
        delays_ms = _measure(threads, batch_size, args.warmup, args.batches, args.seed)
        median_ms = statistics.median(delays_ms)
        results.append((median_ms, threads, statistics.mean(delays_ms), min(delays_ms)))
        print(
            f"threads={threads}: median={median_ms:.3f} ms/batch, "
            f"mean={statistics.mean(delays_ms):.3f}, min={min(delays_ms):.3f}"
        )

    winner = min(results)
    print(f"BEST_THREADS={winner[1]} BEST_MEDIAN_MS={winner[0]:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
