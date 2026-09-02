"""Сохраняет реальные закодированные входы production traversal для offline A/B."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

from tools.benchmark_traversal import _load_agent, load_root_bank, make_state, seed_everything
from src.utils.traversal_profiler import TRAVERSAL_PROFILER, set_traversal_profiler_level


class InputCaptureModule(nn.Module):
    """Прозрачная обёртка сети; копирует не более лимита входов для benchmark."""

    def __init__(self, module: nn.Module, limit: int) -> None:
        super().__init__()
        self.module = module
        self.limit = limit
        self.inputs: list[np.ndarray] = []

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if len(self.inputs) < self.limit:
            remaining = self.limit - len(self.inputs)
            captured = inputs.detach().cpu().numpy()[:remaining].copy()
            self.inputs.extend(captured)
        return self.module(inputs)

    def as_array(self) -> np.ndarray:
        if not self.inputs:
            return np.empty((0, 0), dtype=np.float32)
        return np.asarray(self.inputs, dtype=np.float32)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--root-bank", required=True, type=Path)
    parser.add_argument("--roots", type=int, default=10)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--iteration", type=int, default=1)
    parser.add_argument("--max-states", type=int, default=100_000)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.max_states <= 0:
        raise ValueError("max-states должно быть положительным.")
    bank = load_root_bank(args.root_bank)
    if len(bank) < args.roots:
        raise ValueError("В root bank меньше roots, чем запрошено.")

    agent = _load_agent(args.checkpoint, "cpu", 6)
    advantage = InputCaptureModule(agent.advantage_net, args.max_states)
    original_advantage = agent.advantage_net
    agent.advantage_net = advantage

    set_traversal_profiler_level("off")
    TRAVERSAL_PROFILER.reset()
    agent.prepare_iteration(args.iteration, 0)
    agent.reset_traversal_stats()
    seed_everything(args.seed)
    threads_before = torch.get_num_threads()
    try:
        torch.set_num_threads(1)
        for root in bank[:args.roots]:
            agent.cfr_traverse_multi(make_state(root), args.iteration, 0)
    finally:
        torch.set_num_threads(threads_before)
        agent.advantage_net = original_advantage

    arrays = {
        "advantage": advantage.as_array(),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **arrays)
    metadata = {
        "roots": args.roots,
        "max_states": args.max_states,
        "counts": {name: int(values.shape[0]) for name, values in arrays.items()},
        "shapes": {name: list(values.shape) for name, values in arrays.items()},
    }
    args.output.with_suffix(".json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(metadata, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
