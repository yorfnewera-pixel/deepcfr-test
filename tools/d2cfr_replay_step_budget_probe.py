"""Read-only A/B: хватает ли D2CFR fixed optimizer steps для сохранённого replay."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.core.buffers import DuelingAdvantageBuffer
from src.core.deep_cfr import DeepCFRAgent
from src.training import train as train_mod
from src.utils import config as config_mod
from tools import d2cfr_accumulation_probe as accumulation_probe
from tools import d2cfr_signal_probe as signal_probe


def group_holdout_split(
    states: np.ndarray,
    masks: np.ndarray,
    *,
    holdout_fraction: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Делит replay по exact infoset, не допуская утечки дублей в holdout."""
    states = np.asarray(states, dtype=np.float32)
    masks = np.asarray(masks, dtype=np.float32)
    if states.ndim != 2 or masks.shape != (len(states), 6):
        raise ValueError("states и masks имеют несовместимые формы")
    if not 0.0 < float(holdout_fraction) < 1.0:
        raise ValueError("holdout_fraction должен быть строго между 0 и 1")
    groups: dict[bytes, list[int]] = defaultdict(list)
    for index, (state, mask) in enumerate(zip(states, masks, strict=True)):
        groups[state.tobytes() + mask.tobytes()].append(index)
    keys = list(groups)
    if len(keys) < 2:
        raise ValueError("Для group holdout нужны минимум два различных encoded infoset")
    rng = np.random.default_rng(int(seed))
    rng.shuffle(keys)
    holdout_groups = max(1, min(len(keys) - 1, round(len(keys) * float(holdout_fraction))))
    holdout = np.asarray(sorted(index for key in keys[:holdout_groups] for index in groups[key]), dtype=np.int64)
    train = np.asarray(sorted(index for key in keys[holdout_groups:] for index in groups[key]), dtype=np.int64)
    return train, holdout


def _batch_from_buffer(
    buffer: DuelingAdvantageBuffer, indices: np.ndarray, device: str
) -> signal_probe.ProbeBatch:
    samples = [
        (
            buffer._states[index].copy(),
            buffer._action_values[index].copy(),
            np.float32(buffer._state_values[index]),
            buffer._regrets[index].copy(),
            buffer._masks[index].copy(),
            np.float32(buffer._iterations[index]),
        )
        for index in indices
    ]
    return signal_probe.build_probe_batch(samples, device=device)


def _run_leg(
    *,
    agent: DeepCFRAgent,
    buffer: DuelingAdvantageBuffer,
    short_steps: int,
    long_steps: int,
    batch_size: int,
    holdout_fraction: float,
    seed: int,
    device: str,
) -> dict[str, object]:
    count = len(buffer)
    if count < 2:
        raise ValueError("Replay buffer должен содержать минимум две записи")
    train_indices, holdout_indices = group_holdout_split(
        buffer._states[:count], buffer._masks[:count],
        holdout_fraction=holdout_fraction, seed=seed,
    )
    train_batch = _batch_from_buffer(buffer, train_indices, device)
    holdout_batch = _batch_from_buffer(buffer, holdout_indices, device)
    shared = {
        "network_factory": agent._new_advantage_network,
        "train_batch": train_batch,
        "evaluation_batch": holdout_batch,
        "learning_rate": float(agent.advantage_lr),
        "state_value_loss_weight": float(agent.d2cfr_state_value_loss_weight),
        "seed": int(seed),
        "max_grad_norm": 1.0 if agent.d2cfr_loss_mode == "anchored" else None,
        "iteration_weighted": agent.d2cfr_iteration_weight_mode == "batch_mean_1",
    }

    def fit(steps: int) -> dict[str, object]:
        index_batches = accumulation_probe._random_index_batches(
            sample_count=int(train_batch.states.shape[0]), batch_size=int(batch_size),
            steps=int(steps), seed=int(seed), device=train_batch.states.device,
        )
        return accumulation_probe._fit_d2cfr_arm(
            name=f"steps_{steps}", index_batches=index_batches, **shared
        )

    return {
        "replay_samples": count,
        "train_samples": int(train_batch.states.shape[0]),
        "holdout_samples": int(holdout_batch.states.shape[0]),
        "unique_infosets": int(len({
            buffer._states[index].tobytes() + buffer._masks[index].tobytes()
            for index in range(count)
        })),
        "arms": {f"steps_{short_steps}": fit(short_steps), f"steps_{long_steps}": fit(long_steps)},
    }


def run_probe(
    checkpoint_path: str | Path,
    *,
    config_path: str | Path,
    output_path: str | Path,
    short_steps: int = 750,
    long_steps: int = 3_000,
    batch_size: int = 256,
    holdout_fraction: float = 0.2,
    seed: int = 0,
    device: str = "cpu",
) -> dict[str, object]:
    """Загружает checkpoint и обучает временные сети без записи в checkpoint/replay."""
    if min(int(short_steps), int(long_steps), int(batch_size)) < 1:
        raise ValueError("short_steps, long_steps и batch_size должны быть положительными")
    if int(long_steps) <= int(short_steps):
        raise ValueError("long_steps должен быть больше short_steps")
    config_mod.load_config(config_path)
    agent = DeepCFRAgent(player_id=0, num_players=2, device=device)
    train_mod._create_hu_current_policy_coordinator(agent)
    checkpoint = train_mod._load_hu_checkpoint(agent, checkpoint_path)
    if not agent.d2cfr_enabled or agent.d2cfr_loss_function != "mse":
        raise ValueError("Probe поддерживает только HU D2CFR с MSE loss")

    report = {
        "schema_version": 1,
        "read_only": True,
        "checkpoint": str(checkpoint_path),
        "checkpoint_iteration": int(checkpoint["iteration"]),
        "split": {
            "unit": "exact encoded state plus legal mask",
            "holdout_fraction": float(holdout_fraction),
            "seed": int(seed),
        },
        "training_contract": {
            "batch_size": int(batch_size),
            "short_steps": int(short_steps),
            "long_steps": int(long_steps),
            "iteration_weight_mode": str(agent.d2cfr_iteration_weight_mode),
            "state_value_loss_weight": float(agent.d2cfr_state_value_loss_weight),
            "gradient_clip_norm": 1.0 if agent.d2cfr_loss_mode == "anchored" else None,
        },
        "players": {
            f"P{player_id}": _run_leg(
                agent=agent, buffer=buffer, short_steps=short_steps, long_steps=long_steps,
                batch_size=batch_size, holdout_fraction=holdout_fraction,
                seed=seed + player_id, device=device,
            )
            for player_id, buffer in enumerate(agent.hu_advantage_buffers)
        },
    }
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(target)
    return report


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--output", required=True)
    parser.add_argument("--short-steps", type=int, default=750)
    parser.add_argument("--long-steps", type=int, default=3000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--holdout-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=20260930)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    report = run_probe(
        args.checkpoint, config_path=args.config, output_path=args.output,
        short_steps=args.short_steps, long_steps=args.long_steps,
        batch_size=args.batch_size, holdout_fraction=args.holdout_fraction,
        seed=args.seed, device=args.device,
    )
    print(f"Отчёт сохранён: {args.output}")
    for player, result in report["players"].items():
        assert isinstance(result, dict)
        arms = result["arms"]
        assert isinstance(arms, dict)
        print(
            f"{player}: replay={result['replay_samples']}, train={result['train_samples']}, "
            f"holdout={result['holdout_samples']}, arms={', '.join(arms)}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
