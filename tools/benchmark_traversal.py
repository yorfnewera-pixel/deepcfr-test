"""Воспроизводимый benchmark production traversal без запуска обучения.

Скрипт намеренно не меняет конфигурацию CFR, checkpoint и параметры сетей.
Traversal заполняет только буферы процесса, которые уничтожаются при завершении
процесса; методы обучения и сохранения checkpoint здесь не вызываются.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

import numpy as np
import torch

import pokers as pkrs
from src.core.deep_cfr import DeepCFRAgent
from src.utils.traversal_profiler import TRAVERSAL_PROFILER, set_traversal_profiler_level


PROFILE_LEVELS = ("off", "coarse", "full")
REQUIRED_METRICS = (
    "mode", "roots", "nodes", "terminal_nodes", "decision_nodes",
    "traverser_nodes", "opponent_nodes", "wall_seconds", "nodes_per_root",
    "nodes_per_second", "decisions_per_second", "microseconds_per_node",
    "rss_before_mb", "rss_after_mb", "profile",
)


@dataclass(frozen=True)
class RootSeed:
    """Минимальные данные, достаточные для восстановления стартовой раздачи."""

    button: int
    deal_seed: int


def seed_everything(seed: int) -> None:
    """Фиксирует все RNG, которые используются production traversal."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def generate_root_bank(roots: int, seed: int, num_players: int = 6) -> list[RootSeed]:
    if roots <= 0:
        raise ValueError("Количество roots должно быть положительным.")
    if num_players <= 1:
        raise ValueError("Для benchmark требуется как минимум два игрока.")

    generator = random.Random(seed)
    return [
        RootSeed(
            button=generator.randrange(num_players),
            deal_seed=generator.randrange(0, 1_000_001),
        )
        for _ in range(roots)
    ]


def save_root_bank(path: Path, bank: Sequence[RootSeed]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps([asdict(root) for root in bank], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def load_root_bank(path: Path) -> list[RootSeed]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise ValueError("Банк roots должен быть JSON-массивом.")
    try:
        bank = [RootSeed(button=int(item["button"]), deal_seed=int(item["deal_seed"])) for item in raw]
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("Каждый root должен содержать integer-поля button и deal_seed.") from error
    if not bank:
        raise ValueError("Банк roots пуст.")
    return bank


def make_state(root: RootSeed, num_players: int = 6):
    """Точно повторяет параметры создания стартового состояния в training loop."""
    return pkrs.State.from_seed(
        n_players=num_players,
        button=root.button,
        sb=1,
        bb=2,
        stake=200.0,
        seed=root.deal_seed,
    )


def current_rss_mb() -> float:
    """Возвращает RSS процесса; отсутствие psutil не делает benchmark нерабочим."""
    try:
        import psutil

        return psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)
    except (ImportError, OSError):
        return 0.0


def build_metrics(
    *,
    roots: int,
    wall_seconds: float,
    stats: dict[str, Any],
    rss_before_mb: float,
    rss_after_mb: float,
    profile: dict[str, Any],
    mode: str = "production",
) -> dict[str, Any]:
    nodes = int(stats.get("nodes", 0))
    traverser_nodes = int(stats.get("traversing_decision_nodes", 0))
    opponent_nodes = int(stats.get("opponent_decision_nodes", 0))
    decision_nodes = traverser_nodes + opponent_nodes
    return {
        "mode": mode,
        "roots": roots,
        "nodes": nodes,
        "terminal_nodes": int(stats.get("terminal_nodes", 0)),
        "decision_nodes": decision_nodes,
        "traverser_nodes": traverser_nodes,
        "opponent_nodes": opponent_nodes,
        "wall_seconds": wall_seconds,
        "nodes_per_root": nodes / roots if roots else 0.0,
        "nodes_per_second": nodes / wall_seconds if wall_seconds else 0.0,
        "decisions_per_second": decision_nodes / wall_seconds if wall_seconds else 0.0,
        "microseconds_per_node": wall_seconds * 1_000_000 / nodes if nodes else 0.0,
        "rss_before_mb": rss_before_mb,
        "rss_after_mb": rss_after_mb,
        "profile": profile,
    }


def json_safe(value: Any) -> Any:
    """Преобразует numpy-счётчики profiler/engine в JSON без потери значений."""
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    return value


def run_roots(
    agent: Any,
    roots: Iterable[RootSeed],
    *,
    iteration: int,
    traversing_player: int,
    state_factory: Callable[[RootSeed], Any] = make_state,
) -> None:
    """Выполняет только traversal; обучение и сериализация намеренно отсутствуют."""
    for root in roots:
        agent.cfr_traverse_multi(state_factory(root), iteration, traversing_player)


def _set_eval_mode(agent: DeepCFRAgent) -> None:
    """Исключает dropout/BatchNorm-изменения, не затрагивая CFR-логику."""
    for name in ("advantage_net", "advantage_target_net", "strategy_net"):
        network = getattr(agent, name, None)
        if network is not None:
            network.eval()


def _load_agent(checkpoint: Path, device: str, num_players: int) -> DeepCFRAgent:
    agent = DeepCFRAgent(player_id=0, num_players=num_players, device=device)
    agent._load_checkpoint(str(checkpoint))
    _set_eval_mode(agent)
    return agent


def run_benchmark(
    *,
    checkpoint: Path,
    bank: Sequence[RootSeed],
    seed: int,
    profile_level: str,
    warmup_roots: int,
    iteration: int,
    device: str,
    num_players: int = 6,
) -> dict[str, Any]:
    if profile_level not in PROFILE_LEVELS:
        raise ValueError(f"Недопустимый profile level: {profile_level}")
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint не найден: {checkpoint}")
    if warmup_roots < 0:
        raise ValueError("warmup-roots не может быть отрицательным.")

    # Warmup выполняется на отдельном агенте: измеряемая часть всегда стартует
    # с исходного checkpoint и пустых runtime-буферов.
    if warmup_roots:
        warmup_agent = _load_agent(checkpoint, device, num_players)
        seed_everything(seed ^ 0x5EED)
        warmup_agent.prepare_iteration(iteration, 0)
        warmup_agent.reset_traversal_stats()
        run_roots(
            warmup_agent,
            generate_root_bank(warmup_roots, seed ^ 0xA11CE, num_players),
            iteration=iteration,
            traversing_player=0,
            state_factory=lambda root: make_state(root, num_players),
        )

    agent = _load_agent(checkpoint, device, num_players)
    set_traversal_profiler_level(profile_level)
    TRAVERSAL_PROFILER.reset()
    agent.prepare_iteration(iteration, 0)
    agent.reset_traversal_stats()
    seed_everything(seed)

    threads_before = torch.get_num_threads()
    rss_before_mb = current_rss_mb()
    started_at = time.perf_counter()
    try:
        # Production traversal использует один BLAS-поток для batch=1.
        torch.set_num_threads(1)
        run_roots(
            agent,
            bank,
            iteration=iteration,
            traversing_player=0,
            state_factory=lambda root: make_state(root, num_players),
        )
    finally:
        torch.set_num_threads(threads_before)
    wall_seconds = time.perf_counter() - started_at

    TRAVERSAL_PROFILER.set_traversal_nodes(agent.traversal_nodes)
    TRAVERSAL_PROFILER.assert_balanced()
    traversal_stats = agent.get_traversal_stats()
    result = build_metrics(
        roots=len(bank),
        wall_seconds=wall_seconds,
        stats=traversal_stats,
        rss_before_mb=rss_before_mb,
        rss_after_mb=current_rss_mb(),
        profile=TRAVERSAL_PROFILER.get_all_stats(),
    )
    result["traversal_stats"] = traversal_stats
    result["structural_diagnostics"] = json_safe({
        "depth_histogram": agent.depth_histogram,
        "raise_frequency": traversal_stats["opponent_raise_frequency"],
    })
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--roots", type=int, default=10)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--root-bank", type=Path, default=None)
    parser.add_argument("--warmup-roots", type=int, default=2)
    parser.add_argument("--iteration", type=int, default=1)
    parser.add_argument("--device", choices=("cpu",), default="cpu")
    parser.add_argument("--profile-level", choices=PROFILE_LEVELS, default="off")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.root_bank is not None and args.root_bank.exists():
        bank = load_root_bank(args.root_bank)
        if len(bank) < args.roots:
            raise ValueError("В переданном root bank меньше roots, чем запрошено.")
        bank = bank[:args.roots]
    else:
        bank = generate_root_bank(args.roots, args.seed)
        save_root_bank(args.root_bank or args.output.with_suffix(".roots.json"), bank)

    result = run_benchmark(
        checkpoint=args.checkpoint,
        bank=bank,
        seed=args.seed,
        profile_level=args.profile_level,
        warmup_roots=args.warmup_roots,
        iteration=args.iteration,
        device=args.device,
    )
    result["root_bank"] = [asdict(root) for root in bank]
    result["profile_level"] = args.profile_level
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
