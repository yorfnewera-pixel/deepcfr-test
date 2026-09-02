#!/usr/bin/env python3
"""Офлайн-харнесс обучения sizing-ноги на чекпоинте (S2-харнесс).

Зачем: прогон 30 итераций стоит часы, а симуляция обучения на уже собранном
`sizing_advantage_buffer` — минуты. Харнесс дважды за одну сессию #107 переворачивал выводы,
и дважды же давал НЕВЕРНЫЙ ответ, когда расходился с реальностью. Отсюда два обязательных
self-check, без которых результаты не выдаются.

SELF-CHECK 1 (МЕХАНИКА). Реплика цикла `train_sizing_anchor_network` сверяется с фактическим
лоссом на том же чекпоинте: `total_loss == fresh_share * fresh_loss` и таргет строится ровно
как в `deep_cfr.py:2758-2767`. Провал = симуляция расходится с кодом, результаты не печатаются.
Исторический случай: ранняя версия синхронизировала target-сеть КАЖДЫЙ шаг, тогда как
`:2790` стоит ВНЕ цикла (AST: тело 2748-2787). Это дало фантомный компаундинг 0.9474^300.

SELF-CHECK 2 (МЕТРИКА). `TV(net)/TV(uniform)` печатается всегда и первым, `ptp` — только с
пометкой «диагностика». `regret_matching_anchors` инвариантен к равномерному масштабу: сжатие
таргетов x333 даёт TV=0.0000 при обвале ptp. Гейт по ptp отменён (#107 §2.8 п.12).

Использование:
    python tools/sizing_train_harness.py --checkpoint models/test107v8/multi_checkpoint_iter_30.pt
    python tools/sizing_train_harness.py -c CKPT --steps 300 --only-fresh
    python tools/sizing_train_harness.py -c CKPT --steps 300 --clip 10.0 --lr 1e-3
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
_INVOCATION_CWD = Path.cwd()
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))

from src.core.model import SizingAnchorNet  # noqa: E402
from src.core.sizing import regret_matching_anchors  # noqa: E402

ANCHORS = 15
MIN_AVAILABLE = 10
CURVE_POINTS = (10, 30, 100, 300)
LOSS_RTOL = 0.05


def _resolve_path(raw: str) -> Path:
    """Путь относительно CWD вызова, затем относительно корня проекта.

    Модуль делает chdir(ROOT) ради импортов src.*, поэтому относительные пути,
    переданные из другой директории, иначе теряются.
    """
    candidate = Path(raw)
    if candidate.is_absolute():
        return candidate
    from_invocation = (_INVOCATION_CWD / candidate).resolve()
    if from_invocation.is_file():
        return from_invocation
    return (ROOT / candidate).resolve()


def _discount(iteration: int, alpha: float = 1.5) -> float:
    if iteration <= 1:
        return 0.0
    base = (iteration - 1) ** alpha
    return base / (base + 1.5)


def _min_prob(iteration: int, start: float = 0.015, end: float = 0.003, decay: int = 50) -> float:
    progress = min(max(iteration / max(decay, 1), 0.0), 1.0)
    return start + (end - start) * progress


def load_buffer(checkpoint: dict[str, Any]) -> dict[str, np.ndarray]:
    required = (
        "sizing_advantage_buffer_states",
        "sizing_advantage_buffer_regrets",
        "sizing_advantage_buffer_masks",
        "sizing_advantage_buffer_iterations",
    )
    missing = [key for key in required if key not in checkpoint]
    if missing:
        raise RuntimeError(
            "чекпоинт без sizing_advantage_buffer: "
            f"нет {missing}. Нужен полный чекпоинт, не *_light.pt"
        )
    data: dict[str, np.ndarray | None] = {
        "states": np.asarray(checkpoint[required[0]], dtype=np.float32),
        "regrets": np.asarray(checkpoint[required[1]], dtype=np.float32),
        "masks": np.asarray(checkpoint[required[2]], dtype=np.float32),
        "iterations": np.asarray(checkpoint[required[3]]),
    }
    if "sizing_advantage_buffer_credit_masks" in checkpoint:
        data["credit_masks"] = np.asarray(
            checkpoint["sizing_advantage_buffer_credit_masks"], dtype=np.float32
        )
    else:
        data["credit_masks"] = None
    return data


def build_net(state_dict: dict[str, Any]) -> SizingAnchorNet:
    shapes = {k: tuple(v.shape) for k, v in state_dict.items() if torch.is_tensor(v)}
    net = SizingAnchorNet(
        input_size=shapes["base.0.weight"][1],
        hidden_size=shapes["base.0.weight"][0],
        num_sizes=shapes["anchor_head.weight"][0],
    )
    net.load_state_dict(state_dict, strict=True)
    return net


def make_target(
    net_target: SizingAnchorNet,
    states: torch.Tensor,
    regrets: torch.Tensor,
    masks: torch.Tensor,
    iterations: np.ndarray,
    iteration: int,
    no_bootstrap: bool = False,
) -> torch.Tensor:
    """Точная реплика deep_cfr.py:2759-2767. Снапшот target-сети НЕ обновляется внутри шага.

    no_bootstrap=True: таргет = чистый regret без `target_net(s)·discount` (диагностика дефекта
    сборки — проверяет, топит ли бутстрап-член анкеро-специфичный сигнал).
    """
    if no_bootstrap:
        return regrets * masks
    with torch.no_grad():
        previous = torch.clamp(net_target(states), min=0.0)
        fresh = torch.from_numpy((iterations == iteration).astype(np.float32)).unsqueeze(1)
        accumulated = previous * _discount(iteration) + regrets
        return (fresh * accumulated + (1.0 - fresh) * previous) * masks


def self_check_mechanics(
    net: SizingAnchorNet,
    net_target: SizingAnchorNet,
    buffer: dict[str, np.ndarray],
    iteration: int,
    logged_loss: float | None,
    rng: np.random.Generator,
) -> tuple[bool, list[str]]:
    """Проверяет, что реплика цикла совпадает с механикой deep_cfr."""
    lines: list[str] = []
    size = len(buffer["states"])
    index = rng.choice(size, min(30000, size), replace=False)
    states = torch.from_numpy(buffer["states"][index])
    regrets = torch.from_numpy(buffer["regrets"][index])
    masks = torch.from_numpy(buffer["masks"][index])
    iters = buffer["iterations"][index]

    target = make_target(net_target, states, regrets, masks, iters, iteration)
    with torch.no_grad():
        per_sample = F.mse_loss(net(states), target, reduction="none")

    mask_bool = masks.bool()
    fresh_bool = torch.from_numpy((iters == iteration)).unsqueeze(1) & mask_bool
    stale_bool = mask_bool & ~fresh_bool
    total = float((per_sample * masks).sum() / masks.sum())
    fresh_loss = float(per_sample[fresh_bool].mean()) if int(fresh_bool.sum()) else float("nan")
    stale_loss = float(per_sample[stale_bool].mean()) if int(stale_bool.sum()) else 0.0
    fresh_share = float(fresh_bool.sum()) / float(mask_bool.sum())

    lines.append(f"  fresh-доля записей под маской : {fresh_share:.4f}")
    lines.append(f"  loss на fresh                : {fresh_loss:.4f}")
    lines.append(f"  loss на stale                : {stale_loss:.3e}  (ожидается ~0)")
    lines.append(f"  полный loss                  : {total:.4f}")

    ok = True
    identity = fresh_share * fresh_loss
    if not np.isfinite(identity) or abs(identity - total) > max(LOSS_RTOL * total, 1e-6):
        lines.append(
            f"  ПРОВАЛ: fresh_share*fresh_loss={identity:.4f} != полный loss={total:.4f}"
        )
        ok = False
    else:
        lines.append(f"  OK: fresh_share*fresh_loss={identity:.4f} == полный loss (#107 §2.8 п.17)")

    if stale_loss > 1e-4:
        lines.append(f"  ПРОВАЛ: stale_loss={stale_loss:.3e} должен быть ~0 (deep_cfr.py:2767)")
        ok = False

    if logged_loss is not None:
        delta = abs(total - logged_loss) / max(logged_loss, 1e-9)
        verdict = "OK" if delta <= 0.35 else "ПРОВАЛ"
        lines.append(
            f"  {verdict}: против живого прогона logged={logged_loss:.4f} "
            f"пересчёт={total:.4f} (отклонение {100 * delta:.1f}%)"
        )
        if delta > 0.35:
            ok = False
    else:
        lines.append("  (--logged-loss не задан: сверка с живым прогоном пропущена)")
    return ok, lines


def self_check_metric(min_prob: float) -> tuple[bool, list[str]]:
    """Контроль инвариантности: равномерное сжатие таргетов не меняет политику."""
    rng = np.random.default_rng(12345)
    deltas = []
    for _ in range(200):
        regrets = rng.normal(0.0, 1.5, size=ANCHORS).astype(np.float32)
        base = regret_matching_anchors(regrets, min_prob=min_prob)
        shrunk = regret_matching_anchors(regrets * 0.003, min_prob=min_prob)
        deltas.append(0.5 * float(np.abs(base - shrunk).sum()))
    worst = max(deltas)
    ok = worst < 1e-6
    verdict = "OK" if ok else "ПРОВАЛ"
    return ok, [
        f"  {verdict}: сжатие таргетов x333 -> max TV = {worst:.3e} (ожидается ~0)",
        "  => ptp НЕ является метрикой качества; гейт — TV(net)/TV(uniform) (#107 §2.8 п.12)",
    ]


def evaluate(
    net: SizingAnchorNet,
    states: np.ndarray,
    regrets: np.ndarray,
    masks: np.ndarray,
    min_prob: float,
) -> dict[str, float]:
    net.eval()
    with torch.no_grad():
        predictions = net(torch.from_numpy(states)).numpy()

    tv_net, tv_uniform, ptp_values, spearman = [], [], [], []
    for row in range(len(states)):
        available = masks[row].astype(bool)
        count = int(available.sum())
        if count < 2:
            continue
        policy_net = regret_matching_anchors(predictions[row][available], min_prob=min_prob)
        policy_target = regret_matching_anchors(regrets[row][available], min_prob=min_prob)
        uniform = np.full(count, 1.0 / count)
        tv_net.append(0.5 * float(np.abs(policy_net - policy_target).sum()))
        tv_uniform.append(0.5 * float(np.abs(uniform - policy_target).sum()))
        ptp_values.append(float(np.ptp(predictions[row][available])))
        pred, truth = predictions[row][available], regrets[row][available]
        if pred.std() > 1e-9 and truth.std() > 1e-9:
            spearman.append(float(np.corrcoef(_rank(pred), _rank(truth))[0, 1]))

    mean_net, mean_uniform = float(np.mean(tv_net)), float(np.mean(tv_uniform))
    return {
        "tv_net": mean_net,
        "tv_uniform": mean_uniform,
        "ratio": mean_net / max(mean_uniform, 1e-12),
        "ptp": float(np.mean(ptp_values)),
        "spearman": float(np.mean(spearman)) if spearman else float("nan"),
        "rows": len(tv_net),
    }


def _rank(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    ranks[order] = np.arange(len(values), dtype=np.float64)
    return ranks


def holdout_split(states: np.ndarray, fraction: float = 0.1) -> np.ndarray:
    """Детерминированный хеш-сплит по байтам состояния: все копии состояния на одну сторону.

    Дублей в буфере ~0.5%, поэтому практически это случайный сплит; он ловит переобучение на
    близких состояниях, а не на идентичных (#107 §6.2, 107Result §4).
    """
    words = states.view(np.uint32)
    digest = np.bitwise_xor.reduce(words * np.uint32(2654435761), axis=1)
    threshold = int(fraction * 1000)
    return (digest % 1000) < threshold


def run_training(
    net: SizingAnchorNet,
    net_target: SizingAnchorNet,
    buffer: dict[str, np.ndarray],
    train_index: np.ndarray,
    holdout: dict[str, np.ndarray],
    iteration: int,
    steps: int,
    batch_size: int,
    lr: float,
    clip: float,
    weight_decay: float,
    only_fresh: bool,
    no_bootstrap: bool,
    min_prob: float,
    rng: np.random.Generator,
) -> list[dict[str, float]]:
    optimizer = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=weight_decay)
    pool = train_index
    if only_fresh:
        pool = pool[buffer["iterations"][pool] == iteration]
        if len(pool) < batch_size:
            raise RuntimeError(f"--only-fresh: свежих строк {len(pool)} < batch {batch_size}")

    curve, clipped = [], 0
    checkpoints = [point for point in CURVE_POINTS if point <= steps]
    if steps not in checkpoints:
        checkpoints.append(steps)

    for step in range(1, steps + 1):
        selection = rng.choice(pool, batch_size, replace=False)
        states = torch.from_numpy(buffer["states"][selection])
        regrets = torch.from_numpy(buffer["regrets"][selection])
        masks = torch.from_numpy(buffer["masks"][selection])
        target = make_target(
            net_target, states, regrets, masks, buffer["iterations"][selection], iteration,
            no_bootstrap=no_bootstrap,
        )
        net.train()
        per_sample = F.mse_loss(net(states), target, reduction="none")
        loss = (per_sample * masks).sum() / masks.sum().clamp_min(1)
        optimizer.zero_grad()
        loss.backward()
        norm = float(torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=clip))
        if norm > clip:
            clipped += 1
        optimizer.step()

        if step in checkpoints:
            metrics = evaluate(
                net, holdout["states"], holdout["regrets"], holdout["masks"], min_prob
            )
            metrics["step"] = step
            metrics["clipped_pct"] = 100.0 * clipped / step
            curve.append(metrics)
    return curve


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("-c", "--checkpoint", required=True)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--clip", type=float, default=1.0)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--only-fresh", action="store_true",
                        help="учить только на свежих строках (верхняя граница снятия разбавления)")
    parser.add_argument("--no-bootstrap", action="store_true",
                        help="таргет = чистый regret без target_net·discount (проверка дефекта сборки)")
    parser.add_argument("--min-available", type=int, default=10,
                        help="минимальное число доступных анкеров в маске строки (для v12/v13 маска=credit, ставь 1)")
    parser.add_argument("--min-credits", type=int, default=0,
                        help="минимальное число кредитуемых анкеров в train-пуле (0 = без фильтра; >=2 отсекает группу 1)")
    parser.add_argument("--holdout-rows", type=int, default=3000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--logged-loss", type=float, default=None,
                        help="Sizing anchor loss из живого прогона на этой итерации (self-check 1)")
    args = parser.parse_args()

    torch.set_num_threads(1)
    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)

    path = _resolve_path(args.checkpoint)
    print(f"чекпоинт: {path}")
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    iteration = int(checkpoint.get("iteration", 0))
    buffer = load_buffer(checkpoint)
    min_prob = _min_prob(iteration)
    fresh_share = float((buffer["iterations"] == iteration).mean())
    print(f"iteration={iteration}  buffer={len(buffer['states'])} (кум. записей)  "
          f"min_prob={min_prob:.4f}  discount={_discount(iteration):.4f}")
    print(f"fresh-доля={100 * fresh_share:.2f}%  эфф.сэмплов при 10 шагах x256={10 * 256 * fresh_share:.0f}")
    print("  ВНИМАНИЕ: сравнивать прогоны только на равных КУМУЛЯТИВНЫХ записях буфера, "
          "не по номеру итерации (#107 §2.8 п.11)")

    net = build_net(checkpoint["advantage_sizing_net"])
    target_state = checkpoint.get("sizing_target_net", checkpoint["advantage_sizing_net"])
    net_target = build_net(target_state)
    net_target.eval()

    print()
    print("SELF-CHECK 1 (МЕХАНИКА): реплика цикла против deep_cfr.py:2758-2790")
    ok_mechanics, lines = self_check_mechanics(
        net, net_target, buffer, iteration, args.logged_loss, rng
    )
    for line in lines:
        print(line)

    print()
    print("SELF-CHECK 2 (МЕТРИКА): инвариантность regret matching к масштабу")
    ok_metric, lines = self_check_metric(min_prob)
    for line in lines:
        print(line)

    if not (ok_mechanics and ok_metric):
        print()
        print("SELF-CHECK ПРОВАЛЕН — результаты НЕ выдаются.")
        print("Симуляция расходится с реальным кодом; сверьте механику перед использованием.")
        return 2

    available = buffer["masks"].sum(axis=1) >= args.min_available
    is_holdout = holdout_split(buffer["states"]) & available
    holdout_pool = np.flatnonzero(is_holdout)
    train_index = np.flatnonzero(~is_holdout)
    if buffer.get("credit_masks") is not None and args.min_credits > 0:
        train_index = train_index[
            buffer["credit_masks"][train_index].sum(axis=1) >= args.min_credits
        ]
    if len(holdout_pool) > args.holdout_rows:
        holdout_pool = rng.choice(holdout_pool, args.holdout_rows, replace=False)

    holdout = {
        "states": buffer["states"][holdout_pool],
        "regrets": buffer["regrets"][holdout_pool],
        "masks": buffer["masks"][holdout_pool],
    }
    print()
    print(f"подмножество mask.sum() >= {args.min_available}: {int(available.sum())} строк "
          f"({100 * available.mean():.1f}%)")
    if buffer.get("credit_masks") is not None and args.min_credits > 0:
        print(f"train-пул после credit_mask.sum() >= {args.min_credits}: {len(train_index)} строк")
    print(f"холдаут: {len(holdout_pool)} строк (хеш-сплит 10%), train: {len(train_index)}")

    baseline = evaluate(net, holdout["states"], holdout["regrets"], holdout["masks"], min_prob)
    print()
    print(f"конфигурация: steps={args.steps} batch={args.batch_size} lr={args.lr} "
          f"clip={args.clip} only_fresh={args.only_fresh} no_bootstrap={args.no_bootstrap}")
    print()
    header = f"{'step':>5} {'TV(net)':>9} {'TV(unif)':>9} {'RATIO':>8} | {'ptp':>8} {'Spearman':>9} {'clip%':>6}"
    print(header)
    print("-" * len(header))
    print(f"{'base':>5} {baseline['tv_net']:>9.4f} {baseline['tv_uniform']:>9.4f} "
          f"{baseline['ratio']:>8.3f} | {baseline['ptp']:>8.4f} {baseline['spearman']:>9.4f} "
          f"{'-':>6}")

    curve = run_training(
        net, net_target, buffer, train_index, holdout, iteration, args.steps,
        args.batch_size, args.lr, args.clip, args.weight_decay, args.only_fresh,
        args.no_bootstrap, min_prob, rng,
    )
    for metrics in curve:
        print(f"{metrics['step']:>5} {metrics['tv_net']:>9.4f} {metrics['tv_uniform']:>9.4f} "
              f"{metrics['ratio']:>8.3f} | {metrics['ptp']:>8.4f} {metrics['spearman']:>9.4f} "
              f"{metrics['clipped_pct']:>6.1f}")

    final = curve[-1] if curve else baseline
    print()
    print("ГЕЙТ: RATIO = TV(net)/TV(uniform), цель < 0.8")
    print(f"  база {baseline['ratio']:.3f} -> итог {final['ratio']:.3f}  "
          f"({'ЛУЧШЕ' if final['ratio'] < baseline['ratio'] else 'ХУЖЕ'} базы)")
    if final["ratio"] >= 1.0:
        print("  RATIO >= 1.0: сеть не лучше равномерного выбора среди доступных анкеров.")
    print("  ptp и Spearman — ДИАГНОСТИКА, не гейт (см. SELF-CHECK 2).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
