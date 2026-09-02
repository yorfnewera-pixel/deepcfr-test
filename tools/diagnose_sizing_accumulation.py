#!/usr/bin/env python3
"""S6: усиливает ли бутстрап общую компоненту накопленного таргета.

Фон. Первый замер (diagnose_sizing_marginal) показал: политика сети хуже собственной
маргинальной, а контроль «marginal <= uniform» провален. Сеть учится НЕ на raw-регретах,
а на DCFR+ накоплении `target = clamp(prev) * discount + R` (положительный самореференс).
Опуз уточнил механизм: RM инвариантен к равномерному МАСШТАБУ (поэтому «100x» ничего не
объясняет), но НЕ к равномерному СДВИГУ. Общая компонента (среднее по анкерам строки)
присутствует в каждом анкере и накапливается когерентно, а анкеро-специфичная — только на
1-3 кредитуемых и с шумом. Если так, отношение «общая/специфичная» в накопленном значении
(= логиты сети) обязано РАСТИ по итерациям, а в сырых регретах — оставаться плоским.

ЗАМЕР (офлайн, без прогонов). По чекпоинтам iter 5..30 одной серии:
  по holdout-строкам (credit_mask.sum >= 2) раскладываем на кредитуемых анкерах
      value = common(среднее по анкерам строки) + residual
  и считаем ratio = std(common) / std_pooled(residual) отдельно для
      L = логиты advantage_sizing_net  (реализованное накопленное значение)
      R = сырые регреты буфера          (эталон, до накопления)
  Рост ratio_L(t) при плоском ratio_R(t) => механизм подтверждён.

Лечение (если подтверждено): вычитать среднее по строке ДО накопления.

Использование:
    python tools/diagnose_sizing_accumulation.py --dir models/test107v17 [--iters 5 10 15 20 25 30]
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
_INVOCATION_CWD = Path.cwd()
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))

from src.core.deep_cfr import DeepCFRAgent  # noqa: E402


def _resolve_dir(raw: str) -> Path:
    candidate = Path(raw)
    if candidate.is_absolute():
        return candidate
    return (_INVOCATION_CWD / candidate).resolve()


def _decompose(values, credit_mask_bool):
    """Раскладывает value (N,K) на общую (среднее строки) и остаток по кредитуемым анкерам.

    Возвращает (common_std, residual_std) — std общих по строкам и pooled std остатка.
    var = var(common) + mean(var(residual)) — разложение полной дисперсии.
    """
    common = []
    resid_all = []
    n_rows = values.shape[0]
    for i in range(n_rows):
        sel = credit_mask_bool[i]
        v = values[i, sel]
        if v.size < 2:
            continue
        m = float(v.mean())
        common.append(m)
        resid_all.extend((v - m).tolist())
    common = np.asarray(common, dtype=np.float64)
    resid_all = np.asarray(resid_all, dtype=np.float64)
    if common.size < 2 or resid_all.size < 2:
        return float("nan"), float("nan")
    return float(common.std()), float(resid_all.std())


def _decompose_all15(values):
    """Раскладывает value (N,15) на общую (среднее строки по всем 15) и остаток.

    Определено на ЛЮБОЙ строке (в т.ч. single-credit), где кредитуемое разложение вырождено.
    """
    common = values.mean(axis=1)
    resid = values - common[:, None]
    return float(common.std()), float(resid.std())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dir", required=True, help="каталог с multi_checkpoint_iter_*.pt")
    parser.add_argument("--iters", nargs="+", type=int, default=[5, 10, 15, 20, 25, 30])
    parser.add_argument("-o", "--output")
    args = parser.parse_args()

    torch.set_num_threads(1)
    root = _resolve_dir(args.dir)
    out_path = Path(args.output) if args.output else (root / "accumulation_decomp.txt")

    lines = []

    def emit(text: str = "") -> None:
        print(text)
        lines.append(text)

    def load_iter(it: int):
        path = root / f"multi_checkpoint_iter_{it}.pt"
        agent = DeepCFRAgent(player_id=0, num_players=6, device="cpu")
        agent._load_checkpoint(str(path))
        return agent

    emit(f"каталог: {root}")
    emit(f"{'iter':>4} {'n_hold':>8} {'k_cr':>5} | {'ratio_L(кр)':>11} {'ratio_L(15)':>11} "
         f"{'ratio_R(кр)':>11} {'ratio_R(15)':>11}")
    emit("-" * 76)

    results = []  # (iter, ratio_L_cred, ratio_L_15, ratio_R_cred, ratio_R_15)
    for it in args.iters:
        agent = load_iter(it)
        buf = agent.sizing_advantage_buffer
        net = agent.advantage_sizing_net
        net.eval()
        n_buf = len(buf)
        credits = buf._credit_masks[:n_buf]
        cred_bool = credits > 0.5
        credit_ge2 = credits.sum(axis=1) >= 2
        hold_sel = buf._holdout[:n_buf].astype(bool) & credit_ge2
        n_hold = int(hold_sel.sum())
        if n_hold == 0:
            emit(f"{it:>4} нет holdout-строк — пропуск")
            continue

        states = buf._states[:n_buf][hold_sel].astype(np.float32)
        regrets = buf._regrets[:n_buf][hold_sel].astype(np.float64)
        cred = cred_bool[hold_sel]

        with torch.no_grad():
            logits = []
            for start in range(0, len(states), 4096):
                chunk = states[start:start + 4096]
                logits.append(net(torch.from_numpy(chunk)).numpy())
        logits = np.concatenate(logits, axis=0).astype(np.float64)

        k_cr = cred.sum(axis=1).mean()
        lc_std, lr_std = _decompose(logits, cred)
        rc_std, rr_std = _decompose(regrets, cred)
        ratio_l = lc_std / lr_std if lr_std > 0 else float("nan")
        ratio_r = rc_std / rr_std if rr_std > 0 else float("nan")
        lc15_std, lr15_std = _decompose_all15(logits)
        rc15_std, rr15_std = _decompose_all15(regrets)
        ratio_l15 = lc15_std / lr15_std if lr15_std > 0 else float("nan")
        ratio_r15 = rc15_std / rr15_std if rr15_std > 0 else float("nan")
        results.append((it, ratio_l, ratio_l15, ratio_r, ratio_r15))
        emit(f"{it:>4} {n_hold:>8} {k_cr:>5.2f} | {ratio_l:>11.3f} {ratio_l15:>11.3f} "
             f"{ratio_r:>11.3f} {ratio_r15:>11.3f}")

    emit()
    emit("ТРЕНД: рост ratio_L(t) при плоском ratio_R(t) = бутстрап раздувает общую компоненту.")
    for it, rl, rl15, rr, rr15 in results:
        emit(f"  iter {it:>2}: ratio_L(кр)={rl:>7.3f}  ratio_L(15)={rl15:>7.3f}  "
             f"ratio_R(кр)={rr:>7.3f}  ratio_R(15)={rr15:>7.3f}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nОтчёт сохранён: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())