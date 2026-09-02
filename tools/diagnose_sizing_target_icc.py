#!/usr/bin/env python3
"""#113: потолок R^2 = ICC таргета sizing-advantage по (состояние, анкер).

ICC = between / (between + within), где within — дисперсия таргета внутри клетки
(идентичное состояние, кредитуемый анкер) между визитами, between — дисперсия средних
клетки. Если ICC мал (~0.05-0.1), никакая модель не даст R^2 выше: таргет под одним
сэмплированным chance невыучиваем, вход ни при чём.

Считаем на текущем буфере (raw q_anchor - ev) + на чистом MC-регрете `grounded - ev`
(ev = mean(q_anchors_post - regrets) — разведчик «сколько теряет Q-инъекция»).

Использование:
    python tools/diagnose_sizing_target_icc.py -c models/test107v18/multi_checkpoint_iter_30.pt
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


def _resolve_path(raw: str) -> Path:
    candidate = Path(raw)
    if candidate.is_absolute():
        return candidate
    from_invocation = (_INVOCATION_CWD / candidate).resolve()
    if from_invocation.is_file():
        return from_invocation
    return (ROOT / candidate).resolve()


def _fnv1a_64(states):
    u = np.ascontiguousarray(states, dtype=np.float32).view(np.uint32)
    n, d = u.shape
    h = np.full(n, np.uint64(1469598103934665603), dtype=np.uint64)
    prime = np.uint64(1099511628211)
    for j in range(d):
        h = (h ^ u[:, j].astype(np.uint64)) * prime
    return h


def _icc(cells_means, cells_n, cells_ss_within):
    means = np.asarray(cells_means, dtype=np.float64)
    ns = np.asarray(cells_n, dtype=np.float64)
    ss_within = np.asarray(cells_ss_within, dtype=np.float64)
    n_obs = float(ns.sum())
    n_cells = means.size
    if n_obs <= n_cells or n_cells < 2:
        return float("nan"), float("nan"), float("nan")
    grand = float((means * ns).sum() / n_obs)
    ss_total_within = float(ss_within.sum())
    ss_between = float((ns * (means - grand) ** 2).sum())
    ms_within = ss_total_within / (n_obs - n_cells)
    ms_between = ss_between / (n_cells - 1)
    n0 = (n_obs - float((ns ** 2).sum()) / n_obs) / (n_cells - 1)
    between_var = max((ms_between - ms_within) / max(n0, 1e-9), 0.0)
    icc = between_var / (between_var + ms_within)
    return icc, between_var, ms_within


def _group_by_hash(inverse, cred_bool, tgt):
    n = inverse.size
    order = np.argsort(inverse, kind='stable')
    inv_sorted = inverse[order]
    m = int(inverse.max()) + 1
    bounds = np.searchsorted(inv_sorted, np.arange(m))
    bounds = np.append(bounds, n)
    means, ns, ss = [], [], []
    n_groups2 = 0
    for g in range(m):
        lo, hi = int(bounds[g]), int(bounds[g + 1])
        if hi - lo < 2:
            continue
        n_groups2 += 1
        rows = order[lo:hi]
        cred_g = cred_bool[rows]
        for a in range(15):
            sel = rows[cred_g[:, a]]
            if sel.size < 2:
                continue
            v = tgt[sel, a]
            v = v[np.isfinite(v)]
            if v.size < 2:
                continue
            m_ = float(v.mean())
            means.append(m_)
            ns.append(v.size)
            ss.append(float(((v - m_) ** 2).sum()))
    return means, ns, ss, n_groups2


def _pair_corr(inverse, cred_bool, tgt):
    """ICc(1,1): корреляция между двумя визитами одного состояния на клетках ровно из 2."""
    n = inverse.size
    order = np.argsort(inverse, kind='stable')
    inv_sorted = inverse[order]
    m = int(inverse.max()) + 1
    bounds = np.searchsorted(inv_sorted, np.arange(m))
    bounds = np.append(bounds, n)
    pairs = []
    for g in range(m):
        lo, hi = int(bounds[g]), int(bounds[g + 1])
        if hi - lo < 2:
            continue
        rows = order[lo:hi]
        cred_g = cred_bool[rows]
        for a in range(15):
            sel = rows[cred_g[:, a]]
            if sel.size < 2:
                continue
            v = tgt[sel, a]
            v = v[np.isfinite(v)]
            if v.size < 2:
                continue
            for i in range(v.size):
                for j in range(i + 1, v.size):
                    pairs.append((v[i], v[j]))
    if len(pairs) < 2:
        return float("nan"), 0
    p = np.asarray(pairs, dtype=np.float64)
    c = float(np.corrcoef(p[:, 0], p[:, 1])[0, 1])
    return c, len(p)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-c", "--checkpoint", required=True)
    parser.add_argument("-o", "--output")
    args = parser.parse_args()

    torch.set_num_threads(1)
    path = _resolve_path(args.checkpoint)
    out_path = Path(args.output) if args.output else path.with_name(path.stem + "_icc.txt")

    lines = []

    def emit(text: str = "") -> None:
        print(text)
        lines.append(text)

    emit(f"чекпоинт: {path}")
    agent = DeepCFRAgent(player_id=0, num_players=6, device="cpu")
    agent._load_checkpoint(str(path))
    buf = agent.sizing_advantage_buffer
    n = len(buf)
    credits = buf._credit_masks[:n]
    cred_bool = credits > 0.5
    regrets = buf._regrets[:n].astype(np.float64)
    qpost = buf._q_anchors_post[:n].astype(np.float64)
    grounded = buf._grounded[:n].astype(np.float64)
    states = buf._states[:n]

    emit(f"строк в буфере: {n}  credit>=2: {int((credits.sum(axis=1) >= 2).sum())}")

    emit("хешируем состояния (FNV-1a 64)...")
    h = _fnv1a_64(states)
    uniq_h, inverse, counts = np.unique(h, return_inverse=True, return_counts=True)
    n_dup_groups = int((counts >= 2).sum())
    n_dup_rows = int(counts[counts >= 2].sum())
    emit(f"уникальных состояний={len(uniq_h)}  групп с >=2 визитами={n_dup_groups}  "
         f"строк в них={n_dup_rows}")

    ev = (qpost - regrets).mean(axis=1)

    for label, tgt in (
        ("raw (q_anchor - ev)", regrets),
        ("clean MC (grounded - ev)", grounded - ev[:, None]),
    ):
        means, ns, ss, ng = _group_by_hash(inverse, cred_bool, tgt)
        icc, between, within = _icc(means, ns, ss)
        corr, npairs = _pair_corr(inverse, cred_bool, tgt)
        emit(f"  {label}: ICC(ANOVA)={icc:.4f} between={between:.4f} within={within:.4f}  "
             f"клеток={len(means)}  групп>={ng}")
        emit(f"    corr(визит1,визит2)={corr:.4f}  (R^2-потолок ~ {corr**2:.4f})  пар={npairs}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nОтчёт сохранён: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())