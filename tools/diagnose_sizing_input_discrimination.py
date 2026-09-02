#!/usr/bin/env python3
"""#113: может ли ВХОД различать sizing-анкера (с контролем харнесса).

Офлайн, без прогонов. На чекпоинте берём буфер, честный holdout по состоянию
(buf._holdout), учим С НУЛЯ сеть state -> сырые регреты (маска по credit_mask)
и смотрим слотовую таблицу diag/offdiag (logit[a] ~ regret[b]) на holdout credit>=2.

Контролы (важно — без них вывод про «во входе нет информации» пустой):
  1. --overfit N: обучить MLP ДО ПЕРЕОБУЧЕНИЯ на N train-строках. R^2 -> ~1 значит
     харнесс умеет учить и ноль на holdout реально про представление; R^2 ~0 значит
     таргет на этом объёме не выучивается (шум within > between).
  2. --add-avail: добавить во вход 15 бит avail_mask (_avail_masks). Доступность
     заведомо различает анкеры; если diag не отделится даже с ней -> представление
     подтверждено жёстко; если отделится -> готовая правка sizing_availability_on_input.

Использование:
    python tools/diagnose_sizing_input_discrimination.py -c models/test107v18/multi_checkpoint_iter_30.pt --overfit 2500 --add-avail
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
_INVOCATION_CWD = Path.cwd()
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))

from src.core.deep_cfr import DeepCFRAgent  # noqa: E402
from src.core.model import SizingAnchorNet  # noqa: E402


def _resolve_path(raw: str) -> Path:
    candidate = Path(raw)
    if candidate.is_absolute():
        return candidate
    from_invocation = (_INVOCATION_CWD / candidate).resolve()
    if from_invocation.is_file():
        return from_invocation
    return (ROOT / candidate).resolve()


def _slot_stats(logits, regrets, avail_bool):
    """diag_mean, offdiag_mean, best_off, n_off_pos, n_off_valid."""
    K = logits.shape[1]
    corr = np.full((K, K), np.nan)
    for a in range(K):
        for b in range(K):
            rows = avail_bool[:, a] & avail_bool[:, b]
            if rows.sum() < 100:
                continue
            x = logits[rows, a].astype(np.float64)
            y = regrets[rows, b].astype(np.float64)
            if x.std() < 1e-9 or y.std() < 1e-9:
                continue
            corr[a, b] = float(np.corrcoef(x, y)[0, 1])
    diag = np.diag(corr)
    off = corr.copy()
    np.fill_diagonal(off, np.nan)
    ov = ~np.isnan(off)
    if ov.sum() == 0:
        return float(np.nanmean(diag)), float("nan"), None, 0, 0
    bi = np.unravel_index(int(np.nanargmax(off)), off.shape)
    return (float(np.nanmean(diag)), float(np.nanmean(off)), bi,
            int((off[ov] > 0).sum()), int(ov.sum()))


def _train(model, x, y, m, steps, batch, seed=0, lr=1e-4):
    torch.manual_seed(seed)
    np.random.seed(seed)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    n = x.shape[0]
    loss_ema = None
    for _ in range(steps):
        idx = np.random.choice(n, min(batch, n), replace=False)
        pred = model(x[idx])
        loss = (nn.functional.mse_loss(pred, y[idx], reduction='none') * m[idx]).sum() \
            / m[idx].sum().clamp_min(1)
        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        v = float(loss.item())
        loss_ema = v if loss_ema is None else 0.9 * loss_ema + 0.1 * v
    return loss_ema


def _masked_mse(model, x, y, m):
    with torch.no_grad():
        pred = model(x)
        return float(((nn.functional.mse_loss(pred, y, reduction='none') * m).sum()
                      / m.sum().clamp_min(1)).item())


def _masked_r2(model, x, y, m):
    with torch.no_grad():
        pred = model(x)
        se = (pred - y) ** 2 * m
        mu = (y * m).sum() / m.sum().clamp_min(1)
        total = ((y - mu) ** 2 * m).sum()
        r2 = 1.0 - float(se.sum()) / float(total.clamp_min(1e-12))
    return r2


def _logits(model, x, chunk=4096):
    with torch.no_grad():
        return np.concatenate([model(x[i:i + chunk]).numpy() for i in range(0, len(x), chunk)], 0)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-c", "--checkpoint", required=True)
    parser.add_argument("--steps", type=int, default=6000)
    parser.add_argument("--batch", type=int, default=1024)
    parser.add_argument("--target", choices=("raw", "clean"), default="raw",
                        help="raw = q_anchor - ev; clean = grounded - ev (чистый MC-регрет)")
    parser.add_argument("--overfit", type=int, default=0, help="N строк для контроля переобучения")
    parser.add_argument("--add-avail", action="store_true", help="добавить 15 бит avail_mask во вход")
    parser.add_argument("-o", "--output")
    args = parser.parse_args()

    torch.set_num_threads(1)
    path = _resolve_path(args.checkpoint)
    out_path = Path(args.output) if args.output else path.with_name(path.stem + "_input_discr.txt")

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
    credit_ge2 = credits.sum(axis=1) >= 2
    hold = buf._holdout[:n].astype(bool)
    train_sel = (~hold) & credit_ge2
    hold_sel = hold & credit_ge2

    if args.target == "clean":
        qpost = buf._q_anchors_post[:n].astype(np.float64)
        grounded = buf._grounded[:n].astype(np.float64)
        ev = (qpost - buf._regrets[:n].astype(np.float64)).mean(axis=1)
        target_all = np.nan_to_num(grounded - ev[:, None], nan=0.0).astype(np.float32)
    else:
        target_all = buf._regrets[:n].astype(np.float32)

    states_tr = buf._states[:n][train_sel].astype(np.float32)
    regrets_tr = target_all[train_sel]
    masks_tr = cred_bool[train_sel].astype(np.float32)

    states_ho = buf._states[:n][hold_sel].astype(np.float32)
    regrets_ho = target_all[hold_sel].astype(np.float64)
    masks_ho = cred_bool[hold_sel]

    emit(f"таргет: {args.target}   train строк (credit>=2, ~holdout): {len(states_tr)}")
    emit(f"holdout строк (credit>=2):        {len(states_ho)}")
    inp = int(buf._states.shape[1])

    net = agent.advantage_sizing_net
    net.eval()
    logits_trained = _logits(net, torch.from_numpy(states_ho)).astype(np.float64)
    d, o, bi, npos, nov = _slot_stats(logits_trained, regrets_ho, masks_ho)
    emit()
    emit(f"ЭТАЛОН trained head: diag={d:.3f} offdiag={o:.3f} gap={d - o:+.3f} offdiag>0: {npos}/{nov}")

    x_tr = torch.from_numpy(states_tr)
    y_tr = torch.from_numpy(regrets_tr)
    m_tr = torch.from_numpy(masks_tr)
    x_ho = torch.from_numpy(states_ho)
    y_ho = torch.from_numpy(regrets_ho.astype(np.float32))
    m_ho = torch.from_numpy(masks_ho.astype(np.float32))

    def report(name, model, xtr, ytr, mtr, xho_t, mask_ho):
        lg = _logits(model, xho_t).astype(np.float64)
        d, o, bi, npos, nov = _slot_stats(lg, regrets_ho, mask_ho)
        tr_mse = _masked_mse(model, xtr, ytr, mtr)
        ho_mse = _masked_mse(model, xho_t, y_ho, m_ho)
        ho_r2 = _masked_r2(model, xho_t, y_ho, m_ho)
        emit(f"{name}: diag={d:.3f} offdiag={o:.3f} gap={d - o:+.3f} "
             f"offdiag>0: {npos}/{nov} | MSE train={tr_mse:.4f} holdout={ho_mse:.4f} "
             f"HOLD R^2={ho_r2:+.4f}")

    emit()
    emit(f"свежая MLP 157->256->15, steps={args.steps}")
    mlp = SizingAnchorNet(input_size=inp, hidden_size=256, num_sizes=15)
    _train(mlp, x_tr, y_tr, m_tr, args.steps, args.batch)
    report("  MLP(157)", mlp, x_tr, y_tr, m_tr, x_ho, masks_ho)

    if args.add_avail:
        avail_tr = buf._avail_masks[:n][train_sel].astype(np.float32)
        avail_ho = buf._avail_masks[:n][hold_sel].astype(np.float32)
        x_tr_av = torch.from_numpy(np.concatenate([states_tr, avail_tr], axis=1))
        x_ho_av = torch.from_numpy(np.concatenate([states_ho, avail_ho], axis=1))
        emit()
        emit(f"свежая MLP (157+15 avail)->256->15, steps={args.steps}")
        mlp_av = SizingAnchorNet(input_size=inp + 15, hidden_size=256, num_sizes=15)
        _train(mlp_av, x_tr_av, y_tr, m_tr, args.steps, args.batch)
        report("  MLP(172)", mlp_av, x_tr_av, y_tr, m_tr, x_ho_av, masks_ho)

    if args.overfit > 0:
        N = args.overfit
        rng = np.random.default_rng(42)
        idx = rng.choice(len(states_tr), N, replace=False)
        x_o = x_tr[idx]
        y_o = y_tr[idx]
        m_o = m_tr[idx]
        mu = float((y_o * m_o).sum() / m_o.sum().clamp_min(1))
        total = float(((y_o - mu) ** 2 * m_o).sum())
        emit()
        emit(f"КОНТРОЛЬ харнесса: MLP на {N} train-строках до переобучения (steps={max(args.steps, 20000)}, lr=1e-3)")
        over = SizingAnchorNet(input_size=inp, hidden_size=256, num_sizes=15)
        _train(over, x_o, y_o, m_o, max(args.steps, 20000), min(N, 512), seed=123, lr=1e-3)
        r2_train = _masked_r2(over, x_o, y_o, m_o)
        r2_hold = _masked_r2(over, x_ho, y_ho, m_ho)
        emit(f"  R^2 на train({N}) = {r2_train:+.3f}   R^2 на holdout = {r2_hold:+.3f}   "
             f"(var target train = {total:.3f})")
        emit("  R^2(train) ~ 1 => харнесс рабочий, ноль на holdout = представление. R^2(train) ~ 0 => таргет не выучивается.")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nОтчёт сохранён: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())