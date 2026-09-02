#!/usr/bin/env python3
"""Проверка Q-шумовой гипотезы: согласие advantage_sizing_net с таргетами.

Читает sizing_advantage_buffer из чекпоинта v11+ (поля credit_masks, q_anchors_pre,
q_anchors_post, grounded) и считает четыре метрики:

  A. MAE + sign-agreement per-anchor, credit vs noncredit (контрол на анкер).
  B. TV(regret_matching(net), regret_matching(target)) по группам состояний.
  C. Ошибка Q = q_anchor_pre - grounded на кредитуемых vs std(MC)/std(target).
  D. w = (post - pre)/(grounded - pre) на кредитуемых — полнота коррекции.

Использование:
    python tools/diagnose_credit_split.py -c models/test107v11/multi_checkpoint_iter_30.pt
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from core.deep_cfr import DeepCFRAgent  # noqa: E402
from tools.diagnose_sizing_buffer_targets import (  # noqa: E402
    infer_logits,
    jsonable,
    regret_matching,
    summary,
)
from tools.sizing_report_naming import extract_iteration, report_path  # noqa: E402

ANCHORS = 15


def _sign_agreement(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if x.size == 0:
        return float("nan")
    return float(np.mean(np.sign(x) == np.sign(y)))


def _mae(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    return float(np.mean(np.abs(x - y))) if x.size else float("nan")


def _tv(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    return float(0.5 * np.abs(a - b).sum())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", "-c", default="models/test107v11/multi_checkpoint_iter_30.pt")
    parser.add_argument("--json-output", help="Explicit JSON output path")
    parser.add_argument("--txt-output", help="Explicit TXT output path")
    args = parser.parse_args()

    ckpt_path = (ROOT / args.checkpoint).resolve() if not Path(args.checkpoint).is_absolute() else Path(args.checkpoint)
    print(f"Loading checkpoint: {ckpt_path}")
    checkpoint = torch.load(ckpt_path, weights_only=False, map_location="cpu")

    required = (
        "sizing_advantage_buffer_states",
        "sizing_advantage_buffer_regrets",
        "sizing_advantage_buffer_masks",
        "sizing_advantage_buffer_credit_masks",
        "sizing_advantage_buffer_q_anchors_pre",
        "sizing_advantage_buffer_q_anchors_post",
        "sizing_advantage_buffer_grounded",
        "sizing_advantage_buffer_cur_id",
        "sizing_advantage_buffer_state_dim",
    )
    missing = [k for k in required if k not in checkpoint]
    if missing:
        raise KeyError(f"Missing checkpoint keys: {missing}")

    state_dim = int(checkpoint["sizing_advantage_buffer_state_dim"])
    cur_id = int(checkpoint["sizing_advantage_buffer_cur_id"])
    arrays = {
        k: checkpoint[f"sizing_advantage_buffer_{k}"]
        for k in ("states", "regrets", "masks", "credit_masks",
                  "q_anchors_pre", "q_anchors_post", "grounded")
    }
    n = min([cur_id] + [len(v) for v in arrays.values()])
    if n <= 0:
        raise RuntimeError("Empty sizing_advantage_buffer in checkpoint")

    states = np.asarray(arrays["states"])[:n, :state_dim].astype(np.float32)
    regrets = np.asarray(arrays["regrets"])[:n, :ANCHORS].astype(np.float64)
    masks = np.asarray(arrays["masks"])[:n, :ANCHORS].astype(bool)
    credit = np.asarray(arrays["credit_masks"])[:n, :ANCHORS].astype(bool)
    pre = np.asarray(arrays["q_anchors_pre"])[:n, :ANCHORS].astype(np.float64)
    post = np.asarray(arrays["q_anchors_post"])[:n, :ANCHORS].astype(np.float64)
    grounded = np.asarray(arrays["grounded"])[:n, :ANCHORS].astype(np.float64)

    finite = np.isfinite(states).all(axis=1) & np.isfinite(regrets).all(axis=1) & masks.any(axis=1)
    states, regrets, masks, credit, pre, post, grounded = (
        states[finite], regrets[finite], masks[finite], credit[finite],
        pre[finite], post[finite], grounded[finite],
    )
    if not len(states):
        raise RuntimeError("No finite buffer rows")
    print(f"Rows: {len(states)}; available anchors: {int(masks.sum())}; credited anchors: {int(credit.sum())}")

    agent = DeepCFRAgent(player_id=0, num_players=6, device="cpu")
    net = agent.advantage_sizing_net
    net.load_state_dict(checkpoint["advantage_sizing_net"], strict=False)
    net.eval()
    logits = infer_logits(net, states)[:, :ANCHORS].astype(np.float64)

    per_anchor = []
    for a in range(ANCHORS):
        avail = masks[:, a]
        cred = credit[:, a] & avail
        noncred = ~credit[:, a] & avail
        target = regrets[:, a]
        logit = logits[:, a]
        per_anchor.append({
            "anchor": a,
            "count_available": int(avail.sum()),
            "count_credit": int(cred.sum()),
            "count_noncredit": int(noncred.sum()),
            "mae_credit": _mae(logit[cred], target[cred]),
            "mae_noncredit": _mae(logit[noncred], target[noncred]),
            "sign_credit": _sign_agreement(logit[cred], target[cred]),
            "sign_noncredit": _sign_agreement(logit[noncred], target[noncred]),
        })

    q_err = (pre - grounded)[credit]
    q_err = q_err[np.isfinite(q_err)]
    mc_valid = grounded[credit][np.isfinite(grounded[credit])]
    mc_std = float(np.std(mc_valid)) if mc_valid.size else float("nan")
    target_flat = regrets[masks]
    target_std = float(np.std(target_flat))

    denom = grounded[credit] - pre[credit]
    w_vals = (post[credit] - pre[credit]) / denom
    w_vals = w_vals[np.isfinite(w_vals)]

    avail_count = masks.sum(axis=1).astype(np.float64)
    credit_avail_count = (credit & masks).sum(axis=1).astype(np.float64)
    frac = np.divide(credit_avail_count, avail_count, out=np.zeros_like(avail_count), where=avail_count > 0)
    high = frac >= 0.8
    low = (frac > 0) & (frac <= 0.3)

    def group_tv(sel: np.ndarray) -> list[float]:
        values = []
        for i in np.flatnonzero(sel):
            valid = masks[i]
            if not valid.any():
                continue
            values.append(_tv(regret_matching(logits[i][valid]), regret_matching(regrets[i][valid])))
        return values

    tv_high = group_tv(high)
    tv_low = group_tv(low)

    report = {
        "checkpoint": str(ckpt_path),
        "rows": int(len(states)),
        "metric_A_per_anchor": per_anchor,
        "metric_C_q_error": {
            "mean": float(np.mean(q_err)) if q_err.size else None,
            "std": float(np.std(q_err)) if q_err.size else None,
            "median_abs": float(np.median(np.abs(q_err))) if q_err.size else None,
            "p95_abs": float(np.quantile(np.abs(q_err), 0.95)) if q_err.size else None,
            "mc_std": mc_std,
            "target_std": target_std,
        },
        "metric_D_w": {
            "mean": float(np.mean(w_vals)) if w_vals.size else None,
            "std": float(np.std(w_vals)) if w_vals.size else None,
            "median": float(np.median(w_vals)) if w_vals.size else None,
            "fraction_abs_lt_0p95": float(np.mean(np.abs(w_vals) < 0.95)) if w_vals.size else None,
        },
        "metric_B_tv_by_state_group": {
            "high_credit_states": int(high.sum()),
            "low_credit_states": int(low.sum()),
            "tv_high": summary(np.array(tv_high)) if tv_high else None,
            "tv_low": summary(np.array(tv_low)) if tv_low else None,
        },
    }

    iteration = extract_iteration(checkpoint, ckpt_path)
    out_json = report_path(ROOT / "credit_split_diagnostic.json", iteration=iteration, explicit_output=args.json_output)
    out_txt = report_path(ROOT / "credit_split_diagnostic.txt", iteration=iteration, explicit_output=args.txt_output)
    if not out_json.is_absolute():
        out_json = ROOT / out_json
    if not out_txt.is_absolute():
        out_txt = ROOT / out_txt
    out_json.write_text(json.dumps(jsonable(report), indent=2, allow_nan=False), encoding="utf-8")

    lines = [
        "Credit-split Q-noise diagnostic",
        f"checkpoint: {ckpt_path}",
        f"rows: {len(states)}",
        "",
        "--- metric A (per-anchor, control on anchor) ---",
        "anchor  avail  credit  noncred  mae_credit  mae_noncred  sign_credit  sign_noncred",
    ]
    for r in per_anchor:
        lines.append(
            f"{r['anchor']:>6}  {r['count_available']:>5}  {r['count_credit']:>6}  {r['count_noncredit']:>7}  "
            f"{r['mae_credit']:.4f}  {r['mae_noncredit']:.4f}  {r['sign_credit']:.3f}  {r['sign_noncredit']:.3f}")
    c = report["metric_C_q_error"]
    d = report["metric_D_w"]
    b = report["metric_B_tv_by_state_group"]
    lines.append("")
    lines.append("--- metric C (Q error = pre - grounded on credited) ---")
    lines.append(f"mean={c['mean']} std={c['std']} median_abs={c['median_abs']} "
                 f"mc_std={c['mc_std']} target_std={c['target_std']}")
    lines.append("--- metric D (w = (post-pre)/(grounded-pre)) ---")
    lines.append(f"mean={d['mean']} median={d['median']} fraction_abs_lt_0p95={d['fraction_abs_lt_0p95']}")
    lines.append("--- metric B (TV by state group) ---")
    lines.append(f"high_credit_states={b['high_credit_states']} tv_high={b['tv_high']}")
    lines.append(f"low_credit_states={b['low_credit_states']} tv_low={b['tv_low']}")
    lines.append("")
    lines.append(f"detailed JSON: {out_json}")
    out_txt.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote JSON: {out_json}")
    print(f"Wrote TXT: {out_txt}")


if __name__ == "__main__":
    main()
