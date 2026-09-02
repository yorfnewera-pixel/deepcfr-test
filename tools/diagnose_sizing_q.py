"""Проверка sizing_q_net на iter_200: обучена ли, state-dependent ли.

Гипотеза: если sizing_q_net выдаёт константы (не обучена / q_head=0),
то sizing_cf_regrets = q_anchor - ev_sizing — чистый шум,
и advantage_sizing_net не может выучить ничего кроме маргинала.
"""
from __future__ import annotations

import argparse

import os
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from core.model import SizingQNetwork, SizingAnchorNet  # noqa: E402

CKPT = ROOT / "models" / "test107" / "multi_checkpoint_iter_200.pt"
ANCHORS = [0.1, 0.25, 0.33, 0.5, 0.66, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5, 2.75, 3.0]
MIN_BET = 0.1
MAX_BET = 3.0
RNG = np.random.default_rng(42)


def load_sizing_q(sd):
    """Конструируем SizingQNetwork по выведенным из весов размерностям."""
    shapes = {k: tuple(v.shape) for k, v in sd.items() if torch.is_tensor(v)}
    # size_embed.weight: [size_embed_dim, 1]
    size_embed_dim = shapes["size_embed.weight"][0]
    # base.0.weight: [hidden, state_dim + size_embed_dim]
    hidden = shapes["base.0.weight"][0]
    state_dim = shapes["base.0.weight"][1] - size_embed_dim
    net = SizingQNetwork(state_dim=state_dim, hidden_size=hidden, size_embed_dim=size_embed_dim)
    res = net.load_state_dict(sd, strict=True)
    if res.missing_keys or res.unexpected_keys:
        raise RuntimeError(f"load mismatch: {res}")
    net.eval()
    return net


def load_sizing_anchor(sd):
    shapes = {k: tuple(v.shape) for k, v in sd.items() if torch.is_tensor(v)}
    net = SizingAnchorNet(
        input_size=shapes["base.0.weight"][1],
        hidden_size=shapes["base.0.weight"][0],
        num_sizes=shapes["anchor_head.weight"][0],
        num_buckets=3,
    )
    res = net.load_state_dict(sd, strict=True)
    if res.missing_keys or res.unexpected_keys:
        raise RuntimeError(f"load mismatch: {res}")
    net.eval()
    return net


def eval_q(net, states, anchors, batch=4096):
    """Q(state, size) для всех пар state x anchor. Возвращает [N, K]."""
    K = len(anchors)
    norm = (np.array(anchors, dtype=np.float32) - MIN_BET) / (MAX_BET - MIN_BET)
    out = []
    with torch.no_grad():
        for i in range(0, len(states), batch):
            s = torch.from_numpy(states[i:i+batch]).float()
            B = s.shape[0]
            s_exp = s.unsqueeze(1).expand(-1, K, -1).reshape(-1, s.shape[-1])
            sz_exp = torch.from_numpy(norm).unsqueeze(0).expand(B, -1).reshape(-1).float()
            q = net(s_exp, sz_exp)
            out.append(q.detach().cpu().numpy().reshape(B, K))
    return np.concatenate(out, axis=0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default=str(CKPT.relative_to(ROOT)))
    args = parser.parse_args()
    checkpoint = Path(args.checkpoint)
    if not checkpoint.is_absolute():
        checkpoint = ROOT / checkpoint

    print(f"Loading: {checkpoint}")
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)

    print(f"sizing_q_enabled (checkpoint flag): {ckpt.get('sizing_q_enabled')}")
    cfg = ckpt.get("config", {})
    print(f"config keys: {list(cfg.keys())[:20]}")
    for k in ("sizing_q_enabled", "sizing_q_min_total_samples", "sizing_cfr_mode",
              "sizing_advantage_buffer_size", "use_multi_agent_advantage"):
        if k in cfg:
            print(f"  config[{k}] = {cfg[k]}")

    # --- sizing_q_net веса ---
    sd_q = ckpt["sizing_q_net"]
    print("\n=== sizing_q_net weights ===")
    for name, t in sd_q.items():
        nm = float(t.float().norm().item())
        print(f"  {name:30s} shape={tuple(t.shape)}  L2norm={nm:.6f}")

    # --- sizing_advantage_net веса (для контраста) ---
    sd_a = ckpt["advantage_sizing_net"]
    print("\n=== advantage_sizing_net weights (для контраста) ===")
    for name, t in sd_a.items():
        nm = float(t.float().norm().item())
        print(f"  {name:30s} shape={tuple(t.shape)}  L2norm={nm:.6f}")

    # --- буфер ---
    n = min(int(ckpt["sizing_advantage_buffer_cur_id"]),
            len(ckpt["sizing_advantage_buffer_states"]))
    sd = int(ckpt["sizing_advantage_buffer_state_dim"])
    states = np.asarray(ckpt["sizing_advantage_buffer_states"])[:n, :sd].astype(np.float32)
    regrets = np.asarray(ckpt["sizing_advantage_buffer_regrets"])[:n, :15].astype(np.float64)
    masks = np.asarray(ckpt["sizing_advantage_buffer_masks"])[:n, :15].astype(bool)
    finite = np.isfinite(states).all(1) & np.isfinite(regrets).all(1) & masks.any(1)
    states = states[finite]
    print(f"\nvalid rows: {len(states)}")

    # --- Прогон sizing_q_net на N состояниях ---
    q_net = load_sizing_q(sd_q)
    N = 2000
    idx = RNG.choice(len(states), N, replace=False)
    q_vals = eval_q(q_net, states[idx], ANCHORS)  # [N, 15]

    print(f"\n=== Q(state, size) на {N} состояниях ===")
    print(f"  global mean={q_vals.mean():.4f} std={q_vals.std():.4f}")
    print(f"  per-anchor std (across states): mean={q_vals.std(0).mean():.4f} max={q_vals.std(0).max():.4f}")
    print(f"  per-anchor mean: {np.round(q_vals.mean(0), 4)}")
    print(f"  per-anchor std:  {np.round(q_vals.std(0), 4)}")

    # --- Sanity: Q на двух далёких состояниях ---
    sub = states[:40000]
    p0 = int(RNG.integers(0, len(sub)))
    d2 = ((sub - sub[p0]) ** 2).sum(1)
    p1 = int(np.argmax(d2))
    q2 = eval_q(q_net, sub[[p0, p1]], ANCHORS)
    print(f"\n=== Sanity Q: 2 далёких состояния (dist={d2[p1]**.5:.3f}) ===")
    print(f"  Q(state0) = {np.round(q2[0], 4)}")
    print(f"  Q(state1) = {np.round(q2[1], 4)}")
    print(f"  |diff|    = {np.round(np.abs(q2[0]-q2[1]), 4)}  max={np.abs(q2[0]-q2[1]).max():.4f}")

    # --- Сравнение: Q vs regrets в буфере ---
    # q_vals[i, j] = Q(state_i, anchor_j). regrets[i, j] = stored target.
    # Если Q объясняет regrets, corr должна быть высокой.
    reg_sub = regrets[idx]
    mask_sub = masks[idx]
    q_flat = q_vals[mask_sub]
    r_flat = reg_sub[mask_sub]
    print(f"\n=== corr(Q_pred, stored_regret) на валидных анкорах ===")
    print(f"  n={len(q_flat)}")
    print(f"  Q:     mean={q_flat.mean():.4f} std={q_flat.std():.4f}")
    print(f"  regret: mean={r_flat.mean():.4f} std={r_flat.std():.4f}")
    if q_flat.std() > 1e-8 and r_flat.std() > 1e-8:
        c = np.corrcoef(q_flat, r_flat)[0, 1]
        print(f"  corr(Q, regret) = {c:.4f}")
    else:
        print(f"  Q std ~0 — константа!")

    # --- sizing_q_buffer: что там? ---
    if "sizing_q_buffer_states" in ckpt:
        qbs = np.asarray(ckpt["sizing_q_buffer_states"])
        qbt = np.asarray(ckpt["sizing_q_buffer_targets"])
        print(f"\n=== sizing_q_buffer ===")
        print(f"  states: {qbs.shape}  targets: {qbt.shape}")
        print(f"  target: mean={qbt.mean():.4f} std={qbt.std():.4f} min={qbt.min():.4f} max={qbt.max():.4f}")
        print(f"  state col std: median={np.median(qbs.std(0)):.4f}")


if __name__ == "__main__":
    main()
