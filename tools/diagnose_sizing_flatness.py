"""Диагностика «плоскости» advantage_sizing_net на полном iter_200 чекпоинте.

Цель: разделить две гипотезы про flat-функцию:
  A) таргеты (sizing_cf_regrets) фактически НЕ зависят от состояния;
  B) сеть не дообучена, хотя таргеты несут сигнал.

Проверки:
  1. Разнообразие самих состояний в буфере (не всё ли одинаковое).
  2. Sanity: логиты двух далёких состояний различаются хотя бы слегка.
  3. Дисперсия таргетов ВНУТРИ группы близких состояний vs СЛУЧАЙНОЙ
     группы. Если внутри ≈ случайно ≈ global — таргеты state-independent.
  4. anchor_head.bias против маргинальных средних таргетов по анкорам.
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

from core.model import SizingAnchorNet  # noqa: E402
from tools.sizing_report_naming import extract_iteration, report_path  # noqa: E402

CKPT = ROOT / "models" / "test107" / "multi_checkpoint_iter_200.pt"
ANCHORS = 15
RNG = np.random.default_rng(31337)


def load_net(state_dict: dict) -> torch.nn.Module:
    shapes = {k: tuple(v.shape) for k, v in state_dict.items() if torch.is_tensor(v)}
    net = SizingAnchorNet(
        input_size=shapes["base.0.weight"][1],
        hidden_size=shapes["base.0.weight"][0],
        num_sizes=shapes["anchor_head.weight"][0],
        num_buckets=3,
    )
    res = net.load_state_dict(state_dict, strict=True)
    if res.missing_keys or res.unexpected_keys:
        raise RuntimeError(f"load mismatch: {res}")
    net.eval()
    return net


def infer_anchors(net: torch.nn.Module, states: np.ndarray, batch: int = 4096) -> np.ndarray:
    """Возвращает [N, ANCHORS] anchor-логиты по батчам (кортеж из forward разворачивается верно)."""
    out: list[np.ndarray] = []
    with torch.no_grad():
        for i in range(0, len(states), batch):
            t = torch.from_numpy(states[i : i + batch]).float()
            result = net(t)
            anchor_t = result[1] if isinstance(result, (tuple, list)) else result
            out.append(anchor_t.detach().cpu().numpy())
    return np.concatenate(out, axis=0)


def _json_safe(value):
    """Recursively convert NumPy/Torch values to JSON-serializable Python values."""
    if torch.is_tensor(value):
        return _json_safe(value.detach().cpu().numpy())
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def write_report(report: dict, output: str | Path) -> tuple[Path, Path]:
    """Write JSON and a human-readable sibling TXT, returning absolute paths."""
    json_path = Path(output).expanduser()
    if not json_path.is_absolute():
        json_path = ROOT / json_path
    json_path = json_path.resolve()
    txt_path = json_path.with_suffix(".txt")
    json_path.parent.mkdir(parents=True, exist_ok=True)

    safe_report = _json_safe(report)
    json_text = json.dumps(safe_report, ensure_ascii=False, indent=2, allow_nan=False)
    txt_text = "Sizing flatness diagnostic report\n\n" + json_text + "\n"
    json_path.write_text(json_text + "\n", encoding="utf-8")
    txt_path.write_text(txt_text, encoding="utf-8")
    return json_path, txt_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Diagnose sizing-network flatness and save JSON/TXT reports."
    )
    parser.add_argument("--checkpoint", default=str(CKPT.relative_to(ROOT)))
    parser.add_argument("--output", help="JSON output path (relative paths resolve from project root).")
    args = parser.parse_args()
    checkpoint = Path(args.checkpoint)
    if not checkpoint.is_absolute():
        checkpoint = ROOT / checkpoint

    print(f"Loading checkpoint: {checkpoint}")
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    print("top-level keys (%d):" % len(ckpt))
    print("  " + ", ".join(sorted(ckpt.keys())))

    key_map = {
        "states": "sizing_advantage_buffer_states",
        "regrets": "sizing_advantage_buffer_regrets",
        "masks": "sizing_advantage_buffer_masks",
        "iterations": "sizing_advantage_buffer_iterations",
    }
    buf = {}
    for short, full in key_map.items():
        v = np.asarray(ckpt[full])
        print(f"  {short}: shape={v.shape} dtype={v.dtype}")
        buf[short] = v

    cur_id = int(ckpt["sizing_advantage_buffer_cur_id"])
    state_dim = int(ckpt["sizing_advantage_buffer_state_dim"])
    print(f"  cur_id={cur_id} state_dim={state_dim}")

    n = max(0, min(cur_id, len(buf["states"])))
    states = buf["states"][:n, :state_dim].astype(np.float32, copy=False)
    regrets = buf["regrets"][:n, :ANCHORS].astype(np.float64, copy=False)
    masks = buf["masks"][:n, :ANCHORS].astype(bool, copy=False)
    iters = buf["iterations"].reshape(-1)[:n].astype(np.int64, copy=False)

    finite = (np.isfinite(states).all(axis=1)
              & np.isfinite(regrets).all(axis=1) & masks.any(axis=1))
    states, regrets, masks, iters = states[finite], regrets[finite], masks[finite], iters[finite]
    print(f"valid rows: {states.shape[0]}")
    if len(states) < 100:
        raise RuntimeError("недостаточно валидных строк")

    # --- 1. Разнообразие состояний ---
    col_std = states.std(axis=0)
    print("\n[1] Разнообразие состояний в буфере")
    print(f"  col std: min={col_std.min():.3e} median={np.median(col_std):.3e} "
          f"max={col_std.max():.3e}, живие колонки (>1e-4): {100*np.mean(col_std > 1e-4):.2f}%")

    # --- 2. Загрузка сети и sanity ---
    net = load_net(ckpt["advantage_sizing_net"])
    bias = net.anchor_head.bias.detach().cpu().numpy()
    w_norm = float(net.anchor_head.weight.detach().norm().item())

    sub = states[:40000]
    p0 = int(RNG.integers(0, len(sub)))
    d2 = ((sub - sub[p0]) ** 2).sum(axis=1)
    p1 = int(np.argmax(d2))
    print("\n[2] Sanity: два далёких состояния")
    print(f"  dist(p0,p1)={float(d2[p1]) ** 0.5:.3f}")
    lg = infer_anchors(net, sub[[p0, p1]])
    print(f"  logit0={np.round(lg[0], 5)}")
    print(f"  logit1={np.round(lg[1], 5)}")
    print(f"  |diff|={np.round(np.abs(lg[0] - lg[1]), 5)}  max={np.abs(lg[0]-lg[1]).max():.5f}")

    # --- 3. Разброс логитов сети по 8k состояниям ---
    samp8k = RNG.choice(len(states), size=8000, replace=False)
    lg8k = infer_anchors(net, states[samp8k])
    print("\n[3] Разброс логидов сети по 8000 состояниям")
    print(f"  anchor_logit std по строкам: mean={lg8k.std(axis=0).mean():.5f} "
          f"max={lg8k.std(axis=0).max():.5f}, overall={lg8k.std():.5f}")

    # --- 4. ВНУТРИ близких vs случайная дисперсия таргетов ---
    print("\n[4] Таргет-дисперсия: близкие соседи vs случайные vs global")
    pool = states[:min(len(states), 60000)]
    rig_pool = regrets[:len(pool)]
    mask_pool = masks[:len(pool)]
    probes = RNG.choice(len(pool), size=400, replace=False)
    cand_pool = np.setdiff1d(np.arange(len(pool)), probes)
    n_nbr = 8

    t_all = rig_pool[mask_pool]
    print(f"  global std таргета (по валидным анкорам): {t_all.std():.4f}")

    within = []
    randm = []
    within_absd = []
    rand_absd = []
    prox_distance = []
    for p in probes:
        s = pool[p]
        d2 = ((pool - s) ** 2).sum(axis=1)
        d2[probes] = np.inf  # пробы исключаем из соседей
        nbr = np.argpartition(d2, n_nbr)[:n_nbr]
        prox_distance.append(float(np.sqrt(d2[nbr].mean())))
        for j in range(ANCHORS):
            valid = mask_pool[p, j] & mask_pool[nbr, j]
            if not valid.any():
                continue
            tj = rig_pool[nbr[valid], j]
            within.append(float(tj.std()))
            within_absd.append(float(np.mean(np.abs(tj - rig_pool[p, j]))))
            ri = RNG.choice(cand_pool, size=n_nbr, replace=False)
            valid_r = mask_pool[p, j] & mask_pool[ri, j]
            if valid_r.any():
                tr = rig_pool[ri[valid_r], j]
                randm.append(float(tr.std()))
                rand_absd.append(float(np.mean(np.abs(tr - rig_pool[p, j]))))

    within = np.asarray(within)
    randm = np.asarray(randm)
    prox_distance = np.asarray(prox_distance)
    print(f"  probes={len(probes)}, n_nbr={n_nbr}")
    print(f"  mean dist до соседа: {prox_distance.mean():.4f}")
    print(f"  std таргета ВНУТРИ близких:  mean={within.mean():.4f} median={np.median(within):.4f}  (n={len(within)})")
    print(f"  std таргета СЛУЧАЙНЫХ:      mean={randm.mean():.4f} median={np.median(randm):.4f}  (n={len(randm)})")
    ratio = within.mean() / (randm.mean() + 1e-12) if len(randm) else float("nan")
    print(f"  ratio внутри/случайные = {ratio:.3f}  (1.0 = ближайшие соседи не несут сигнала больше случайности)")
    print(f"  |d-target| внутри близких: {np.mean(within_absd):.4f} vs случайных: {np.mean(rand_absd):.4f}  "
          f"ratio={np.mean(within_absd)/(np.mean(rand_absd) + 1e-12):.3f}")

    # --- 5. Bias головы vs маргинальные средние таргетов ---
    t_mean_j = np.array([
        float(np.mean(regrets[masks[:, j], j])) if masks[:, j].any() else float("nan")
        for j in range(ANCHORS)
    ])
    print("\n[5] anchor_head.bias vs маргинальные средние таргетов")
    print(f"  bias        = {np.round(bias, 5)}")
    print(f"  target_mean = {np.round(t_mean_j, 5)}")
    corr = np.corrcoef(bias, t_mean_j)[0, 1]
    print(f"  corr(bias, target_mean) = {corr:.4f}")

    # --- Итерационный профиль буфера ---
    print("\n[6] Профиль буфера по итерациям")
    it_min, it_max = int(iters.min()), int(iters.max())
    print(f"  iteration range: {it_min}..{it_max}")
    edges = np.quantile(iters, [0, .2, .4, .6, .8, 1.0])
    iteration_profile = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        sel = (iters >= lo) & (iters <= hi)
        mean_target = float(regrets[sel][masks[sel]].mean())
        print(f"  iter [{int(lo)},{int(hi)}]: rows={sel.sum()} "
              f"masked_anchors={int(masks[sel].sum())} mean_target={mean_target:.3f}")
        iteration_profile.append({
            "lo": int(lo), "hi": int(hi), "rows": int(sel.sum()),
            "masked_anchors": int(masks[sel].sum()), "mean_target": mean_target,
        })

    report = {
        "checkpoint": str(checkpoint.resolve()),
        "valid_rows": int(len(states)),
        "state_column_std": col_std,
        "anchor_head_bias": bias,
        "anchor_head_weight_norm": w_norm,
        "target_mean_by_anchor": t_mean_j,
        "bias_target_correlation": corr,
        "neighbor_random_std_ratio": ratio,
        "iteration_range": [it_min, it_max],
        "iteration_profile": iteration_profile,
    }
    output = args.output or report_path(checkpoint, "sizing_flatness", extract_iteration(checkpoint))
    json_path, txt_path = write_report(report, output)
    print(f"\nSaved JSON: {json_path}")
    print(f"Saved TXT:  {txt_path}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ERROR: analysis failed; no successful report was saved: {exc}", file=sys.stderr)
        raise