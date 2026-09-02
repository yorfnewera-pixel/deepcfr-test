"""Diagnose stored sizing-buffer regret targets against checkpoint sizing logits.

No Q-derived reconstruction is attempted because the required Q fields are absent.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from core.deep_cfr import DeepCFRAgent  # noqa: E402
from tools.sizing_report_naming import extract_iteration, report_path  # noqa: E402

REQUIRED_BUFFER_KEYS = (
    "sizing_advantage_buffer_states",
    "sizing_advantage_buffer_regrets",
    "sizing_advantage_buffer_masks",
    "sizing_advantage_buffer_iterations",
    "sizing_advantage_buffer_cur_id",
    "sizing_advantage_buffer_state_dim",
)
ANCHORS = 15


def jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def summary(x: np.ndarray) -> dict[str, Any]:
    x = np.asarray(x, dtype=np.float64)
    x = x[np.isfinite(x)]
    if not x.size:
        return {"count": 0}
    return {
        "count": int(x.size),
        "mean": float(x.mean()),
        "std": float(x.std()),
        "min": float(x.min()),
        "p05": float(np.quantile(x, 0.05)),
        "p25": float(np.quantile(x, 0.25)),
        "median": float(np.quantile(x, 0.50)),
        "p75": float(np.quantile(x, 0.75)),
        "p95": float(np.quantile(x, 0.95)),
        "max": float(x.max()),
    }


def rankdata(x: np.ndarray) -> np.ndarray:
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(len(x), dtype=np.float64)
    sorted_x = x[order]
    start = 0
    while start < len(x):
        end = start + 1
        while end < len(x) and sorted_x[end] == sorted_x[start]:
            end += 1
        ranks[order[start:end]] = 0.5 * (start + end - 1) + 1.0
        start = end
    return ranks


def pearson(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 2 or np.std(x) == 0 or np.std(y) == 0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    return pearson(rankdata(x), rankdata(y))


def cosine(x: np.ndarray, y: np.ndarray) -> float:
    denom = float(np.linalg.norm(x) * np.linalg.norm(y))
    return float(np.dot(x, y) / denom) if denom else float("nan")


def masked_softmax(x: np.ndarray) -> np.ndarray:
    z = x - np.max(x)
    e = np.exp(z)
    return e / e.sum()


def regret_matching(x: np.ndarray) -> np.ndarray:
    p = np.maximum(x, 0.0)
    return p / p.sum() if p.sum() > 0 else np.full(len(x), 1.0 / len(x))


def sample_metrics(target: np.ndarray, logit: np.ndarray) -> dict[str, float]:
    error = logit - target
    target_order = np.argsort(-target, kind="mergesort")
    logit_order = np.argsort(-logit, kind="mergesort")
    target_rank = rankdata(-target)
    logit_rank = rankdata(-logit)
    return {
        "pearson": pearson(target, logit),
        "spearman": spearman(target, logit),
        "cosine": cosine(target, logit),
        "sign_agreement": float(np.mean(np.sign(target) == np.sign(logit))),
        "mae": float(np.mean(np.abs(error))),
        "rmse": float(np.sqrt(np.mean(error * error))),
        "top1_match": float(target_order[0] == logit_order[0]),
        "top3_overlap": float(len(set(target_order[:3]) & set(logit_order[:3])) / min(3, len(target))),
        "mean_abs_rank_error": float(np.mean(np.abs(target_rank - logit_rank))),
        "tv_softmax_vs_regret_matching": float(
            0.5 * np.abs(masked_softmax(logit) - regret_matching(target)).sum()
        ),
    }


def aggregate_metrics(targets: np.ndarray, logits: np.ndarray, masks: np.ndarray) -> dict[str, float]:
    t = targets[masks]
    l = logits[masks]
    e = l - t
    return {
        "pearson": pearson(t, l),
        "spearman": spearman(t, l),
        "cosine": cosine(t, l),
        "sign_agreement": float(np.mean(np.sign(t) == np.sign(l))),
        "mae": float(np.mean(np.abs(e))),
        "rmse": float(np.sqrt(np.mean(e * e))),
    }


def infer_logits(net: torch.nn.Module, states: np.ndarray, batch_size: int = 4096) -> np.ndarray:
    """Return anchor logits for both current and legacy sizing-network outputs.

    Current SizingAnchorNet.forward returns a single [B, K] tensor.  Older
    architectures may return multiple tensors; select the [B, K] anchor head
    without changing the logits used by the metrics.
    """
    anchors = []
    with torch.no_grad():
        for start in range(0, len(states), batch_size):
            state_t = torch.as_tensor(states[start : start + batch_size], dtype=torch.float32)
            output = net(state_t)
            candidates = (output,) if isinstance(output, torch.Tensor) else tuple(output)
            anchor_t = next(
                (value for value in candidates
                 if isinstance(value, torch.Tensor) and value.ndim == 2 and value.shape[1] >= ANCHORS),
                None,
            )
            if anchor_t is None:
                shapes = [tuple(value.shape) for value in candidates if isinstance(value, torch.Tensor)]
                raise RuntimeError(f"Could not identify {ANCHORS}-anchor logits in network outputs: {shapes}")
            anchors.append(anchor_t.detach().cpu().numpy())
    return np.concatenate(anchors)


def grouped_summaries(iterations: np.ndarray, cardinality: np.ndarray, rows: list[dict[str, Any]]) -> dict[str, Any]:
    metric_names = [k for k in rows[0] if k not in {"sample_index", "iteration", "mask_cardinality"}]

    def collect(indices: np.ndarray) -> dict[str, Any]:
        return {
            "samples": int(len(indices)),
            "metrics": {name: summary(np.array([rows[i][name] for i in indices])) for name in metric_names},
        }

    by_cardinality = {str(k): collect(np.flatnonzero(cardinality == k)) for k in np.unique(cardinality)}
    by_iteration_bin: dict[str, Any] = {}
    if len(iterations):
        edges = np.unique(np.quantile(iterations, np.linspace(0, 1, 6)).astype(np.int64))
        if len(edges) == 1:
            by_iteration_bin[str(int(edges[0]))] = collect(np.arange(len(iterations)))
        else:
            for lo, hi in zip(edges[:-1], edges[1:]):
                selected = np.flatnonzero((iterations >= lo) & (iterations <= hi if hi == edges[-1] else iterations < hi))
                if len(selected):
                    by_iteration_bin[f"[{int(lo)}, {int(hi)}{']' if hi == edges[-1] else ')'}"] = collect(selected)
    return {"by_iteration_bin": by_iteration_bin, "by_mask_cardinality": by_cardinality}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default="models/test107/multi_checkpoint_iter_300.pt")
    parser.add_argument("--max-samples", type=int, default=0, help="0 means all valid samples")
    parser.add_argument("--seed", type=int, default=300)
    parser.add_argument("--json-output", help="Explicit JSON output path")
    parser.add_argument("--txt-output", help="Explicit TXT output path")
    args = parser.parse_args()

    checkpoint_path = (ROOT / args.checkpoint).resolve() if not Path(args.checkpoint).is_absolute() else Path(args.checkpoint)
    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, weights_only=False, map_location="cpu")
    missing_checkpoint_keys = [key for key in REQUIRED_BUFFER_KEYS if key not in checkpoint]
    if missing_checkpoint_keys:
        raise KeyError(f"Missing required checkpoint keys: {missing_checkpoint_keys}")

    states_raw = np.asarray(checkpoint["sizing_advantage_buffer_states"])
    regrets_raw = np.asarray(checkpoint["sizing_advantage_buffer_regrets"])
    masks_raw = np.asarray(checkpoint["sizing_advantage_buffer_masks"])
    iterations_raw = np.asarray(checkpoint["sizing_advantage_buffer_iterations"]).reshape(-1)
    cur_id = int(checkpoint["sizing_advantage_buffer_cur_id"])
    state_dim = int(checkpoint["sizing_advantage_buffer_state_dim"])
    valid_count = max(0, min(cur_id, len(states_raw), len(regrets_raw), len(masks_raw), len(iterations_raw)))
    states = states_raw[:valid_count, :state_dim].astype(np.float32, copy=False)
    regrets = regrets_raw[:valid_count, :ANCHORS].astype(np.float64, copy=False)
    masks = masks_raw[:valid_count, :ANCHORS].astype(bool, copy=False)
    iterations = iterations_raw[:valid_count].astype(np.int64, copy=False)

    finite = np.isfinite(states).all(axis=1) & np.isfinite(regrets).all(axis=1) & masks.any(axis=1)
    original_indices = np.flatnonzero(finite)
    states, regrets, masks, iterations = states[finite], regrets[finite], masks[finite], iterations[finite]
    if args.max_samples > 0 and len(states) > args.max_samples:
        rng = np.random.default_rng(args.seed)
        chosen = np.sort(rng.choice(len(states), size=args.max_samples, replace=False))
        states, regrets, masks, iterations, original_indices = (
            states[chosen], regrets[chosen], masks[chosen], iterations[chosen], original_indices[chosen]
        )
    if not len(states):
        raise RuntimeError("No finite buffer rows with at least one valid anchor")
    print(f"Using {len(states)} valid samples from {valid_count} sliced rows")

    agent = DeepCFRAgent(player_id=0, num_players=6, device="cpu")
    net = agent.advantage_sizing_net
    if "advantage_sizing_net" in checkpoint:
        weight_key = "advantage_sizing_net"
        state_dict = checkpoint[weight_key]
    elif "advantage_sizing_nets" in checkpoint and len(checkpoint["advantage_sizing_nets"]):
        weight_key = "advantage_sizing_nets[0]"
        state_dict = checkpoint["advantage_sizing_nets"][0]
    else:
        raise KeyError("Missing advantage_sizing_net and fallback advantage_sizing_nets[0]")
    load_result = net.load_state_dict(state_dict, strict=False)
    net.eval()
    flat_parameters = torch.cat([p.detach().cpu().reshape(-1).double() for p in net.parameters()])
    checksum = hashlib.sha256(flat_parameters.numpy().tobytes()).hexdigest()
    print(f"Loaded weights from {weight_key}; running direct stored-state inference")
    anchor_logits = infer_logits(net, states)
    if anchor_logits.ndim != 2 or anchor_logits.shape[1] < ANCHORS:
        raise RuntimeError(f"Expected at least {ANCHORS} anchor logits, got {anchor_logits.shape}")
    anchor_logits = anchor_logits[:, :ANCHORS].astype(np.float64, copy=False)

    rows: list[dict[str, Any]] = []
    for i in range(len(states)):
        valid = masks[i]
        metrics = sample_metrics(regrets[i, valid], anchor_logits[i, valid])
        rows.append({
            "sample_index": int(original_indices[i]),
            "iteration": int(iterations[i]),
            "mask_cardinality": int(valid.sum()),
            **metrics,
        })

    target_flat, logit_flat = regrets[masks], anchor_logits[masks]
    design = np.column_stack([logit_flat, np.ones_like(logit_flat)])
    slope, intercept = np.linalg.lstsq(design, target_flat, rcond=None)[0]
    calibrated = slope * logit_flat + intercept
    calibration = {
        "target_equals_slope_times_logit_plus_intercept": {"slope": float(slope), "intercept": float(intercept)},
        "mae_after": float(np.mean(np.abs(calibrated - target_flat))),
        "rmse_after": float(np.sqrt(np.mean((calibrated - target_flat) ** 2))),
    }

    per_anchor = []
    for anchor in range(ANCHORS):
        valid = masks[:, anchor]
        per_anchor.append({
            "anchor": anchor,
            "count": int(valid.sum()),
            "target_std": float(np.std(regrets[valid, anchor])) if valid.any() else None,
            "logit_std": float(np.std(anchor_logits[valid, anchor])) if valid.any() else None,
            "target_distribution": summary(regrets[valid, anchor]),
            "logit_distribution": summary(anchor_logits[valid, anchor]),
        })

    cardinality = masks.sum(axis=1)
    report = {
        "metadata": {
            "checkpoint": str(checkpoint_path),
            "required_checkpoint_keys": list(REQUIRED_BUFFER_KEYS),
            "weight_key": weight_key,
            "missing_keys": list(load_result.missing_keys),
            "unexpected_keys": list(load_result.unexpected_keys),
            "parameter_sha256": checksum,
            "parameter_l2_norm": float(torch.linalg.vector_norm(flat_parameters)),
            "buffer_cur_id": cur_id,
            "buffer_state_dim": state_dim,
            "sliced_rows": valid_count,
            "finite_masked_rows": int(finite.sum()),
            "sample_count": len(states),
            "masked_anchor_count": int(masks.sum()),
            "seed": args.seed,
            "max_samples": args.max_samples,
            "anchor_output_shape": list(anchor_logits.shape),
            "note": "No Q-derived reconstruction is attempted because required fields are absent; stored states are already _encode_state_for_sizing outputs.",
        },
        "aggregate": aggregate_metrics(regrets, anchor_logits, masks),
        "global_affine_calibration": calibration,
        "per_sample_metric_summaries": {
            key: summary(np.array([row[key] for row in rows]))
            for key in rows[0]
            if key not in {"sample_index", "iteration", "mask_cardinality"}
        },
        "per_anchor": per_anchor,
        "target_distribution": summary(target_flat),
        "logit_distribution": summary(logit_flat),
        "grouped": grouped_summaries(iterations, cardinality, rows),
        "samples": rows,
    }

    iteration = extract_iteration(checkpoint, checkpoint_path)
    json_path = report_path(ROOT / "sizing_buffer_targets.json", iteration=iteration, explicit_output=args.json_output)
    txt_path = report_path(ROOT / "sizing_buffer_targets.txt", iteration=iteration, explicit_output=args.txt_output)
    if not json_path.is_absolute(): json_path = ROOT / json_path
    if not txt_path.is_absolute(): txt_path = ROOT / txt_path
    json_path.write_text(json.dumps(jsonable(report), indent=2, allow_nan=False), encoding="utf-8")
    aggregate = report["aggregate"]
    txt_lines = [
        "Sizing buffer targets vs sizing-network anchor logits",
        f"checkpoint: {checkpoint_path}",
        f"weights: {weight_key}",
        f"samples: {len(states)}; masked anchors: {int(masks.sum())}",
        f"missing/unexpected keys: {list(load_result.missing_keys)} / {list(load_result.unexpected_keys)}",
        f"parameter sha256: {checksum}",
        "aggregate: " + ", ".join(f"{k}={v:.6g}" for k, v in aggregate.items()),
        f"affine target ~= {slope:.6g} * logit + {intercept:.6g}",
        "No Q-derived reconstruction is attempted because required fields are absent.",
        f"detailed JSON: {json_path}",
    ]
    txt_path.write_text("\n".join(txt_lines) + "\n", encoding="utf-8")
    print(f"Saved detailed JSON: {json_path}")
    print(f"Saved concise TXT: {txt_path}")


if __name__ == "__main__":
    main()

