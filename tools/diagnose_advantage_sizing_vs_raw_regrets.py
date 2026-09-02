#!/usr/bin/env python3
"""Compare advantage-sizing anchor logits with raw sizing regrets at iteration 300."""
from __future__ import annotations

import argparse

import json
import os
import sys
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from core.deep_cfr import DeepCFRAgent
from solver.benchmark_multiway_river import controlled_s3c_multiboard_states
from tools.sizing_report_naming import extract_iteration, report_path

TARGET = ROOT / "sizing_current_api_diagnostic.json"
CHECKPOINT = ROOT / "models" / "test107" / "multi_checkpoint_iter_300.pt"
OUT_JSON = ROOT / "advantage_sizing_vs_raw_regrets.json"
OUT_TXT = ROOT / "advantage_sizing_vs_raw_regrets.txt"
ANCHOR_COUNT = 15
EPS = 1e-12


def json_default(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if torch.is_tensor(value):
        return value.detach().cpu().tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Not JSON serializable: {type(value).__name__}")


def extract_advantage_sizing_state_dict(checkpoint: Mapping[str, Any]) -> dict[str, Any]:
    """Extract advantage_sizing_net weights from common checkpoint layouts."""
    if not isinstance(checkpoint, Mapping):
        raise TypeError("Checkpoint must be a mapping")

    candidates: list[Mapping[str, Any]] = []
    for key in ("advantage_sizing_net", "advantage_sizing_net_state_dict", "state_dict", "model_state_dict"):
        value = checkpoint.get(key)
        if isinstance(value, Mapping):
            candidates.append(value)
    for wrapper in ("model", "models", "networks", "agent"):
        value = checkpoint.get(wrapper)
        if isinstance(value, Mapping):
            nested = value.get("advantage_sizing_net")
            if isinstance(nested, Mapping):
                candidates.append(nested)
            candidates.append(value)

    prefixes = (
        "module.advantage_sizing_net.",
        "agent.advantage_sizing_net.",
        "model.advantage_sizing_net.",
        "advantage_sizing_net.",
        "module.",
    )
    for candidate in candidates:
        if candidate and all(torch.is_tensor(v) for v in candidate.values()):
            keys = [str(k) for k in candidate]
            for prefix in prefixes:
                selected = {str(k)[len(prefix):]: v for k, v in candidate.items() if str(k).startswith(prefix)}
                if selected:
                    return selected
            if not any("advantage_sizing_net." in key for key in keys):
                return dict(candidate)
    raise KeyError("Could not extract checkpoint['advantage_sizing_net'] state_dict")


def rankdata(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(values.size, dtype=np.float64)
    sorted_values = values[order]
    start = 0
    while start < values.size:
        end = start + 1
        while end < values.size and sorted_values[end] == sorted_values[start]:
            end += 1
        ranks[order[start:end]] = (start + end - 1) / 2.0 + 1.0
        start = end
    return ranks


def pearson(x: np.ndarray, y: np.ndarray) -> float | None:
    x, y = np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)
    if x.size < 2 or np.std(x) <= EPS or np.std(y) <= EPS:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def spearman(x: np.ndarray, y: np.ndarray) -> float | None:
    return pearson(rankdata(x), rankdata(y))


def metrics(logits: np.ndarray, targets: np.ndarray) -> dict[str, Any]:
    error = logits - targets
    denom = float(np.linalg.norm(logits) * np.linalg.norm(targets))
    return {
        "count": int(logits.size),
        "pearson": pearson(logits, targets),
        "spearman": spearman(logits, targets),
        "cosine": float(np.dot(logits, targets) / denom) if denom > EPS else None,
        "sign_agreement": float(np.mean(np.sign(logits) == np.sign(targets))) if logits.size else None,
        "mae": float(np.mean(np.abs(error))) if logits.size else None,
        "rmse": float(np.sqrt(np.mean(error * error))) if logits.size else None,
    }


def softmax_masked(values: np.ndarray, mask: np.ndarray) -> np.ndarray:
    result = np.zeros(ANCHOR_COUNT, dtype=np.float64)
    available = np.flatnonzero(mask)
    shifted = values[available] - np.max(values[available])
    weights = np.exp(shifted)
    result[available] = weights / weights.sum()
    return result


def regret_matching(values: np.ndarray, mask: np.ndarray) -> np.ndarray:
    result = np.zeros(ANCHOR_COUNT, dtype=np.float64)
    available = np.flatnonzero(mask)
    positive = np.maximum(values[available], 0.0)
    result[available] = positive / positive.sum() if positive.sum() > EPS else 1.0 / available.size
    return result


def tv(a: np.ndarray, b: np.ndarray) -> float:
    return float(0.5 * np.abs(a - b).sum())


def summary(values: list[float]) -> dict[str, Any]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "count": int(array.size),
        "min": float(array.min()),
        "mean": float(array.mean()),
        "median": float(np.median(array)),
        "max": float(array.max()),
        "population_std": float(array.std(ddof=0)),
    }


def top_metrics(logits: np.ndarray, targets: np.ndarray, available: np.ndarray) -> dict[str, Any]:
    pred_order = available[np.argsort(-logits, kind="mergesort")]
    target_order = available[np.argsort(-targets, kind="mergesort")]
    pred3, target3 = set(pred_order[:3].tolist()), set(target_order[:3].tolist())
    intersection, union = pred3 & target3, pred3 | target3
    return {
        "top1_agreement": bool(pred_order[0] == target_order[0]),
        "top1_logit_anchor": int(pred_order[0]),
        "top1_target_anchor": int(target_order[0]),
        "top3_logit_anchors": pred_order[:3].astype(int).tolist(),
        "top3_target_anchors": target_order[:3].astype(int).tolist(),
        "top3_overlap_count": len(intersection),
        "top3_overlap_fraction": float(len(intersection) / min(3, available.size)),
        "top3_jaccard": float(len(intersection) / len(union)),
        "rank_correlation": spearman(logits, targets),
    }


def run_model(agent: DeepCFRAgent, state: Any) -> np.ndarray:
    """Anchor-логиты advantage_sizing_net.

    `SizingAnchorNet.forward` возвращает ОДИН тензор [B, K]; bucket-головы у неё нет
    (#107 §4: смена контракта форм). Кортеж допускается только для обратной совместимости
    со старыми чекпоинтами, как в diagnose_sizing_flatness.py.
    """
    encoded = agent._encode_state_for_sizing(state, int(state.current_player))
    tensor = torch.as_tensor(encoded, dtype=torch.float32, device="cpu")
    with torch.inference_mode():
        result = agent.advantage_sizing_net(tensor.unsqueeze(0))
    anchor_t = result[-1] if isinstance(result, (tuple, list)) else result
    anchor_logits = anchor_t[0].detach().cpu().numpy().astype(np.float64)
    if anchor_logits.size != ANCHOR_COUNT:
        raise RuntimeError(f"Expected {ANCHOR_COUNT} anchor logits, got {anchor_logits.size}")
    return anchor_logits


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default=str(CHECKPOINT.relative_to(ROOT)))
    parser.add_argument("--json-output", help="Explicit JSON output path")
    parser.add_argument("--txt-output", help="Explicit TXT output path")
    args = parser.parse_args()
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.is_absolute():
        checkpoint_path = ROOT / checkpoint_path

    print(f"Loading target rows: {TARGET}")
    document = json.loads(TARGET.read_text(encoding="utf-8"))
    states = document.get("states")
    controlled_states = controlled_s3c_multiboard_states()
    if not isinstance(states, list) or len(states) != 24 or len(controlled_states) != 24:
        raise RuntimeError(f"Expected 24 rows and controlled states; rows={len(states) if isinstance(states, list) else 'invalid'}, states={len(controlled_states)}")

    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, weights_only=False, map_location="cpu")
    state_dict = extract_advantage_sizing_state_dict(checkpoint)
    agent = DeepCFRAgent(player_id=0, num_players=6, device="cpu")
    agent.advantage_sizing_net.load_state_dict(state_dict)
    agent.advantage_sizing_net.eval()

    rows: list[dict[str, Any]] = []
    all_logits: list[float] = []
    all_targets: list[float] = []
    logits_matrix = np.full((24, ANCHOR_COUNT), np.nan)
    targets_matrix = np.full((24, ANCHOR_COUNT), np.nan)
    masks: list[np.ndarray] = []
    target_distributions: list[np.ndarray] = []
    logit_distributions: list[np.ndarray] = []

    for index, (state, target_row) in enumerate(zip(controlled_states, states)):
        print(f"Processing state {index + 1}/24")
        anchor_logits = run_model(agent, state)
        targets = np.asarray(target_row["raw_q_anchor_minus_ev_sizing"], dtype=np.float64)
        mask = np.asarray(target_row["available_anchor_mask"], dtype=bool)
        if targets.size != ANCHOR_COUNT or mask.size != ANCHOR_COUNT or not mask.any():
            raise RuntimeError(f"Invalid target or mask at state {index + 1}")
        available = np.flatnonzero(mask)
        pred, truth = anchor_logits[available], targets[available]
        logits_matrix[index, mask] = anchor_logits[mask]
        targets_matrix[index, mask] = targets[mask]
        masks.append(mask)
        all_logits.extend(pred.tolist())
        all_targets.extend(truth.tolist())
        target_p = regret_matching(targets, mask)
        logit_p = softmax_masked(anchor_logits, mask)
        target_distributions.append(target_p)
        logit_distributions.append(logit_p)
        rows.append({
            "state": int(target_row.get("state", index + 1)),
            "available_anchor_indices": available.astype(int).tolist(),
            "available_anchor_mask": mask.tolist(),
            "anchor_logits": anchor_logits.tolist(),
            "raw_q_anchor_minus_ev_sizing": targets.tolist(),
            "bucket_logits_note": "advantage_sizing_net (SizingAnchorNet) has no bucket head: it returns a single [B, K] tensor. See #107 section 4.",
            "direct_metrics": metrics(pred, truth),
            "ranking": top_metrics(pred, truth, available),
            "target_regret_matching_distribution": target_p.tolist(),
            "masked_softmax_logit_distribution": logit_p.tolist(),
        })

    global_logits = np.asarray(all_logits, dtype=np.float64)
    global_targets = np.asarray(all_targets, dtype=np.float64)
    design = np.column_stack((global_logits, np.ones(global_logits.size)))
    slope, intercept = np.linalg.lstsq(design, global_targets, rcond=None)[0]
    calibrated = slope * global_logits + intercept
    cursor = 0
    for row in rows:
        count = len(row["available_anchor_indices"])
        row_calibrated = calibrated[cursor:cursor + count]
        row_targets = global_targets[cursor:cursor + count]
        row["calibrated_anchor_targets"] = row_calibrated.tolist()
        row["calibrated_errors"] = metrics(row_calibrated, row_targets)
        cursor += count

    per_anchor = []
    for anchor in range(ANCHOR_COUNT):
        valid = np.isfinite(logits_matrix[:, anchor]) & np.isfinite(targets_matrix[:, anchor])
        per_anchor.append({
            "anchor_index": anchor,
            "board_count": int(valid.sum()),
            "logit_population_std": float(np.std(logits_matrix[valid, anchor], ddof=0)) if valid.any() else None,
            "target_population_std": float(np.std(targets_matrix[valid, anchor], ddof=0)) if valid.any() else None,
        })

    target_pair_tv: list[float] = []
    logit_pair_tv: list[float] = []
    pairs: list[dict[str, Any]] = []
    for i in range(24):
        for j in range(i + 1, 24):
            common = masks[i] & masks[j]
            if not common.any():
                raise RuntimeError(f"No common available anchors for states {i + 1}, {j + 1}")
            target_i = regret_matching(np.asarray(states[i]["raw_q_anchor_minus_ev_sizing"], dtype=np.float64), common)
            target_j = regret_matching(np.asarray(states[j]["raw_q_anchor_minus_ev_sizing"], dtype=np.float64), common)
            logit_i = softmax_masked(logits_matrix[i], common)
            logit_j = softmax_masked(logits_matrix[j], common)
            target_value, logit_value = tv(target_i, target_j), tv(logit_i, logit_j)
            target_pair_tv.append(target_value)
            logit_pair_tv.append(logit_value)
            pairs.append({"state_a": i + 1, "state_b": j + 1, "common_available_count": int(common.sum()), "target_regret_matching_tv": target_value, "masked_softmax_logits_tv": logit_value})
    if len(pairs) != 276:
        raise RuntimeError(f"Expected 276 state pairs, got {len(pairs)}")

    report = {
        "checkpoint": str(checkpoint_path.relative_to(ROOT)).replace("\\", "/") if checkpoint_path.is_relative_to(ROOT) else str(checkpoint_path),
        "target_json": TARGET.name,
        "state_count": 24,
        "pair_count": 276,
        "comparison": "15 anchor logits are compared index-to-index with raw_q_anchor_minus_ev_sizing under available_anchor_mask.",
        "bucket_logits": "None: advantage_sizing_net returns a single anchor-logit tensor and has no bucket head.",
        "global_direct_metrics": metrics(global_logits, global_targets),
        "global_affine_calibration": {
            "formula": "target = slope * logit + intercept",
            "slope": float(slope),
            "intercept": float(intercept),
            "calibrated_metrics": metrics(calibrated, global_targets),
        },
        "aggregate_ranking": {
            "top1_agreement_rate": float(np.mean([row["ranking"]["top1_agreement"] for row in rows])),
            "mean_top3_overlap_fraction": float(np.mean([row["ranking"]["top3_overlap_fraction"] for row in rows])),
            "mean_top3_jaccard": float(np.mean([row["ranking"]["top3_jaccard"] for row in rows])),
            "mean_per_board_rank_correlation": float(np.mean([row["ranking"]["rank_correlation"] for row in rows if row["ranking"]["rank_correlation"] is not None])),
        },
        "per_anchor_cross_board_variation": per_anchor,
        "pairwise_tv": {
            "definition": "For each pair, distributions are recomputed on the intersection of available anchors; targets use positive-regret matching with uniform fallback, logits use masked softmax.",
            "target_regret_matching": summary(target_pair_tv),
            "masked_softmax_logits": summary(logit_pair_tv),
        },
        "states": rows,
        "pairs": pairs,
    }
    iteration = extract_iteration(checkpoint, checkpoint_path)
    out_json = report_path(OUT_JSON, iteration=iteration, explicit_output=args.json_output)
    out_txt = report_path(OUT_TXT, iteration=iteration, explicit_output=args.txt_output)
    if not out_json.is_absolute(): out_json = ROOT / out_json
    if not out_txt.is_absolute(): out_txt = ROOT / out_txt
    out_json.write_text(json.dumps(report, indent=2, default=json_default), encoding="utf-8")

    lines = [
        f"Advantage sizing logits vs raw sizing regrets (iteration {iteration})",
        f"Checkpoint: {report['checkpoint']}",
        "States: 24; pairs: 276; anchors: 15; excluded bucket logits: 3",
        f"Global direct metrics: {json.dumps(report['global_direct_metrics'], default=json_default)}",
        f"Affine calibration: {json.dumps(report['global_affine_calibration'], default=json_default)}",
        f"Ranking: {json.dumps(report['aggregate_ranking'], default=json_default)}",
        f"Target regret-matching pairwise TV: {json.dumps(report['pairwise_tv']['target_regret_matching'], default=json_default)}",
        f"Masked-softmax-logit pairwise TV: {json.dumps(report['pairwise_tv']['masked_softmax_logits'], default=json_default)}",
    ]
    out_txt.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote JSON: {out_json}")
    print(f"Wrote TXT:  {out_txt}")


if __name__ == "__main__":
    main()
