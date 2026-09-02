#!/usr/bin/env python3
"""Offline-аудит покрытия и ранжирования sizing-Q.

Читает только checkpoint с сохранёнными sizing-Q и sizing-advantage replay.
Не запускает traversal и не меняет checkpoint. Grounded значения из holdout
используются исключительно для оценки ранжирования Q.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.core.buffers import SizingQBuffer  # noqa: E402
from src.core.model import SizingQNetwork  # noqa: E402


def _state_keys(states: np.ndarray) -> np.ndarray:
    return np.asarray([
        hashlib.blake2b(np.ascontiguousarray(row).view(np.uint8), digest_size=8).hexdigest()
        for row in states
    ])


def _load_q_network(state_dict: dict[str, torch.Tensor]) -> SizingQNetwork:
    embed_dim = int(state_dict['size_embed.weight'].shape[0])
    hidden_size = int(state_dict['base.0.weight'].shape[0])
    state_dim = int(state_dict['base.0.weight'].shape[1]) - embed_dim
    net = SizingQNetwork(state_dim, hidden_size, embed_dim)
    net.load_state_dict(state_dict, strict=True)
    net.eval()
    return net


def _evaluate(net: SizingQNetwork, states: np.ndarray, anchors: np.ndarray,
              min_bet: float, max_bet: float, batch_size: int) -> np.ndarray:
    normalized = (anchors - min_bet) / (max_bet - min_bet)
    values = []
    with torch.inference_mode():
        for start in range(0, len(states), batch_size):
            batch = torch.from_numpy(states[start:start + batch_size]).float()
            count = len(batch)
            expanded_states = batch.unsqueeze(1).expand(-1, len(anchors), -1).reshape(-1, batch.shape[1])
            expanded_sizes = torch.from_numpy(normalized).unsqueeze(0).expand(count, -1).reshape(-1).float()
            values.append(net(expanded_states, expanded_sizes).reshape(count, len(anchors)).numpy())
    return np.concatenate(values, axis=0) if values else np.empty((0, len(anchors)))


def _safe_mean(values: np.ndarray) -> float | None:
    return float(np.mean(values)) if values.size else None


def _ranking_metrics(predictions: np.ndarray, grounded: np.ndarray,
                     available: np.ndarray) -> dict[str, float | int | None]:
    pair_correct = 0
    pairs = 0
    top1_correct = 0
    topk_correct = 0
    eligible = 0
    for pred, truth, legal in zip(predictions, grounded, available, strict=True):
        valid = legal & np.isfinite(truth)
        indices = np.flatnonzero(valid)
        if len(indices) < 2:
            continue
        eligible += 1
        truth_values = truth[indices]
        pred_values = pred[indices]
        delta_truth = truth_values[:, None] - truth_values[None, :]
        delta_pred = pred_values[:, None] - pred_values[None, :]
        upper = np.triu(np.abs(delta_truth) > 1e-8, k=1)
        pairs += int(upper.sum())
        pair_correct += int(((delta_truth * delta_pred > 0) & upper).sum())
        best_truth = indices[truth_values == truth_values.max()]
        ranked = indices[np.argsort(pred_values)[::-1]]
        top1_correct += int(ranked[0] in best_truth)
        topk_correct += int(bool(set(ranked[:min(3, len(ranked))]) & set(best_truth)))
    return {
        'states': eligible,
        'pairs': pairs,
        'pairwise_accuracy': pair_correct / pairs if pairs else None,
        'top1_accuracy': top1_correct / eligible if eligible else None,
        'top3_accuracy': topk_correct / eligible if eligible else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', '-c', required=True, help='Путь к полному checkpoint.')
    parser.add_argument('--output', '-o', help='Путь к JSON-отчёту; по умолчанию рядом с checkpoint.')
    parser.add_argument('--batch-size', type=int, default=4096)
    args = parser.parse_args()

    checkpoint_path = Path(args.checkpoint).resolve()
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    required = ('sizing_q_net', 'sizing_q_buffer_states', 'sizing_q_buffer_targets',
                'sizing_q_buffer_anchor_indices', 'sizing_advantage_buffer_states',
                'sizing_advantage_buffer_grounded', 'sizing_advantage_buffer_avail_masks',
                'sizing_advantage_buffer_cur_id')
    missing = [key for key in required if key not in checkpoint]
    if missing:
        raise KeyError(f'Checkpoint не содержит данные для аудита: {missing}')

    config = checkpoint.get('config', {})
    anchors = np.asarray(
        checkpoint.get('anchors', config.get('fixed_sizing_grid', config.get('anchor_sizes'))),
        dtype=np.float32,
    )
    if anchors.ndim != 1 or not len(anchors):
        raise ValueError('Не найден корректный список anchors в checkpoint.')
    min_bet = float(checkpoint.get('min_bet_size', config.get('min_bet_size', 0.1)))
    max_bet = float(checkpoint.get('max_bet_size', config.get('max_bet_size', 3.0)))
    anchor_count = len(anchors)

    q_states = np.asarray(checkpoint['sizing_q_buffer_states'], dtype=np.float32)
    q_targets = np.asarray(checkpoint['sizing_q_buffer_targets'], dtype=np.float64)
    q_anchor_indices = np.asarray(checkpoint['sizing_q_buffer_anchor_indices'], dtype=np.int64)
    q_selected_indices = np.asarray(
        checkpoint.get('sizing_q_buffer_selected_anchor_indices', q_anchor_indices), dtype=np.int64)
    q_effective_indices = np.asarray(
        checkpoint.get('sizing_q_buffer_effective_anchor_indices', q_anchor_indices), dtype=np.int64)
    q_sources = np.asarray(checkpoint.get('sizing_q_buffer_sources', np.zeros(len(q_states))), dtype=np.int64)
    q_kinds = np.asarray(checkpoint.get('sizing_q_buffer_selected_kind_ids', np.zeros(len(q_states))), dtype=np.int64)
    q_count = min(len(q_states), len(q_targets), len(q_anchor_indices), len(q_selected_indices),
                  len(q_effective_indices), len(q_sources), len(q_kinds))
    q_states, q_targets, q_anchor_indices, q_selected_indices, q_effective_indices, q_sources, q_kinds = (
        q_states[:q_count], q_targets[:q_count], q_anchor_indices[:q_count], q_selected_indices[:q_count],
        q_effective_indices[:q_count], q_sources[:q_count], q_kinds[:q_count])

    net = _load_q_network(checkpoint['sizing_q_net'])
    q_predictions = _evaluate(net, q_states, anchors, min_bet, max_bet, args.batch_size)
    normalized_sizes = (anchors - min_bet) / (max_bet - min_bet)
    sample_predictions = np.full(q_count, np.nan)
    valid_q_anchor = (q_anchor_indices >= 0) & (q_anchor_indices < anchor_count)
    sample_predictions[valid_q_anchor] = q_predictions[np.arange(q_count)[valid_q_anchor], q_anchor_indices[valid_q_anchor]]
    q_errors = sample_predictions[valid_q_anchor] - q_targets[valid_q_anchor]

    q_state_keys = _state_keys(q_states)
    per_anchor = []
    for anchor_index, anchor in enumerate(anchors):
        rows = q_anchor_indices == anchor_index
        per_anchor.append({
            'anchor_index': anchor_index,
            'anchor': float(anchor),
            'samples': int(rows.sum()),
            'unique_states': int(len(set(q_state_keys[rows]))),
            'selected_samples': int((q_selected_indices == anchor_index).sum()),
            'effective_samples': int((q_effective_indices == anchor_index).sum()),
            'mse': _safe_mean(np.square(q_errors[q_anchor_indices[valid_q_anchor] == anchor_index])),
            'mae': _safe_mean(np.abs(q_errors[q_anchor_indices[valid_q_anchor] == anchor_index])),
        })

    state_dim = int(checkpoint.get('sizing_advantage_buffer_state_dim', q_states.shape[1]))
    adv_count = min(int(checkpoint['sizing_advantage_buffer_cur_id']),
                    len(checkpoint['sizing_advantage_buffer_states']))
    adv_states = np.asarray(checkpoint['sizing_advantage_buffer_states'], dtype=np.float32)[:adv_count, :state_dim]
    grounded = np.asarray(checkpoint['sizing_advantage_buffer_grounded'], dtype=np.float64)[:adv_count, :anchor_count]
    available = np.asarray(checkpoint['sizing_advantage_buffer_avail_masks'], dtype=bool)[:adv_count, :anchor_count]
    holdout = np.asarray(checkpoint.get('sizing_advantage_buffer_holdout', np.zeros(adv_count)), dtype=bool)[:adv_count]
    finite = np.isfinite(adv_states).all(axis=1) & available.any(axis=1)
    holdout_rows = finite & holdout
    if not holdout_rows.any():
        holdout_rows = finite
    holdout_predictions = _evaluate(net, adv_states[holdout_rows], anchors, min_bet, max_bet, args.batch_size)
    holdout_grounded = grounded[holdout_rows]
    holdout_available = available[holdout_rows]
    grounded_mask = holdout_available & np.isfinite(holdout_grounded)
    grounded_error = holdout_predictions[grounded_mask] - holdout_grounded[grounded_mask]

    source_counts = {str(key): int(value) for key, value in Counter(q_sources.tolist()).items()}
    kind_names = SizingQBuffer.SELECTED_ID_TO_KIND
    kind_counts = {kind_names.get(int(key), 'UNKNOWN'): int(value)
                   for key, value in Counter(q_kinds.tolist()).items()}
    report = {
        'checkpoint': str(checkpoint_path),
        'anchors': anchors.tolist(),
        'q_replay_samples': q_count,
        'per_anchor': per_anchor,
        'source_counts': source_counts,
        'selected_kind_counts': kind_counts,
        'q_replay_mse': _safe_mean(np.square(q_errors)),
        'q_replay_mae': _safe_mean(np.abs(q_errors)),
        'q_replay_unique_states': int(len(set(q_state_keys))),
        'q_replay_selected_samples_by_anchor': [
            int((q_selected_indices == index).sum()) for index in range(anchor_count)],
        'q_replay_effective_samples_by_anchor': [
            int((q_effective_indices == index).sum()) for index in range(anchor_count)],
        'q_replay_conditional_frequency_by_anchor': [
            float((q_anchor_indices == index).sum() / q_count) if q_count else None
            for index in range(anchor_count)],
        'legal_only_coverage': [int(holdout_available[:, index].sum()) for index in range(anchor_count)],
        'played_coverage': [int((grounded_mask[:, index]).sum()) for index in range(anchor_count)],
        'unplayed_coverage': [int((holdout_available[:, index] & ~grounded_mask[:, index]).sum()) for index in range(anchor_count)],
        'grounded_holdout': {
            'rows': int(holdout_rows.sum()),
            'anchor_values': int(grounded_mask.sum()),
            'mse': _safe_mean(np.square(grounded_error)),
            'mae': _safe_mean(np.abs(grounded_error)),
            'ranking': _ranking_metrics(holdout_predictions, holdout_grounded, holdout_available),
        },
    }
    output_path = Path(args.output).resolve() if args.output else checkpoint_path.with_name(
        f'{checkpoint_path.stem}_sizing_q_ranking.json')
    output_path.write_text(json.dumps(report, indent=2, allow_nan=False), encoding='utf-8')
    print(f'Отчёт записан: {output_path}')
    print(f"Q replay: {q_count}; holdout rows: {report['grounded_holdout']['rows']}")
    print(f"Ranking: {report['grounded_holdout']['ranking']}")


if __name__ == '__main__':
    main()
