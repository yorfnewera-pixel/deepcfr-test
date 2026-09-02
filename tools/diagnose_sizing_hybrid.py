#!/usr/bin/env python3
"""Офлайн shadow-анализ CFR, Q и гибридной sizing-политики.

Скрипт читает полный checkpoint и пишет JSON. Он не запускает traversal,
обучение или изменение checkpoint. Сравнение с grounded значениями проводится
только на legal-анкерах, для которых в replay действительно есть target.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.core.model import SizingAnchorNet, SizingQNetwork  # noqa: E402


def _load_q_network(state_dict: dict[str, torch.Tensor]) -> SizingQNetwork:
    embed_dim = int(state_dict['size_embed.weight'].shape[0])
    hidden_size = int(state_dict['base.0.weight'].shape[0])
    state_dim = int(state_dict['base.0.weight'].shape[1]) - embed_dim
    net = SizingQNetwork(state_dim, hidden_size, embed_dim)
    net.load_state_dict(state_dict, strict=True)
    net.eval()
    return net


def _load_advantage_network(state_dict: dict[str, torch.Tensor]) -> SizingAnchorNet:
    input_size = int(state_dict['base.0.weight'].shape[1])
    hidden_size = int(state_dict['base.0.weight'].shape[0])
    anchors = int(state_dict['anchor_head.weight'].shape[0])
    net = SizingAnchorNet(input_size, hidden_size, anchors)
    # Старые checkpoint могут содержать удалённую bucket_head.
    compatible = {key: value for key, value in state_dict.items() if not key.startswith('bucket_head.')}
    net.load_state_dict(compatible, strict=True)
    net.eval()
    return net


def _evaluate_q(net: SizingQNetwork, states: np.ndarray, anchors: np.ndarray,
                min_bet: float, max_bet: float, batch_size: int) -> np.ndarray:
    normalized = (anchors - min_bet) / (max_bet - min_bet)
    values = []
    with torch.inference_mode():
        for start in range(0, len(states), batch_size):
            batch = torch.from_numpy(states[start:start + batch_size]).float()
            count = len(batch)
            repeated_states = batch.unsqueeze(1).expand(-1, len(anchors), -1).reshape(-1, batch.shape[1])
            repeated_sizes = torch.from_numpy(normalized).unsqueeze(0).expand(count, -1).reshape(-1).float()
            values.append(net(repeated_states, repeated_sizes).reshape(count, len(anchors)).numpy())
    return np.concatenate(values, axis=0) if values else np.empty((0, len(anchors)), dtype=np.float32)


def _evaluate_advantage(net: SizingAnchorNet, states: np.ndarray, batch_size: int) -> np.ndarray:
    values = []
    with torch.inference_mode():
        for start in range(0, len(states), batch_size):
            values.append(net(torch.from_numpy(states[start:start + batch_size]).float()).numpy())
    return np.concatenate(values, axis=0) if values else np.empty((0, 0), dtype=np.float32)


def _regret_matching(values: np.ndarray, legal: np.ndarray) -> np.ndarray:
    """Regret matching с нулевой массой у illegal-анкерoв."""
    values = np.asarray(values, dtype=np.float64)
    legal = np.asarray(legal, dtype=bool)
    result = np.zeros_like(values, dtype=np.float64)
    count = int(legal.sum())
    if count == 0:
        return result
    positive = np.maximum(values[legal], 0.0)
    total = float(positive.sum())
    result[legal] = positive / total if total > 1e-12 else 1.0 / count
    return result


def _restrict_policy(policy: np.ndarray, observed: np.ndarray) -> np.ndarray:
    result = np.zeros_like(policy, dtype=np.float64)
    mass = float(np.asarray(policy, dtype=np.float64)[observed].sum())
    count = int(np.asarray(observed, dtype=bool).sum())
    if count == 0:
        return result
    result[observed] = policy[observed] / mass if mass > 1e-12 else 1.0 / count
    return result


def _values_to_regrets(values: np.ndarray, legal: np.ndarray, baseline: np.ndarray) -> tuple[float, np.ndarray]:
    """Преобразует value в regret относительно фиксированной CFR-политики."""
    baseline = _restrict_policy(baseline, legal)
    expected_value = float(np.dot(baseline, np.where(legal, values, 0.0)))
    return expected_value, np.where(legal, values - expected_value, 0.0)


def _policy_from_values(values: np.ndarray, legal: np.ndarray, baseline: np.ndarray) -> np.ndarray:
    _, regrets = _values_to_regrets(values, legal, baseline)
    return _regret_matching(regrets, legal)


def _build_shadow_policies(cfr_logits: np.ndarray, q_values: np.ndarray,
                           grounded: np.ndarray, legal: np.ndarray) -> dict[str, np.ndarray]:
    """Строит политики, не подменяя Q там, где grounded-target отсутствует."""
    legal = np.asarray(legal, dtype=bool)
    cfr = _regret_matching(cfr_logits, legal)
    q_ev, q_regrets = _values_to_regrets(q_values, legal, cfr)
    q = _regret_matching(q_regrets, legal)
    credited = legal & np.isfinite(grounded)
    hybrid_values = np.asarray(q_values, dtype=np.float64).copy()
    hybrid_values[credited] = grounded[credited]
    hybrid_ev, hybrid_regrets = _values_to_regrets(hybrid_values, legal, cfr)
    hybrid = _regret_matching(hybrid_regrets, legal)
    observed = credited
    if observed.any():
        grounded_ev, grounded_regrets = _values_to_regrets(grounded, observed, cfr)
        target = _regret_matching(grounded_regrets, observed)
    else:
        grounded_ev, grounded_regrets, target = 0.0, np.zeros_like(cfr), np.zeros_like(cfr)
    return {
        'cfr': cfr,
        'q': q,
        'hybrid': hybrid,
        'grounded': target,
        'credited': credited,
        'hybrid_values': hybrid_values,
        'q_ev': q_ev,
        'q_regrets': q_regrets,
        'hybrid_ev': hybrid_ev,
        'hybrid_regrets': hybrid_regrets,
        'grounded_ev': grounded_ev,
        'grounded_regrets': grounded_regrets,
    }


def _tv(left: np.ndarray, right: np.ndarray) -> float:
    return float(0.5 * np.abs(left - right).sum())


def _entropy(policy: np.ndarray) -> float:
    positive = policy[policy > 0.0]
    return float(-(positive * np.log(positive)).sum())


def _summary(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {'mean': None, 'median': None, 'p90': None}
    array = np.asarray(values, dtype=np.float64)
    return {'mean': float(array.mean()), 'median': float(np.median(array)), 'p90': float(np.quantile(array, 0.9))}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', '-c', required=True, help='Путь к полному checkpoint.')
    parser.add_argument('--output', '-o', help='Путь к JSON-отчёту; по умолчанию рядом с checkpoint.')
    parser.add_argument('--max-states', type=int, default=10000, help='Максимум строк replay для анализа.')
    parser.add_argument('--examples', type=int, default=30, help='Сколько диагностических примеров сохранить в JSON.')
    parser.add_argument('--batch-size', type=int, default=4096)
    args = parser.parse_args()

    if args.max_states <= 0 or args.examples < 0 or args.batch_size <= 0:
        raise ValueError('max-states и batch-size должны быть положительными; examples — неотрицательным.')
    checkpoint_path = Path(args.checkpoint).resolve()
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    required = ('sizing_q_net', 'advantage_sizing_net', 'sizing_advantage_buffer_states',
                'sizing_advantage_buffer_grounded', 'sizing_advantage_buffer_avail_masks',
                'sizing_advantage_buffer_cur_id')
    missing = [key for key in required if key not in checkpoint]
    if missing:
        raise KeyError(f'Checkpoint не содержит данные для shadow-анализа: {missing}')

    config = checkpoint.get('config', {})
    anchors = np.asarray(checkpoint.get('anchors', config.get('fixed_sizing_grid', config.get('anchor_sizes'))), dtype=np.float32)
    if anchors.ndim != 1 or not len(anchors):
        raise ValueError('Не найден корректный список anchors в checkpoint.')
    min_bet = float(checkpoint.get('min_bet_size', config.get('min_bet_size', 0.1)))
    max_bet = float(checkpoint.get('max_bet_size', config.get('max_bet_size', 3.0)))
    if max_bet <= min_bet:
        raise ValueError('max_bet_size должен быть больше min_bet_size.')

    count = min(int(checkpoint['sizing_advantage_buffer_cur_id']), len(checkpoint['sizing_advantage_buffer_states']))
    state_dim = int(checkpoint.get('sizing_advantage_buffer_state_dim', np.asarray(checkpoint['sizing_advantage_buffer_states']).shape[1]))
    states = np.asarray(checkpoint['sizing_advantage_buffer_states'], dtype=np.float32)[:count, :state_dim]
    grounded = np.asarray(checkpoint['sizing_advantage_buffer_grounded'], dtype=np.float64)[:count, :len(anchors)]
    legal = np.asarray(checkpoint['sizing_advantage_buffer_avail_masks'], dtype=bool)[:count, :len(anchors)]
    holdout = np.asarray(checkpoint.get('sizing_advantage_buffer_holdout', np.zeros(count)), dtype=bool)[:count]
    finite_rows = np.isfinite(states).all(axis=1) & legal.any(axis=1)
    candidates = np.flatnonzero(finite_rows & holdout)
    selection = 'holdout'
    if not len(candidates):
        candidates = np.flatnonzero(finite_rows)
        selection = 'all_valid_fallback'
    if len(candidates) > args.max_states:
        candidates = candidates[np.linspace(0, len(candidates) - 1, args.max_states, dtype=np.int64)]
    states, grounded, legal = states[candidates], grounded[candidates], legal[candidates]

    q_net = _load_q_network(checkpoint['sizing_q_net'])
    advantage_net = _load_advantage_network(checkpoint['advantage_sizing_net'])
    q_values = _evaluate_q(q_net, states, anchors, min_bet, max_bet, args.batch_size)
    cfr_logits = _evaluate_advantage(advantage_net, states, args.batch_size)
    if q_values.shape != cfr_logits.shape or q_values.shape != legal.shape:
        raise ValueError(f'Несовместимые формы: Q={q_values.shape}, CFR={cfr_logits.shape}, legal={legal.shape}.')

    tv_metrics = {name: [] for name in ('cfr', 'q', 'hybrid', 'uniform')}
    entropy_metrics = {name: [] for name in ('cfr', 'q', 'hybrid')}
    hybrid_cfr_tv: list[float] = []
    q_spreads: list[float] = []
    hybrid_spreads: list[float] = []
    top_changed = 0
    examples = []
    eligible = 0
    credited_counts, legal_counts = [], []
    for index, (cfr_logits_row, q_row, grounded_row, legal_row) in enumerate(zip(cfr_logits, q_values, grounded, legal, strict=True)):
        policies = _build_shadow_policies(cfr_logits_row, q_row, grounded_row, legal_row)
        credited = policies['credited']
        legal_counts.append(int(legal_row.sum()))
        credited_counts.append(int(credited.sum()))
        q_spreads.append(float(q_row[legal_row].max() - q_row[legal_row].min()))
        hybrid_spreads.append(float(policies['hybrid_values'][legal_row].max() - policies['hybrid_values'][legal_row].min()))
        for name in entropy_metrics:
            entropy_metrics[name].append(_entropy(policies[name]))
        hybrid_cfr_tv.append(_tv(policies['hybrid'], policies['cfr']))
        if int(np.argmax(policies['hybrid'])) != int(np.argmax(policies['cfr'])):
            top_changed += 1
        if int(credited.sum()) < 2:
            continue
        eligible += 1
        observed_cfr = _restrict_policy(policies['cfr'], credited)
        observed_q = _restrict_policy(policies['q'], credited)
        observed_hybrid = _restrict_policy(policies['hybrid'], credited)
        uniform = np.zeros_like(observed_cfr)
        uniform[credited] = 1.0 / int(credited.sum())
        for name, policy in (('cfr', observed_cfr), ('q', observed_q), ('hybrid', observed_hybrid), ('uniform', uniform)):
            tv_metrics[name].append(_tv(policy, policies['grounded']))
        if len(examples) < args.examples:
            examples.append({
                'replay_index': int(candidates[index]),
                'legal_indices': np.flatnonzero(legal_row).tolist(),
                'credited_indices': np.flatnonzero(credited).tolist(),
                'q_values': q_row.tolist(),
                'hybrid_values': policies['hybrid_values'].tolist(),
                'cfr_values': cfr_logits_row.tolist(),
                'grounded_values': [None if not np.isfinite(value) else float(value) for value in grounded_row],
                'cfr_policy': policies['cfr'].tolist(),
                'q_policy': policies['q'].tolist(),
                'hybrid_policy': policies['hybrid'].tolist(),
                'grounded_policy_on_credited': policies['grounded'].tolist(),
                'q_ev': policies['q_ev'],
                'q_regrets': policies['q_regrets'].tolist(),
                'hybrid_ev': policies['hybrid_ev'],
                'hybrid_regrets': policies['hybrid_regrets'].tolist(),
                'grounded_ev': policies['grounded_ev'],
                'grounded_regrets': policies['grounded_regrets'].tolist(),
            })

    report = {
        'checkpoint': str(checkpoint_path),
        'anchors': anchors.tolist(),
        'selection': selection,
        'analysed_rows': int(len(states)),
        'eligible_grounded_rows': eligible,
        'comparison_scope': 'Политики ограничены и перенормированы на legal grounded-анкерах; uncredited Q-решения не имеют истинного target и не считаются доказанной выгодой.',
        'mean_legal_anchors': float(np.mean(legal_counts)) if legal_counts else None,
        'mean_credited_anchors': float(np.mean(credited_counts)) if credited_counts else None,
        'q_value_spread': _summary(q_spreads),
        'hybrid_value_spread': _summary(hybrid_spreads),
        'tv_to_grounded_policy': {name: _summary(values) for name, values in tv_metrics.items()},
        'policy_entropy': {name: _summary(values) for name, values in entropy_metrics.items()},
        'hybrid_vs_cfr_tv': _summary(hybrid_cfr_tv),
        'hybrid_top_anchor_changed_fraction': top_changed / len(states) if len(states) else None,
        'examples': examples,
    }
    output_path = Path(args.output).resolve() if args.output else checkpoint_path.with_name(f'{checkpoint_path.stem}_sizing_hybrid_shadow.json')
    output_path.write_text(json.dumps(report, indent=2, allow_nan=False), encoding='utf-8')
    print(f'Отчёт записан: {output_path}')
    print(f"Строк проанализировано: {report['analysed_rows']}; пригодно для grounded-сравнения: {eligible}")
    print(f"TV hybrid -> grounded: {report['tv_to_grounded_policy']['hybrid']}")


if __name__ == '__main__':
    main()
