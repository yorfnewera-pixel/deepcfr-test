#!/usr/bin/env python
# relapse_diff.py — сравнение чекпоинтов на фиксированном наборе состояний.
#
# Выявляет что ломается ПЕРВЫМ при relapse коллапсе: H1 (буфер),
# H2 (Q-деградация), H3 (regret-накопление).
#
# Использование:
#   python relapse_diff.py models/test84seed1 --iters 100,200,300 --num-states 1000 --output sizing_reports/84b-relapse-mechanism.md

import argparse
import json
import os
import sys
from collections import defaultdict
from typing import Dict, List, Optional

import numpy as np
import torch

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_SCRIPT_DIR)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

import pokers as pkrs
from checkpoint_tools import (
    ANCHORS_DEFAULT,
    N_PLAYERS,
    encode_batch,
    filter_raise_states,
    load_full_checkpoint,
    run_action_q_buffer_diagnostics,
    run_action_q_model_diagnostics,
    run_collapse_diag,
    run_q_comparison,
    run_sizing_q_buffer_stats,
    run_sizing_q_target_diag_stats,
    sample_states,
    _stats,
)

ACTION_NAMES = ['fold', 'check', 'call', 'raise']


def get_action_masks(states: list, num_actions: int = 4) -> np.ndarray:
    """Строит legal action mask из списка pokers.State."""
    masks = np.zeros((len(states), num_actions), dtype=np.float32)
    for i, s in enumerate(states):
        for a in s.legal_actions:
            idx = int(a)
            if 0 <= idx < num_actions:
                masks[i, idx] = 1.0
    return masks


def compute_action_distribution(nets: dict, X_states: torch.Tensor,
                                action_masks: np.ndarray) -> Optional[Dict[str, float]]:
    if 'advantage_net' not in nets:
        return None

    adv_net = nets['advantage_net']
    adv_net.eval()

    with torch.inference_mode():
        adv = adv_net(X_states)[:, :4].cpu().numpy()

    pos = np.maximum(adv, 0) * action_masks
    pos_sum = pos.sum(axis=1, keepdims=True)
    strategy = np.where(
        pos_sum > 0,
        pos / pos_sum.clip(1e-8),
        action_masks / action_masks.sum(axis=1, keepdims=True).clip(1.0),
    )

    dist = {}
    for i, name in enumerate(ACTION_NAMES):
        dist[name] = float(strategy[:, i].mean())
    return dist


def compute_advantage_stats(nets: dict, X_states: torch.Tensor,
                            action_masks: np.ndarray) -> Optional[Dict[str, dict]]:
    if 'advantage_net' not in nets:
        return None

    adv_net = nets['advantage_net']
    adv_net.eval()

    with torch.inference_mode():
        adv = adv_net(X_states)[:, :4].cpu().numpy()

    result = {}
    for i, name in enumerate(ACTION_NAMES):
        result[name] = _stats(adv[:, i])
    return result


def format_pct_change(v1: float, v2: float) -> str:
    """Форматирует изменение в процентах."""
    delta = v2 - v1
    if abs(v1) < 1e-8:
        return '—'
    pct = delta / abs(v1) * 100.0
    sign = '+' if delta > 0 else ''
    return f'{sign}{pct:.1f}%'


def _clean_dict_for_json(obj):
    if isinstance(obj, dict):
        return {k: _clean_dict_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_clean_dict_for_json(item) for item in obj]
    elif isinstance(obj, (np.integer,)):
        return int(obj)
    elif isinstance(obj, (np.floating,)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


def markdown_section(lines: List[str], title: str, level: int = 2):
    lines.append(f'{"#" * level} {title}')
    lines.append('')


def markdown_table(lines: List[str], headers: List[str], rows: List[List[str]]):
    header_line = '| ' + ' | '.join(headers) + ' |'
    sep_line = '|' + '|'.join([' --- ' for _ in headers]) + '|'
    lines.append(header_line)
    lines.append(sep_line)
    for row in rows:
        lines.append('| ' + ' | '.join(str(c) for c in row) + ' |')
    lines.append('')


def run_relapse_diff(checkpoint_dir: str, iters: List[int],
                     num_states: int = 1000, seed: int = 42,
                     output_md: Optional[str] = None,
                     output_json: Optional[str] = None) -> dict:
    """Главная функция: загружает чекпоинты, генерирует состояния, вычисляет диффы."""

    print(f'[relapse_diff] Генерация {num_states} фиксированных состояний (seed={seed})...')
    all_states = sample_states(num_states, num_players=N_PLAYERS, seed=seed)
    raise_states = filter_raise_states(all_states)
    print(f'  Всего состояний: {len(all_states)}, raise-legal: {len(raise_states)}')

    if len(raise_states) < 100:
        print(f'  WARNING: мало raise-legal состояний ({len(raise_states)}), диагностика ненадёжна')

    action_masks_all = get_action_masks(all_states)
    action_masks_raise = get_action_masks(raise_states)

    _use_multi = False
    for it in iters:
        _peek_path = os.path.join(checkpoint_dir, f'multi_checkpoint_iter_{it}.pt')
        if os.path.exists(_peek_path):
            _peek = torch.load(_peek_path, map_location='cpu', weights_only=False)
            _use_multi = _peek.get('use_multi_agent_advantage', False)
            break

    X_all = encode_batch(all_states, use_multi_agent=_use_multi)
    X_raise = encode_batch(raise_states, use_multi_agent=_use_multi)

    print(f'[relapse_diff] Загрузка чекпоинтов из {checkpoint_dir}...')
    results = {}
    for it in iters:
        ckpt_path = os.path.join(checkpoint_dir, f'multi_checkpoint_iter_{it}.pt')
        if not os.path.exists(ckpt_path):
            print(f'  SKIP iter {it}: файл не найден ({ckpt_path})')
            continue

        print(f'  Загрузка iter {it}...')
        nets = load_full_checkpoint(ckpt_path, device='cpu')
        iter_num = nets.get('iteration', it)

        print(f'    → collapse_diag...')
        collapse = run_collapse_diag(nets, X_raise)

        print(f'    → q_comparison...')
        q_comp = run_q_comparison(nets, X_raise)

        print(f'    → action_distribution + advantage_stats...')
        action_dist = compute_action_distribution(nets, X_all, action_masks_all)
        adv_stats = compute_advantage_stats(nets, X_all, action_masks_all)

        print(f'    → buffer_diag...')
        buf_diag = run_action_q_buffer_diagnostics(nets)
        model_diag = run_action_q_model_diagnostics(nets, device='cpu')

        print(f'    → sizing_q_buffer + target_diag...')
        sq_buf = run_sizing_q_buffer_stats(nets)
        sq_target = run_sizing_q_target_diag_stats(nets)

        results[iter_num] = {
            'iter': iter_num,
            'path': ckpt_path,
            'collapse': collapse,
            'q_compare': q_comp,
            'action_distribution': action_dist,
            'advantage_stats': adv_stats,
            'buffer_diag': buf_diag,
            'model_diag': model_diag,
            'sizing_q_buffer': sq_buf,
            'sizing_q_target': sq_target,
        }

    sorted_iters = sorted(results.keys())
    print(f'[relapse_diff] Загружено {len(sorted_iters)} чекпоинтов: {sorted_iters}')

    diff_report = _build_diff_report(results, sorted_iters, all_states, raise_states)

    if output_json:
        json_path = output_json
        if not json_path.startswith(_PROJECT_ROOT):
            json_path = os.path.join(_PROJECT_ROOT, json_path)
        os.makedirs(os.path.dirname(json_path), exist_ok=True)
        clean = _clean_dict_for_json(diff_report)
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(clean, f, indent=2, ensure_ascii=False, default=str)
        print(f'[relapse_diff] JSON-отчёт: {json_path}')

    if output_md:
        md_path = output_md
        if not md_path.startswith(_PROJECT_ROOT):
            md_path = os.path.join(_PROJECT_ROOT, md_path)
        os.makedirs(os.path.dirname(md_path), exist_ok=True)
        _write_markdown_report(diff_report, md_path)
        print(f'[relapse_diff] Markdown-отчёт: {md_path}')

    return diff_report


def _build_diff_report(results: dict, sorted_iters: List[int],
                       all_states: list, raise_states: list) -> dict:
    """Строит структурированный дифф-отчёт."""

    report = {
        'methodology': {
            'num_all_states': len(all_states),
            'num_raise_states': len(raise_states),
            'checkpoints': sorted_iters,
            'description': 'Сравнение на фиксированном наборе состояний (seed=42). '
                           'Все метрики (collapse, Q, advantage) вычислены на одних и тех же состояниях. '
                           'Buffer/model диагностика — на снапшотах буферов из каждого чекпоинта.',
        },
        'pairs': {},
        'timeline': {},
        'hypotheses': {},
    }

    for i in range(len(sorted_iters) - 1):
        it_a = sorted_iters[i]
        it_b = sorted_iters[i + 1]
        pair_key = f'{it_a}_vs_{it_b}'
        report['pairs'][pair_key] = _compute_pair_diff(results[it_a], results[it_b])

    if len(sorted_iters) == 1:
        r = results[sorted_iters[0]]
        anchor_probs = {}
        for s in r.get('collapse', {}).get('anchor_stats', []):
            anchor_probs[f'{s["anchor"]:.2f}'] = s['avg_raw_prob']
        report['_single_data'] = {
            'action_distribution': r.get('action_distribution'),
            'anchor_raw_probs': anchor_probs,
            'advantage_stats': r.get('advantage_stats'),
            'q_compare': r.get('q_compare'),
            'buffer_diag': r.get('buffer_diag'),
            'model_diag': r.get('model_diag'),
        }

    _build_timeline(report, results, sorted_iters)
    _evaluate_hypotheses(report, results, sorted_iters)

    return report


def _compute_pair_diff(r1: dict, r2: dict) -> dict:
    """Вычисляет разницу между двумя чекпоинтами."""
    pair = {'iter_a': r1['iter'], 'iter_b': r2['iter']}

    # --- Policy: anchor distribution ---
    c1 = r1.get('collapse', {})
    c2 = r2.get('collapse', {})
    anchor_diff = []
    if c1 and c2:
        a1_stats = {s['anchor']: s for s in c1.get('anchor_stats', [])}
        a2_stats = {s['anchor']: s for s in c2.get('anchor_stats', [])}
        for anchor in sorted(set(a1_stats.keys()) | set(a2_stats.keys())):
            s1 = a1_stats.get(anchor, {})
            s2 = a2_stats.get(anchor, {})
            anchor_diff.append({
                'anchor': anchor,
                'raw_prob_a': round(s1.get('avg_raw_prob', 0), 4),
                'raw_prob_b': round(s2.get('avg_raw_prob', 0), 4),
                'selected_pct_a': round(s1.get('selected_pct', 0), 1),
                'selected_pct_b': round(s2.get('selected_pct', 0), 1),
            })
    pair['anchor_distribution'] = anchor_diff

    # --- Policy: action distribution ---
    ad1 = r1.get('action_distribution')
    ad2 = r2.get('action_distribution')
    action_diff = {}
    if ad1 and ad2:
        for name in ACTION_NAMES:
            v1 = ad1.get(name, 0)
            v2 = ad2.get(name, 0)
            action_diff[name] = {
                'a': round(v1, 4),
                'b': round(v2, 4),
                'delta': round(v2 - v1, 4),
                'pct_change': format_pct_change(v1, v2),
            }
    pair['action_distribution'] = action_diff

    # --- Advantage stats ---
    as1 = r1.get('advantage_stats', {})
    as2 = r2.get('advantage_stats', {})
    adv_diff = {}
    if as1 and as2:
        for name in ACTION_NAMES:
            if name in as1 and name in as2:
                adv_diff[name] = {
                    'mean_a': round(as1[name]['mean'], 4),
                    'mean_b': round(as2[name]['mean'], 4),
                    'delta': round(as2[name]['mean'] - as1[name]['mean'], 4),
                }
    pair['advantage_diff'] = adv_diff

    # --- Q comparison ---
    q1 = r1.get('q_compare', {})
    q2 = r2.get('q_compare', {})
    q_diff = {}
    if q1 and q2:
        if 'raise_vs' in q1 and 'raise_vs' in q2:
            for key in ['raise_lt_call_pct', 'raise_lt_fold_pct']:
                a = q1['raise_vs'].get(key, 0)
                b = q2['raise_vs'].get(key, 0)
                q_diff[key] = {'a': round(a, 1), 'b': round(b, 1),
                               'delta': round(b - a, 1)}
        if 'q_action' in q1 and 'q_action' in q2:
            for name in ACTION_NAMES:
                if name in q1['q_action'] and name in q2['q_action']:
                    q_diff[f'q_{name}_mean'] = {
                        'a': round(q1['q_action'][name]['mean'], 3),
                        'b': round(q2['q_action'][name]['mean'], 3),
                        'delta': round(q2['q_action'][name]['mean'] - q1['q_action'][name]['mean'], 3),
                    }
        if 'sizing_q_stats' in q1 and 'sizing_q_stats' in q2:
            q_diff['sq_max_mean'] = {
                'a': round(q1['sizing_q_stats']['max']['mean'], 3),
                'b': round(q2['sizing_q_stats']['max']['mean'], 3),
                'delta': round(q2['sizing_q_stats']['max']['mean'] - q1['sizing_q_stats']['max']['mean'], 3),
            }
        if 'raise_vs_sizing' in q1 and 'raise_vs_sizing' in q2:
            q_diff['raise_gt_max_sq_pct'] = {
                'a': round(q1['raise_vs_sizing'].get('q_raise_gt_max_sq_pct', 0), 1),
                'b': round(q2['raise_vs_sizing'].get('q_raise_gt_max_sq_pct', 0), 1),
                'delta': round(q2['raise_vs_sizing'].get('q_raise_gt_max_sq_pct', 0) - q1['raise_vs_sizing'].get('q_raise_gt_max_sq_pct', 0), 1),
            }
    pair['q_diff'] = q_diff

    # --- Buffer diagnostics ---
    b1 = r1.get('buffer_diag')
    b2 = r2.get('buffer_diag')
    buf_diff = {}
    if b1 and b1.get('available') and b2 and b2.get('available'):
        buf_diff['total_a'] = b1.get('total_samples', 0)
        buf_diff['total_b'] = b2.get('total_samples', 0)
        for name in ACTION_NAMES:
            c1 = b1.get('count_by_action', {}).get(name, 0)
            c2 = b2.get('count_by_action', {}).get(name, 0)
            t1 = b1.get('terminal_ratio_by_action', {}).get(name)
            t2 = b2.get('terminal_ratio_by_action', {}).get(name)
            r1_stats = b1.get('reward_stats_by_action', {}).get(name)
            r2_stats = b2.get('reward_stats_by_action', {}).get(name)
            buf_diff[name] = {
                'count_a': c1, 'count_b': c2,
                'terminal_pct_a': t1['terminal_pct'] if t1 else None,
                'terminal_pct_b': t2['terminal_pct'] if t2 else None,
                'reward_mean_a': r1_stats['mean'] if r1_stats else None,
                'reward_mean_b': r2_stats['mean'] if r2_stats else None,
            }
    pair['buffer_diff'] = buf_diff

    # --- Model diagnostics ---
    m1 = r1.get('model_diag')
    m2 = r2.get('model_diag')
    model_diff = {}
    if m1 and m1.get('available') and m2 and m2.get('available'):
        for name in ACTION_NAMES:
            bv1 = m1.get('bootstrap_value_stats_by_action', {}).get(name)
            bv2 = m2.get('bootstrap_value_stats_by_action', {}).get(name)
            ts1 = m1.get('target_stats_by_action', {}).get(name)
            ts2 = m2.get('target_stats_by_action', {}).get(name)
            ns1 = m1.get('next_strategy_raise_mass_by_action', {}).get(name)
            ns2 = m2.get('next_strategy_raise_mass_by_action', {}).get(name)
            model_diff[name] = {}
            if bv1 and bv2:
                model_diff[name]['bootstrap_v_mean'] = {
                    'a': round(bv1['mean'], 4),
                    'b': round(bv2['mean'], 4),
                    'delta': round(bv2['mean'] - bv1['mean'], 4),
                }
            if ts1 and ts2:
                model_diff[name]['target_mean'] = {
                    'a': round(ts1['mean'], 4),
                    'b': round(ts2['mean'], 4),
                    'delta': round(ts2['mean'] - ts1['mean'], 4),
                }
            if ns1 and ns2:
                model_diff[name]['next_raise_mass'] = {
                    'a': round(ns1['mean'], 4),
                    'b': round(ns2['mean'], 4),
                    'delta': round(ns2['mean'] - ns1['mean'], 4),
                }
        if 'bootstrap_by_next_role' in m1 and 'bootstrap_by_next_role' in m2:
            model_diff['by_role'] = {}
            for role in ['hero', 'opponent', 'terminal']:
                r1_data = m1['bootstrap_by_next_role'].get(role, {})
                r2_data = m2['bootstrap_by_next_role'].get(role, {})
                if r1_data and r2_data:
                    model_diff['by_role'][role] = {
                        'count_a': r1_data.get('count', 0),
                        'count_b': r2_data.get('count', 0),
                    }
    pair['model_diff'] = model_diff

    # --- Sizing Q target diag ---
    st1 = r1.get('sizing_q_target')
    st2 = r2.get('sizing_q_target')
    target_diff = {}
    if st1 and st2:
        src_cross1 = st1.get('source2_cross_anchor', {})
        src_cross2 = st2.get('source2_cross_anchor', {})
        if src_cross1 and src_cross2:
            target_diff['source2_mean_range'] = {
                'a': round(src_cross1.get('mean_range', 0), 4),
                'b': round(src_cross2.get('mean_range', 0), 4),
                'delta': round(src_cross2.get('mean_range', 0) - src_cross1.get('mean_range', 0), 4),
            }
            target_diff['source2_n_anchors'] = {
                'a': src_cross1.get('n_anchors_represented', 0),
                'b': src_cross2.get('n_anchors_represented', 0),
            }
        for src_id in [0, 1, 2]:
            cs1 = st1.get('clip_by_source', {}).get(src_id, {})
            cs2 = st2.get('clip_by_source', {}).get(src_id, {})
            if cs1 and cs2:
                target_diff[f'source{src_id}'] = {
                    'count_a': cs1.get('count', 0),
                    'count_b': cs2.get('count', 0),
                    'clip_high_a': cs1.get('clip_high', 0),
                    'clip_high_b': cs2.get('clip_high', 0),
                    'clip_low_a': cs1.get('clip_low', 0),
                    'clip_low_b': cs2.get('clip_low', 0),
                }
    pair['target_diag_diff'] = target_diff

    return pair


def _build_timeline(report: dict, results: dict, sorted_iters: List[int]):
    """Строит временную шкалу: что меняется на каждом шаге."""
    tl = report['timeline']

    for it in sorted_iters:
        r = results[it]
        ad = r.get('action_distribution', {})
        c = r.get('collapse', {})
        q = r.get('q_compare', {})

        anchor_010 = None
        top2_mass = 0
        unique = 0
        if c:
            for s in c.get('anchor_stats', []):
                if abs(s['anchor'] - 0.10) < 1e-6:
                    anchor_010 = round(s['selected_pct'], 1)
            raw_probs = sorted([s['avg_raw_prob'] for s in c.get('anchor_stats', [])], reverse=True)
            top2_mass = round(sum(raw_probs[:2]) * 100, 1) if len(raw_probs) >= 2 else 0
            unique = c.get('unique_selected', 0)

        raise_freq = round(ad.get('raise', 0) * 100, 1) if ad else None
        fold_freq = round(ad.get('fold', 0) * 100, 1) if ad else None

        q_fold_mean = None
        q_raise_mean = None
        if q and 'q_action' in q:
            q_fold_mean = round(q['q_action'].get('fold', {}).get('mean', 0), 3)
            q_raise_mean = round(q['q_action'].get('raise', {}).get('mean', 0), 3)

        raise_lt_fold = None
        if q and 'raise_vs' in q:
            raise_lt_fold = round(q['raise_vs'].get('raise_lt_fold_pct', 0), 1)

        tl[str(it)] = {
            'raise_freq': raise_freq,
            'fold_freq': fold_freq,
            'anchor_010_mass': anchor_010,
            'top2_mass': top2_mass,
            'unique_anchors': unique,
            'q_fold_mean': q_fold_mean,
            'q_raise_mean': q_raise_mean,
            'raise_lt_fold_pct': raise_lt_fold,
        }

    if len(sorted_iters) >= 2:
        first = tl[str(sorted_iters[0])]
        last = tl[str(sorted_iters[-1])]
        tl['_delta'] = {
            'raise_freq': f'{first["raise_freq"]}% → {last["raise_freq"]}%',
            'anchor_010': f'{first["anchor_010_mass"]}% → {last["anchor_010_mass"]}%',
            'q_raise_mean': f'{first["q_raise_mean"]} → {last["q_raise_mean"]}',
            'q_fold_mean': f'{first["q_fold_mean"]} → {last["q_fold_mean"]}',
        }


def _evaluate_hypotheses(report: dict, results: dict, sorted_iters: List[int]):
    """Оценивает гипотезы H1-H3 по данным диффа."""
    hyp = report['hypotheses']
    pairs = report.get('pairs', {})

    # H1: буфер наполняется fold-terminal сэмплами
    h1_evidence = []
    for pk, pd in pairs.items():
        bd = pd.get('buffer_diff', {})
        if bd and 'fold' in bd:
            ft_a = bd['fold'].get('terminal_pct_a')
            ft_b = bd['fold'].get('terminal_pct_b')
            if ft_a is not None and ft_b is not None:
                if abs(ft_b - ft_a) > 5:
                    direction = 'растёт' if ft_b > ft_a else 'падает'
                    h1_evidence.append(f'{pk}: fold terminal% {ft_a:.0f} → {ft_b:.0f} ({direction})')
    hyp['H1_buffer_fold_terminal'] = {
        'description': 'Буфер наполняется fold-terminal сэмплами → цели смещаются в пользу fold',
        'evidence': h1_evidence if h1_evidence else ['недостаточно данных'],
        'supported': len(h1_evidence) > 0 and any('растёт' in e for e in h1_evidence),
    }

    # H2: q_net деградирует → портит advantage через bootstrap
    h2_evidence = []
    tl = report.get('timeline', {})
    for it in sorted_iters:
        t = tl.get(str(it), {})
        if t.get('q_raise_mean') is not None:
            h2_evidence.append(f'iter {it}: q_raise_mean={t["q_raise_mean"]}, '
                               f'q_fold_mean={t["q_fold_mean"]}, '
                               f'raise_lt_fold={t.get("raise_lt_fold_pct")}%')
    hyp['H2_q_degradation'] = {
        'description': 'q_net деградирует и через bootstrap-канал портит advantage-цели',
        'evidence': h2_evidence,
    }

    # H3: отрицательные regret за raise накапливаются
    h3_evidence = []
    for pk, pd in pairs.items():
        ad = pd.get('advantage_diff', {})
        if 'raise' in ad and 'fold' in ad:
            rd = ad['raise']
            fd = ad['fold']
            h3_evidence.append(
                f'{pk}: advantage raise {rd["mean_a"]:.4f} → {rd["mean_b"]:.4f} '
                f'(delta={rd["delta"]:+.4f}), fold {fd["mean_a"]:.4f} → {fd["mean_b"]:.4f} '
                f'(delta={fd["delta"]:+.4f})')
    hyp['H3_regret_accumulation'] = {
        'description': 'Накопление отрицательных regret за raise перевешивает и возвращает политику',
        'evidence': h3_evidence if h3_evidence else ['недостаточно данных'],
    }

    # Определяем что ломается ПЕРВЫМ
    first_vs_second_key = f'{sorted_iters[0]}_vs_{sorted_iters[1]}' if len(sorted_iters) >= 2 else None
    if first_vs_second_key and first_vs_second_key in pairs:
        fp = pairs[first_vs_second_key]
        signals = []

        ad = fp.get('action_distribution', {})
        if ad and 'raise' in ad:
            d = abs(ad['raise']['delta'])
            signals.append(('action: raise_freq shift', d))

        qd = fp.get('q_diff', {})
        if 'q_raise_mean' in qd:
            d = abs(qd['q_raise_mean']['delta'])
            signals.append(('Q: raise_mean shift', d))

        avd = fp.get('advantage_diff', {})
        if 'raise' in avd:
            d = abs(avd['raise']['delta'])
            signals.append(('advantage: raise mean shift', d))

        signals.sort(key=lambda x: x[1], reverse=True)
        hyp['_first_to_break'] = {
            'pair': first_vs_second_key,
            'signals_ranked': [{'metric': s[0], 'abs_delta': round(s[1], 4)} for s in signals],
        }


def _write_markdown_report(report: dict, path: str):
    """Генерирует markdown-отчёт."""
    lines = []
    lines.append('# Relapse Diff — Анализ чекпоинтов')
    lines.append('')
    lines.append(f'> **Чекпоинты:** {report["methodology"]["checkpoints"]}')
    lines.append(f'> **Состояний:** {report["methodology"]["num_all_states"]} '
                 f'(raise-legal: {report["methodology"]["num_raise_states"]})')
    lines.append(f'> **Метод:** фиксированный набор состояний (seed=42), все метрики вычислены на одних и тех же состояниях')
    lines.append('> **Buffer/model диагностика:** на снапшотах буферов из каждого чекпоинта')
    lines.append('')

    single_mode = len(report["methodology"]["checkpoints"]) == 1

    # --- Timeline ---
    markdown_section(lines, 'Временная шкала', 2)
    tl = report.get('timeline', {})
    if tl:
        sorted_iters = sorted(k for k in tl.keys() if not k.startswith('_'))
        headers = ['Метрика'] + [f'iter {it}' for it in sorted_iters]
        metrics = ['raise_freq', 'fold_freq', 'anchor_010_mass', 'top2_mass',
                   'unique_anchors', 'q_fold_mean', 'q_raise_mean', 'raise_lt_fold_pct']
        metric_labels = {
            'raise_freq': 'raise %',
            'fold_freq': 'fold %',
            'anchor_010_mass': 'анкер 0.10 масса%',
            'top2_mass': 'top-2 масса%',
            'unique_anchors': 'уник. анкеров',
            'q_fold_mean': 'q_fold mean',
            'q_raise_mean': 'q_raise mean',
            'raise_lt_fold_pct': 'raise<fold %',
        }
        rows = []
        for m in metrics:
            row = [metric_labels.get(m, m)]
            for it in sorted_iters:
                row.append(str(tl.get(str(it), {}).get(m, '—')))
            rows.append(row)
        markdown_table(lines, headers, rows)

        delta = tl.get('_delta', {})
        if delta:
            lines.append('')
            lines.append('**Суммарное изменение (первый → последний):**')
            for k, v in delta.items():
                lines.append(f'- {metric_labels.get(k, k)}: {v}')
        lines.append('')

    # --- Single checkpoint data (если 1 чекпоинт) ---
    if single_mode:
        it = report["methodology"]["checkpoints"][0]
        r = report.get('_single_data', {})
        markdown_section(lines, f'Данные чекпоинта iter {it}', 2)

        ad = r.get('action_distribution', {})
        if ad:
            markdown_section(lines, 'Policy: распределение действий', 3)
            headers = ['Действие', '%']
            rows = []
            for name in ACTION_NAMES:
                if name in ad:
                    rows.append([name, f'{ad[name]*100:.1f}%'])
            markdown_table(lines, headers, rows)

        anc = r.get('anchor_raw_probs', {})
        if anc:
            markdown_section(lines, 'Policy: распределение анкеров (raw probs)', 3)
            headers = ['Анкер', 'raw_prob']
            rows = []
            for a_k, a_v in sorted(anc.items(), key=lambda x: x[1], reverse=True):
                rows.append([a_k, f'{a_v:.4f}'])
            markdown_table(lines, headers, rows)

        avs = r.get('advantage_stats', {})
        if avs:
            markdown_section(lines, 'Advantage: средний advantage по действиям', 3)
            headers = ['Действие', 'mean', 'std', 'min', 'max']
            rows = []
            for name in ACTION_NAMES:
                if name in avs:
                    s = avs[name]
                    rows.append([name, f'{s["mean"]:.4f}', f'{s["std"]:.4f}',
                                 f'{s["min"]:.4f}', f'{s["max"]:.4f}'])
            markdown_table(lines, headers, rows)

        q = r.get('q_compare', {})
        if q and 'q_action' in q:
            markdown_section(lines, 'Action-Q (на raise-legal состояниях)', 3)
            headers = ['Метрика', 'Значение']
            rows = []
            for name in ACTION_NAMES:
                if name in q.get('q_action', {}):
                    rows.append([f'q_{name} mean', f'{q["q_action"][name]["mean"]:.4f}'])
            if 'raise_vs' in q:
                for k, v in q['raise_vs'].items():
                    rows.append([k, str(v)])
            markdown_table(lines, headers, rows)

        bd = r.get('buffer_diag', {})
        if bd and bd.get('available'):
            markdown_section(lines, 'Buffer: состав', 3)
            headers = ['Действие', 'count', 'term%', 'reward mean']
            rows = []
            for name in ACTION_NAMES:
                cnt = bd.get('count_by_action', {}).get(name, 0)
                tr = bd.get('terminal_ratio_by_action', {}).get(name)
                rs = bd.get('reward_stats_by_action', {}).get(name)
                rows.append([
                    name,
                    str(cnt),
                    f'{tr["terminal_pct"]}%' if tr else '—',
                    f'{rs["mean"]:.4f}' if rs else '—',
                ])
            markdown_table(lines, headers, rows)

        md = r.get('model_diag', {})
        if md and md.get('available'):
            markdown_section(lines, 'Model: bootstrap V / targets / next_raise_mass', 3)
            headers = ['Действие', 'Метрика', 'Значение']
            rows = []
            for name in ACTION_NAMES:
                for mkey, mlabel in [('bootstrap_value_stats_by_action', 'bootstrap V'),
                                     ('target_stats_by_action', 'target'),
                                     ('next_strategy_raise_mass_by_action', 'next raise mass')]:
                    stats = md.get(mkey, {}).get(name)
                    if stats:
                        rows.append([name, mlabel, f'{stats["mean"]:.4f}'])
            markdown_table(lines, headers, rows)

    # --- Pair diffs ---
    for pk in sorted(report.get('pairs', {}).keys()):
        pd = report['pairs'][pk]
        it_a, it_b = pd['iter_a'], pd['iter_b']

        markdown_section(lines, f'Срез iter {it_a} → iter {it_b}', 2)

        # Policy: action distribution
        ad = pd.get('action_distribution', {})
        if ad:
            markdown_section(lines, 'Policy: распределение действий (advantage → regret-matching)', 3)
            headers = ['Действие', f'iter {it_a}', f'iter {it_b}', 'Δ', '% изм.']
            rows = []
            for name in ACTION_NAMES:
                if name in ad:
                    d = ad[name]
                    rows.append([
                        name,
                        f'{d["a"]*100:.1f}%',
                        f'{d["b"]*100:.1f}%',
                        f'{d["delta"]*100:+.1f}%',
                        d['pct_change'],
                    ])
            markdown_table(lines, headers, rows)

        # Policy: anchor distribution
        anc = pd.get('anchor_distribution', [])
        if anc:
            markdown_section(lines, 'Policy: распределение анкеров (strategy_sizing_net, raw probs)', 3)
            headers = ['Анкер', f'raw_prob iter {it_a}', f'raw_prob iter {it_b}',
                       f'selected% iter {it_a}', f'selected% iter {it_b}']
            rows = []
            for a in anc:
                rows.append([
                    f'{a["anchor"]:.2f}',
                    f'{a["raw_prob_a"]:.4f}',
                    f'{a["raw_prob_b"]:.4f}',
                    f'{a["selected_pct_a"]:.1f}%',
                    f'{a["selected_pct_b"]:.1f}%',
                ])
            markdown_table(lines, headers, rows)

        # Advantage stats
        avd = pd.get('advantage_diff', {})
        if avd:
            markdown_section(lines, 'Advantage: средний advantage по действиям', 3)
            headers = ['Действие', f'mean iter {it_a}', f'mean iter {it_b}', 'Δ']
            rows = []
            for name in ACTION_NAMES:
                if name in avd:
                    rows.append([
                        name,
                        f'{avd[name]["mean_a"]:.4f}',
                        f'{avd[name]["mean_b"]:.4f}',
                        f'{avd[name]["delta"]:+.4f}',
                    ])
            markdown_table(lines, headers, rows)

        # Q comparison
        qd = pd.get('q_diff', {})
        if qd:
            markdown_section(lines, 'Action-Q сравнение (на raise-legal состояниях)', 3)
            headers = ['Метрика', f'iter {it_a}', f'iter {it_b}', 'Δ']
            rows = []
            for key, val in qd.items():
                rows.append([key, str(val['a']), str(val['b']),
                             f'{val["delta"]:+.3f}' if isinstance(val['delta'], float) else str(val['delta'])])
            markdown_table(lines, headers, rows)

        # Buffer
        bd = pd.get('buffer_diff', {})
        if bd:
            markdown_section(lines, 'Buffer: состав (terminal%, reward)', 3)
            markdown_section(lines, f'Всего сэмплов: {bd.get("total_a", 0)} → {bd.get("total_b", 0)}', 4)
            headers = ['Действие', 'count', 'term%', 'reward mean', 'count', 'term%', 'reward mean']
            sub_headers = ['', f'iter {it_a}', '', '', f'iter {it_b}', '', '']
            lines.append('| Действие | count A | term% A | rew A | count B | term% B | rew B |')
            lines.append('| --- | --- | --- | --- | --- | --- | --- |')
            for name in ACTION_NAMES:
                if name in bd:
                    d = bd[name]
                    lines.append(
                        f'| {name} | {d["count_a"]} | {d["terminal_pct_a"]}% | {d["reward_mean_a"]} | '
                        f'{d["count_b"]} | {d["terminal_pct_b"]}% | {d["reward_mean_b"]} |')
            lines.append('')

        # Model diagnostics
        md = pd.get('model_diff', {})
        if md:
            markdown_section(lines, 'Model: bootstrap V / targets / next_raise_mass', 3)
            headers = ['Действие', 'Метрика', f'iter {it_a}', f'iter {it_b}', 'Δ']
            rows = []
            for name in ACTION_NAMES:
                if name in md:
                    for mkey, mlabel in [('bootstrap_v_mean', 'bootstrap V'),
                                         ('target_mean', 'target'),
                                         ('next_raise_mass', 'next raise mass')]:
                        if mkey in md[name]:
                            v = md[name][mkey]
                            rows.append([name, mlabel, str(v['a']), str(v['b']),
                                         f'{v["delta"]:+.4f}'])
            by_role = md.pop('by_role', {})
            markdown_table(lines, headers, rows)

            if by_role:
                lines.append('')
                lines.append('**Bootstrap по next_role:**')
                lines.append('')
                headers_r = ['Роль', f'count iter {it_a}', f'count iter {it_b}']
                rows_r = []
                for role, data in by_role.items():
                    rows_r.append([role, str(data.get('count_a', '')), str(data.get('count_b', ''))])
                markdown_table(lines, headers_r, rows_r)

        # Target diagnostics
        td = pd.get('target_diag_diff', {})
        if td:
            markdown_section(lines, 'Sizing Q Target Diagnostics', 3)
            for key, val in td.items():
                if isinstance(val, dict) and 'a' in val:
                    d_val = val.get("delta", "—")
                    lines.append(f'- **{key}:** {val["a"]} → {val["b"]} (Δ={d_val})')
            lines.append('')

    # --- Hypotheses ---
    markdown_section(lines, 'Гипотезы', 2)
    hyp = report.get('hypotheses', {})
    for hkey in ['H1_buffer_fold_terminal', 'H2_q_degradation', 'H3_regret_accumulation']:
        h = hyp.get(hkey, {})
        supported = '✓ ПОДТВЕРЖДЕНА' if h.get('supported') else '~ требует проверки'
        lines.append(f'### {hkey.split("_", 1)[0].upper()}: {h.get("description", "")}')
        lines.append(f'**Статус:** {supported}')
        lines.append('')
        for e in h.get('evidence', []):
            lines.append(f'- {e}')
        lines.append('')

    # --- First to break ---
    ftb = hyp.get('_first_to_break', {})
    if ftb:
        markdown_section(lines, 'Что ломается ПЕРВЫМ', 2)
        lines.append(f'Анализ первого среза ({ftb.get("pair", "")}):')
        lines.append('')
        for i, s in enumerate(ftb.get('signals_ranked', []), 1):
            lines.append(f'{i}. **{s["metric"]}** — |Δ|={s["abs_delta"]}')
        lines.append('')

    # --- Recommendations ---
    markdown_section(lines, 'Рекомендации для #85 и шага 4', 2)
    lines.append('На основе выявленного механизма:')
    lines.append('')
    lines.append('1. Если H1 подтверждена (буфер fold-terminal): #85 terminal-balanced loss должен смягчить relapse — '
                 'взвешивание по действиям компенсирует дисбаланс сэмплов.')
    lines.append('2. Если H2 подтверждена (Q-деградация): шаг 4 — нормализация целей (Huber/Pop-Art) '
                 'и n-step/MC propagation — критичны для стабилизации bootstrap-канала.')
    lines.append('3. Если H3 подтверждена (regret-накопление): нужен механизм сброса/регуляризации regret, '
                 'возможно annealing max_norm или warmup.')

    with open(path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))


def main():
    parser = argparse.ArgumentParser(
        description='relapse_diff.py — сравнение чекпоинтов на фиксированном наборе состояний',
    )
    parser.add_argument('checkpoint_dir', help='Путь к директории с чекпоинтами (напр. models/test84seed1)')
    parser.add_argument('--iters', type=str, default='100,200,300',
                        help='Номера итераций через запятую (default: 100,200,300)')
    parser.add_argument('--num-states', type=int, default=1000,
                        help='Количество фиксированных состояний (default: 1000)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Seed для генерации состояний (default: 42)')
    parser.add_argument('--output', type=str, default=None,
                        help='Путь к выходному markdown-файлу')
    parser.add_argument('--json', type=str, default=None,
                        help='Путь к выходному JSON-файлу')
    args = parser.parse_args()

    iters = [int(x.strip()) for x in args.iters.split(',')]

    run_relapse_diff(
        checkpoint_dir=args.checkpoint_dir,
        iters=iters,
        num_states=args.num_states,
        seed=args.seed,
        output_md=args.output,
        output_json=args.json,
    )


if __name__ == '__main__':
    main()
