"""Action-Q Role & Legal Diagnostics — read-only для старого чекпоинта.

Разводит гипотезы:
  H1: raise обездолен bootstrap-only/delayed reward.
  H2: q_compare fold>raise misleading из-за illegal fold / distribution mismatch.

Использование:
  py tools/action_q_role_legal_diag.py \
    --checkpoint models/test79seed1/multi_checkpoint_iter_100.pt \
    --out-json models/test79seed1/action_q_role_legal_diag_iter_100.json \
    --out-md sizing_reports/81-action-q-role-legal-diagnostics.md
"""
import argparse
import json
import math
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

_pkrs = None
_pkrs_error = None
try:
    import pkrs
    _pkrs = pkrs
except ImportError as e:
    _pkrs_error = str(e)

from tools.checkpoint_tools import (
    load_full_checkpoint, sample_states, filter_raise_states,
    encode_batch, run_q_comparison, run_action_q_buffer_diagnostics,
    run_action_q_model_diagnostics,
)
from tools.checkpoint_tools import encode_state as _encode_state

ACTION_NAMES = {0: 'fold', 1: 'check', 2: 'call', 3: 'raise'}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _has_nets(nets, *keys):
    return all(k in nets for k in keys)


def _rounded(d, ndigits=4):
    if isinstance(d, dict):
        return {k: _rounded(v, ndigits) for k, v in d.items()}
    if isinstance(d, list):
        return [_rounded(v, ndigits) for v in d]
    if isinstance(d, float):
        return round(d, ndigits)
    return d


def _quantiles(arr, *qs):
    arr = np.asarray(arr, dtype=np.float64)
    arr = arr[~np.isnan(arr)]
    if len(arr) == 0:
        return {str(q): None for q in qs}
    return {str(q): float(np.percentile(arr, float(q))) for q in qs}


def _ci_proportion(n, p):
    """Wilson confidence interval нижняя/верхняя граница."""
    if n == 0:
        return 0.0, 0.0
    z = 1.96
    p_ = p / 100.0
    denom = 1 + z ** 2 / n
    centre = (p_ + z ** 2 / (2 * n)) / denom
    margin = z * math.sqrt((p_ * (1 - p_) + z ** 2 / (4 * n)) / n) / denom
    lo = max(0.0, centre - margin) * 100.0
    hi = min(100.0, centre + margin) * 100.0
    return round(lo, 1), round(hi, 1)


# ---------------------------------------------------------------------------
# 1. q_compare legal-mask
# ---------------------------------------------------------------------------

def run_q_compare_legal_mask(nets, n_states=500, seed=42):
    """Генерирует состояния, получает legal mask и Q от q_net.

    Ключевой вопрос: fold легален в этих состояниях?
    Если нет — fold>raise из q_compare может быть misleading.
    """
    try:
        states = sample_states(n_states, seed=seed)
        raise_states = filter_raise_states(states)
    except Exception as e:
        return {'available': False, 'reason': f'cannot sample states: {e}'}

    if not raise_states:
        return {'available': False, 'reason': 'no raise-legal states', 'n_states': 0}

    result = {'available': True, 'n_states': len(raise_states)}
    has_q = _has_nets(nets, 'q_net')
    has_sq = _has_nets(nets, 'sizing_q_net')

    # Legal mask per state
    legal_masks = []
    action_names = ['fold', 'check', 'call', 'raise']
    for s in raise_states:
        mask = np.zeros(4, dtype=np.float32)
        for a in s.legal_actions:
            mask[int(a)] = 1.0
        legal_masks.append(mask)
    legal_masks = np.stack(legal_masks)

    legal_counts = legal_masks.sum(axis=0)
    for i, name in enumerate(action_names):
        result[f'{name}_legal_pct'] = round(float(legal_counts[i]) / len(raise_states) * 100.0, 1)

    # Q by legal action
    if has_q:
        _use_multi = nets.get('config', {}).get('use_multi_agent_advantage', False)
        X_raise = encode_batch(raise_states, use_multi_agent=_use_multi)
        with torch.inference_mode():
            q_vals = nets['q_net'](X_raise).cpu().numpy()

        # Best legal action per state
        best_legal = {}
        for i in range(len(raise_states)):
            mask = legal_masks[i]
            legal_q = np.where(mask > 0, q_vals[i], -np.inf)
            best_a = int(np.argmax(legal_q))
            best_legal[action_names[best_a]] = best_legal.get(action_names[best_a], 0) + 1
        result['best_legal_action_dist'] = best_legal

        # raise vs best legal
        raise_lt_best_legal = 0
        raise_lt_fold_when_legal = 0
        fold_legal_count = 0
        for i in range(len(raise_states)):
            mask = legal_masks[i]
            legal_q = np.where(mask > 0, q_vals[i], -np.inf)
            best_q = legal_q.max()
            if q_vals[i, 3] < best_q:
                raise_lt_best_legal += 1
            if mask[0] > 0:
                fold_legal_count += 1
                if q_vals[i, 3] < q_vals[i, 0]:
                    raise_lt_fold_when_legal += 1

        result['raise_lt_best_legal_pct'] = round(float(raise_lt_best_legal) / len(raise_states) * 100.0, 1)
        result['raise_lt_fold_when_fold_legal_pct'] = round(
            float(raise_lt_fold_when_legal) / max(fold_legal_count, 1) * 100.0, 1
        )
    else:
        result['best_legal_action_dist'] = None
        result['raise_lt_best_legal_pct'] = None
        result['raise_lt_fold_when_fold_legal_pct'] = None

    return result


# ---------------------------------------------------------------------------
# 2. replay action-Q by (action, next_is_hero, terminal)
# ---------------------------------------------------------------------------

def run_replay_by_role(nets):
    """Cross-tab: action × next_is_hero × terminal.

    next_is_hero = True означает, что следующий state.current_player == traversing_player.
    Не actor_role, но полезный прокси.
    """
    has_model = all(
        k in nets for k in ['q_net', 'q_buffer_states', 'q_buffer_next_states',
                            'q_buffer_next_policy_states', 'q_buffer_next_masks']
    )
    has_target = 'q_target_net' in nets
    has_next_hero = 'q_buffer_next_is_hero' in nets

    actions = np.asarray(nets['q_buffer_actions'])
    rewards = np.asarray(nets['q_buffer_rewards'], dtype=np.float32)
    terminals = np.asarray(nets['q_buffer_terminals']).astype(np.float32)

    if has_next_hero:
        next_hero = np.asarray(nets['q_buffer_next_is_hero']).astype(np.float32)
    else:
        next_hero = np.full_like(terminals, -1.0)

    result = {
        'available': True,
        'total_samples': int(len(actions)),
        'actor_role_available': False,
        'actor_role_note': 'next_is_hero = (next_state.current_player == traversing_player). Не actor_role.',
        'cells': {},
    }

    # q_pred if model available
    q_live = None
    q_target_arr = None
    if has_model:
        states = np.asarray(nets['q_buffer_states'], dtype=np.float32)
        next_states = np.asarray(nets['q_buffer_next_states'], dtype=np.float32)
        next_pol = np.asarray(nets['q_buffer_next_policy_states'], dtype=np.float32)
        next_msk = np.asarray(nets['q_buffer_next_masks'], dtype=np.float32)

        with torch.inference_mode():
            s_t = torch.from_numpy(states).float()
            q_live = nets['q_net'](s_t).cpu().numpy()

            if has_target:
                ns_t = torch.from_numpy(next_states).float()
                nq = nets['q_target_net'](ns_t).cpu().numpy()
                adv = nets['advantage_net'](torch.from_numpy(next_pol).float()).cpu().numpy()
                pos = np.maximum(adv, 0) * next_msk
                pos_sum = pos.sum(axis=1, keepdims=True)
                next_strat = np.where(pos_sum > 0, pos / pos_sum.clip(1e-8),
                                      next_msk / next_msk.sum(axis=1, keepdims=True).clip(1.0))
                next_v = (next_strat * nq).sum(axis=1)
                q_target_arr = np.where(terminals > 0.5, rewards, rewards + next_v)
            else:
                q_target_arr = rewards.copy()

    role_names = {1.0: 'next_hero', 0.0: 'next_opponent', -1.0: 'unknown_next_role'}
    term_names = {1.0: 'terminal', 0.0: 'nonterminal'}

    for a_id, a_name in ACTION_NAMES.items():
        a_mask = actions == a_id
        a_count = int(a_mask.sum())
        if a_count == 0:
            continue

        for nh_val in sorted(set(next_hero[a_mask].tolist())):
            nh_mask = (next_hero == nh_val) if nh_val >= 0 else slice(None)
            for t_val, t_name in term_names.items():
                cell_mask = a_mask & nh_mask & (terminals == t_val)
                n = int(cell_mask.sum())
                if n == 0:
                    continue

                cell_key = f'{a_name}|{role_names.get(float(nh_val), str(nh_val))}|{t_name}'
                cell = {'count': n}

                cell_r = rewards[cell_mask]
                cell.update({
                    'reward_mean': float(cell_r.mean()),
                    'reward_std': float(cell_r.std()),
                    'reward_min': float(cell_r.min()),
                    'reward_max': float(cell_r.max()),
                })

                if q_live is not None:
                    cell['q_pred_mean'] = float(q_live[cell_mask, a_id].mean())
                    cell['q_pred_std'] = float(q_live[cell_mask, a_id].std())

                if q_target_arr is not None:
                    cell['target_mean'] = float(q_target_arr[cell_mask].mean())
                    cell['target_std'] = float(q_target_arr[cell_mask].std())
                    cell['target_min'] = float(q_target_arr[cell_mask].min())
                    cell['target_max'] = float(q_target_arr[cell_mask].max())

                    if q_live is not None:
                        td = q_target_arr[cell_mask] - q_live[cell_mask, a_id]
                        cell['td_error_mean'] = float(td.mean())
                        cell['td_error_std'] = float(td.std())

                result['cells'][cell_key] = _rounded(cell)

    return result


# ---------------------------------------------------------------------------
# 3. terminal positive tail
# ---------------------------------------------------------------------------

def run_terminal_positive_tail(nets):
    """Для fold/call/check/raise: сколько terminal sample имеют target > threshold."""
    actions = np.asarray(nets['q_buffer_actions'])
    rewards = np.asarray(nets['q_buffer_rewards'], dtype=np.float32)
    terminals = np.asarray(nets['q_buffer_terminals']).astype(np.float32)

    result = {
        'available': True,
        'actor_role_available': False,
        'note': 'terminal samples сгруппированы по action и next_is_hero. hero_target_gt_0 из next_hero=1',
        'actions': {},
    }

    thresholds = [0, 50, 100, 250, 400]

    has_nh = 'q_buffer_next_is_hero' in nets
    if has_nh:
        next_hero = np.asarray(nets['q_buffer_next_is_hero']).astype(np.float32)
    else:
        next_hero = np.full_like(terminals, -1.0)

    for a_id, a_name in ACTION_NAMES.items():
        a_mask = actions == a_id
        term_mask = a_mask & (terminals > 0.5)
        term_n = int(term_mask.sum())

        entry = {'terminal_count': term_n}
        if term_n == 0:
            entry['target_positive_tail'] = 'no terminal samples'
            result['actions'][a_name] = entry
            continue

        for nh_val in sorted(set(next_hero[term_mask].tolist())):
            role = 'next_hero' if nh_val > 0.5 else ('next_opponent' if nh_val >= 0 else 'unknown_next')
            role_mask = term_mask & (next_hero == nh_val)
            role_n = int(role_mask.sum())
            if role_n == 0:
                continue

            role_rewards = rewards[role_mask]
            role_entry = {'count': role_n}
            for th in thresholds:
                role_entry[f'target_gt_{th}'] = int((role_rewards > th).sum())
            role_entry['target_p99'] = float(np.percentile(role_rewards, 99)) if role_n > 0 else None
            role_entry['target_max'] = float(role_rewards.max())

            entry[role] = role_entry

        result['actions'][a_name] = entry

    return result


# ---------------------------------------------------------------------------
# 4. Raise bootstrap ceiling
# ---------------------------------------------------------------------------

def run_bootstrap_ceiling(nets, device='cpu'):
    """Target quantiles по action с правильным TD-target (reward + next_v).

    Для nonterminal: target = reward + next_v (bootstrap через q_target_net).
    Для terminal: target = reward.
    """
    actions = np.asarray(nets['q_buffer_actions'])
    rewards = np.asarray(nets['q_buffer_rewards'], dtype=np.float32)
    terminals = np.asarray(nets['q_buffer_terminals']).astype(np.float32)

    has_model = all(
        k in nets for k in ['q_target_net', 'advantage_net',
                            'q_buffer_next_states', 'q_buffer_next_policy_states',
                            'q_buffer_next_masks']
    )

    targets = None
    if has_model:
        next_states = np.asarray(nets['q_buffer_next_states'], dtype=np.float32)
        next_pol = np.asarray(nets['q_buffer_next_policy_states'], dtype=np.float32)
        next_msk = np.asarray(nets['q_buffer_next_masks'], dtype=np.float32)
        with torch.inference_mode():
            nq = nets['q_target_net'](torch.from_numpy(next_states).float()).cpu().numpy()
            adv = nets['advantage_net'](torch.from_numpy(next_pol).float()).cpu().numpy()
            pos = np.maximum(adv, 0) * next_msk
            pos_sum = pos.sum(axis=1, keepdims=True)
            next_strat = np.where(pos_sum > 0, pos / pos_sum.clip(1e-8),
                                  next_msk / next_msk.sum(axis=1, keepdims=True).clip(1.0))
            next_v = (next_strat * nq).sum(axis=1)
            targets = np.where(terminals > 0.5, rewards, rewards + next_v)
    else:
        targets = rewards.copy()

    result = {'available': True, 'actions': {}, 'targets_are_td': has_model}

    for a_id, a_name in ACTION_NAMES.items():
        a_mask = actions == a_id
        entry = {}

        for t_val, t_name in [(0.0, 'nonterminal'), (1.0, 'terminal')]:
            cell_mask = a_mask & (terminals == t_val)
            n = int(cell_mask.sum())
            if n == 0:
                entry[t_name] = {'count': 0}
                continue
            cell_t = targets[cell_mask]
            entry[t_name] = {
                'count': n,
                'target_quantiles': _quantiles(cell_t, 1, 10, 50, 90, 99),
                'target_max': float(cell_t.max()),
                'target_min': float(cell_t.min()),
                'target_mean': float(cell_t.mean()),
                'target_std': float(cell_t.std()),
            }
        result['actions'][a_name] = entry

    return result


# ---------------------------------------------------------------------------
# 5. replay vs q_compare distribution comparison
# ---------------------------------------------------------------------------

def run_replay_vs_q_compare_distribution(nets, n_states=500, seed=42):
    """Сравнивает pot/legal_actions распределение между replay и q_compare states."""
    replay_pots = []
    replay_legal_counts = []

    if 'q_buffer_states' in nets:
        # replay: decode первые N состояний
        states_arr = np.asarray(nets['q_buffer_states'], dtype=np.float32)
        n_sample = min(5000, len(states_arr))
        idx = np.random.RandomState(seed + 1).choice(len(states_arr), n_sample, replace=False)
        for i in idx:
            try:
                s = _decode_state_simple(states_arr[i])
                replay_pots.append(s.pot)
                replay_legal_counts.append(len(s.legal_actions))
            except Exception:
                pass

    try:
        compare_states = sample_states(min(n_states, 300), seed=seed + 2)
    except Exception:
        compare_states = []

    q_pots = []
    q_legal_counts = []
    for s in compare_states:
        q_pots.append(s.pot)
        q_legal_counts.append(len(s.legal_actions))

    result = {'available': True}

    def _st(arr):
        if not arr:
            return {'count': 0}
        a = np.array(arr, dtype=np.float32)
        return {
            'count': len(a),
            'mean': float(a.mean()),
            'std': float(a.std()),
            'p50': float(np.percentile(a, 50)),
            'p90': float(np.percentile(a, 90)),
            'max': float(a.max()),
            'min': float(a.min()),
        }

    result['replay_pot'] = _st(replay_pots)
    result['q_compare_pot'] = _st(q_pots)
    result['replay_legal_actions'] = _st(replay_legal_counts)
    result['q_compare_legal_actions'] = _st(q_legal_counts)

    if replay_pots and q_pots:
        result['pot_distribution_match'] = 'close' if abs(
            np.mean(replay_pots) - np.mean(q_pots)) < 30 else 'divergent'
    else:
        result['pot_distribution_match'] = 'cannot_compare'

    return result


def _decode_state_simple(state_arr):
    """Пытается восстановить poker state из numpy encoding.

    Использует pkrs.State.from_vector если доступен, иначе raise.
    """
    if _pkrs is None:
        raise RuntimeError(f'pkrs unavailable: {_pkrs_error}')
    try:
        return _pkrs.State.from_vector(state_arr)
    except (AttributeError, TypeError):
        raise RuntimeError('pkrs.State.from_vector not available')


# ---------------------------------------------------------------------------
# 6. Interpretation flags
# ---------------------------------------------------------------------------

def build_interpretation_flags(legal_mask, by_role, tail, ceiling):
    flags = {}

    # B1
    flags['b1_active_confirmed'] = None  # заполняется из sizing_q_config

    # Raise bootstrap-only
    raise_terminal = 0
    if 'raise' in tail.get('actions', {}):
        raise_terminal = tail['actions']['raise'].get('terminal_count', 0)
    flags['raise_bootstrap_only'] = raise_terminal == 0

    # Raise ceiling
    flags['raise_target_ceiling_confirmed'] = False
    if ceiling.get('actions', {}).get('raise', {}).get('nonterminal', {}).get('target_max', -999) < 1.0:
        flags['raise_target_ceiling_confirmed'] = True

    # Fold positive tail
    fold_tail = tail.get('actions', {}).get('fold', {})
    flags['fold_positive_tail_present'] = False
    for role in ['next_hero', 'next_opponent']:
        if fold_tail.get(role, {}).get('target_gt_100', 0) > 0:
            flags['fold_positive_tail_present'] = True

    # Hero fold positive?
    hero_fold = fold_tail.get('next_hero', {})
    flags['hero_fold_positive_tail_suspected'] = hero_fold.get('target_gt_0', 0) > hero_fold.get('count', 1) * 0.1

    # q_compare fold misleading?
    flags['q_compare_fold_misleading_possible'] = False
    if legal_mask.get('available'):
        if legal_mask.get('fold_legal_pct', 100) < 50:
            flags['q_compare_fold_misleading_possible'] = True

    # Action-Q fold > raise on legal states?
    flags['action_q_fold_gt_raise_confirmed_on_legal_states'] = False
    if legal_mask.get('available') and legal_mask.get('raise_lt_fold_when_fold_legal_pct') is not None:
        if legal_mask['raise_lt_fold_when_fold_legal_pct'] > 50:
            flags['action_q_fold_gt_raise_confirmed_on_legal_states'] = True

    # Actor role
    flags['actor_role_unavailable'] = True  # next_is_hero не равно actor_role

    return flags


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='Action-Q Role & Legal Diagnostics')
    parser.add_argument('--checkpoint', required=True, help='Path to full checkpoint .pt')
    parser.add_argument('--out-json', required=True, help='Output JSON path')
    parser.add_argument('--out-md', required=True, help='Output Markdown path')
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--num-states', type=int, default=500)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    print(f'Loading checkpoint: {args.checkpoint}')
    nets = load_full_checkpoint(args.checkpoint, device=args.device)

    # Existing diagnostics
    print('Running existing diagnostics...')
    try:
        states = sample_states(args.num_states, seed=args.seed)
        raise_states = filter_raise_states(states)
        if raise_states:
            _use_multi = nets.get('config', {}).get('use_multi_agent_advantage', False)
            X_raise = encode_batch(raise_states, device=args.device, use_multi_agent=_use_multi)
            q_compare = run_q_comparison(nets, X_raise)
        else:
            q_compare = {'n_states': 0, 'error': 'no raise-legal states'}
    except Exception as e:
        q_compare = {'error': str(e)}

    aq_buf = run_action_q_buffer_diagnostics(nets)
    aq_model = run_action_q_model_diagnostics(nets, device=args.device)

    # B1 flag — check multiple locations
    b1_flag = None
    for loc in [nets.get('sizing_q_config', {}), nets.get('config', {}), nets]:
        v = loc.get('sizing_q_selected_credit_enabled') if isinstance(loc, dict) else None
        if v is not None:
            b1_flag = bool(v)
            break

    # New diagnostics
    print('Running q_compare legal mask...')
    legal_mask = run_q_compare_legal_mask(nets, n_states=args.num_states, seed=args.seed)

    print('Running replay by role...')
    by_role = run_replay_by_role(nets)

    print('Running terminal positive tail...')
    tail = run_terminal_positive_tail(nets)

    print('Running bootstrap ceiling...')
    ceiling = run_bootstrap_ceiling(nets)

    print('Running replay vs q_compare distribution...')
    dist_cmp = run_replay_vs_q_compare_distribution(nets, n_states=args.num_states, seed=args.seed)

    # Interpretation
    flags = build_interpretation_flags(legal_mask, by_role, tail, ceiling)
    flags['b1_active_confirmed'] = bool(b1_flag) if b1_flag is not None else None

    # Assemble
    output = {
        'checkpoint': os.path.basename(args.checkpoint),
        'iteration': nets.get('iteration', -1),
        'sizing_q_selected_credit_enabled': b1_flag,
        'available_inputs': {
            'q_net': _has_nets(nets, 'q_net'),
            'q_target_net': _has_nets(nets, 'q_target_net'),
            'advantage_net': _has_nets(nets, 'advantage_net'),
            'sizing_q_net': _has_nets(nets, 'sizing_q_net'),
            'q_buffer_states': _has_nets(nets, 'q_buffer_states'),
            'q_buffer_next_is_hero': _has_nets(nets, 'q_buffer_next_is_hero'),
            'checkpoint_format_version': nets.get('checkpoint_format_version'),
        },
        'q_compare': q_compare,
        'q_compare_legal_mask': legal_mask,
        'action_q_buffer': aq_buf,
        'action_q_model': aq_model,
        'replay_action_q_by_role': by_role,
        'terminal_positive_tail': tail,
        'raise_bootstrap_ceiling': ceiling,
        'replay_vs_q_compare_distribution': dist_cmp,
        'interpretation_flags': flags,
    }

    with open(args.out_json, 'w') as f:
        json.dump(output, f, indent=2)
    print(f'JSON saved to {args.out_json}')

    # Markdown
    md = build_markdown_summary(output)
    with open(args.out_md, 'w', encoding='utf-8') as f:
        f.write(md)
    print(f'Markdown saved to {args.out_md}')


def build_markdown_summary(report):
    flags = report.get('interpretation_flags', {})
    legal = report.get('q_compare_legal_mask', {})
    tail = report.get('terminal_positive_tail', {})
    ceiling = report.get('raise_bootstrap_ceiling', {})
    qc = report.get('q_compare', {})
    dist = report.get('replay_vs_q_compare_distribution', {})

    lines = [
        f'# Action-Q Role & Legal Diagnostics — iter {report["iteration"]}',
        '',
        f'> **Checkpoint:** {report["checkpoint"]}',
        f'> **B1 active:** {report.get("sizing_q_selected_credit_enabled")}',
        '',
        '---',
        '',
        '## 1. Interpretation Flags',
        '',
        '| Flag | Value |',
        '|---|---|',
    ]

    flag_labels = {
        'b1_active_confirmed': 'B1 selected-credit active',
        'raise_bootstrap_only': 'Raise — bootstrap-only (0 terminal samples)',
        'raise_target_ceiling_confirmed': 'Raise target ceiling < 1.0 confirmed',
        'fold_positive_tail_present': 'Fold has terminal positive targets >100',
        'hero_fold_positive_tail_suspected': 'HERO fold terminal positive suspected',
        'q_compare_fold_misleading_possible': 'q_compare fold>raise may be misleading (fold illegal)',
        'action_q_fold_gt_raise_confirmed_on_legal_states': 'Fold>raise confirmed on LEGAL states',
        'actor_role_unavailable': 'Actor role not available (next_is_hero proxy)',
    }
    for key, label in flag_labels.items():
        val = flags.get(key)
        lines.append(f'| {label} | {val} |')

    lines += [
        '',
        '---',
        '',
        '## 2. q_compare Legal Mask',
        '',
    ]
    if legal.get('available'):
        lines += [
            f'- **n_states:** {legal["n_states"]}',
            f'- **fold_legal_pct:** {legal.get("fold_legal_pct", "N/A")}%',
            f'- **raise_legal_pct:** {legal.get("raise_legal_pct", "N/A")}%',
            f'- **raise_lt_best_legal_pct:** {legal.get("raise_lt_best_legal_pct", "N/A")}%',
            f'- **raise_lt_fold_when_fold_legal_pct:** {legal.get("raise_lt_fold_when_fold_legal_pct", "N/A")}%',
            '',
            '**Best legal action distribution:**',
            '```',
        ]
        for a_name, cnt in sorted(legal.get('best_legal_action_dist', {}).items()):
            lines.append(f'  {a_name}: {cnt}')
        lines.append('```')
    else:
        lines.append(f'Not available: {legal.get("reason", "unknown")}')

    lines += [
        '',
        '---',
        '',
        '## 3. Raise Bootstrap Ceiling',
        '',
        'Target quantiles:',
        '',
    ]
    for a_name in ['fold', 'check', 'call', 'raise']:
        act = ceiling.get('actions', {}).get(a_name, {})
        lines.append(f'### {a_name}')
        for t_name in ['nonterminal', 'terminal']:
            t = act.get(t_name, {})
            if t.get('count', 0) == 0:
                lines.append(f'- **{t_name}:** no samples')
            else:
                q = t.get('target_quantiles', {})
                lines.append(
                    f'- **{t_name}** (n={t["count"]}): '
                    f'p50={q.get("50")}, p90={q.get("90")}, p99={q.get("99")}, '
                    f'max={t["target_max"]}, mean={t["target_mean"]}'
                )
        lines.append('')

    lines += [
        '---',
        '',
        '## 4. Terminal Positive Tail',
        '',
    ]
    for a_name in ['fold', 'check', 'call', 'raise']:
        act = tail.get('actions', {}).get(a_name, {})
        lines.append(f'### {a_name}')
        lines.append(f'- **terminal_count:** {act.get("terminal_count", 0)}')
        for role in ['next_hero', 'next_opponent', 'unknown_next']:
            r = act.get(role, {})
            if r.get('count', 0) == 0:
                continue
            lines.append(f'- **{role}** (n={r["count"]}): '
                         f'gt_0={r.get("target_gt_0")}, gt_50={r.get("target_gt_50")}, '
                         f'gt_100={r.get("target_gt_100")}, gt_250={r.get("target_gt_250")}, '
                         f'gt_400={r.get("target_gt_400")}, '
                         f'p99={r.get("target_p99")}, max={r.get("target_max")}')
        lines.append('')

    lines += [
        '---',
        '',
        '## 5. Replay vs q_compare Distribution',
        '',
    ]
    for label, key in [('replay pot', 'replay_pot'), ('q_compare pot', 'q_compare_pot'),
                       ('replay legal_actions', 'replay_legal_actions'), ('q_compare legal_actions', 'q_compare_legal_actions')]:
        d = dist.get(key, {})
        if d.get('count', 0) > 0:
            lines.append(f'- **{label}:** mean={d["mean"]:.1f}, p50={d["p50"]:.1f}, '
                         f'p90={d["p90"]:.1f}, max={d["max"]:.1f}, n={d["count"]}')
    lines.append(f'- **pot_distribution_match:** {dist.get("pot_distribution_match", "N/A")}')
    lines.append('')

    # H1 vs H2 resolution
    lines += [
        '---',
        '',
        '## 6. Most Likely Root Cause',
        '',
    ]
    raise_target_only = flags.get('raise_bootstrap_only', False)
    raise_ceiling = flags.get('raise_target_ceiling_confirmed', False)
    fold_legal_gt_raise = flags.get('action_q_fold_gt_raise_confirmed_on_legal_states', False)
    fold_misleading = flags.get('q_compare_fold_misleading_possible', False)
    hero_fold_suspect = flags.get('hero_fold_positive_tail_suspected', False)

    causes = []
    if raise_target_only and raise_ceiling and fold_legal_gt_raise:
        causes.append('**H1: Raise структурно обездолен.** Bootstrap-only + низкий потолок (~+0.4). '
                      'Fold>raise подтверждён на legal states. Это reward-grounding асимметрия.')
    elif fold_misleading:
        causes.append('**H2: q_compare fold>raise misleading.** Fold нелегален в большинстве eval-состояний. '
                      'Нужен diagnostic fix, не training fix.')
    elif raise_target_only and not fold_legal_gt_raise:
        causes.append('**Mixed.** Raise bootstrap-only, но fold>raise не подтверждён на legal states. '
                      'Нужно и legal-mask на q_compare, и reward-propagation для raise.')
    else:
        causes.append('**Inconclusive.** Требуется дополнительная диагностика.')

    if hero_fold_suspect:
        causes.append('**Дополнительно: hero fold positive tail подозрителен.** Возможен bug reward sign/role.')

    for c in causes:
        lines.append(f'- {c}')

    lines += [
        '',
        '---',
        '',
        '## 7. What NOT to do',
        '',
        '- Не запускать новое обучение до разрешения H1/H2.',
        '- Не включать legal-anchor mask / kind-filter.',
        '- Не менять action-Q архитектуру без диагностики.',
        '',
        '---',
        '',
        '## 8. Next Experiment',
        '',
    ]

    if raise_target_only and fold_legal_gt_raise:
        lines += [
            '1. **Action-Q delayed reward propagation** — n-step return или MC target для raise chains.',
            '2. Отдельно: проверить reward sign для hero fold terminal samples.',
            '3. После фикса action-Q — повторный прогон с B1 ON.',
        ]
    elif fold_misleading:
        lines += [
            '1. **Fix q_compare** — legal-mask в diagnostic report.',
            '2. Переоценить fold>raise с legal mask.',
            '3. Если raise всё ещё хуже — переходить к reward-propagation.',
        ]
    else:
        lines += [
            '1. Расширить diagnostic: добавить actor_role в buffer.',
            '2. Повторить q_compare с legal-mask на большем количестве состояний.',
            '3. Принять решение о фиксе после уточнённой диагностики.',
        ]

    return '\n'.join(lines)


if __name__ == '__main__':
    main()
