"""Action-Q grad-clip / scale bottleneck audit — READ-ONLY.

Не обучает, не пишет веса, не делает optimizer.step().
Только forward+backward для ИЗМЕРЕНИЯ нормы градиента до/после clip_grad_norm_(1.0).

Target-математика 1-в-1 из train_q_network:
  reward_unit = bb (по умолчанию 2.0)
  targets = rewards/reward_unit + (1-terminals) * next_v
  next_v = Σ backup_strategy[a] * q_target_net(next_state)[a]
  backup_strategy = (1-mix)*next_strategy + mix*uniform (mix=0.15)
  loss = MSE(q_selected, targets)
  grad clip: max_norm=1.0

4 сценария батчей:
  mixed_natural — случайный сэмпл из всего буфера
  terminal_only — только terminal=1 сэмплы
  bootstrap_only — только terminal=0 сэмплы
  raise_only — только action=3 сэмплы

Для каждого сценария (--n-batches повторов, усреднённо):
  target_abs_mean, target_abs_max
  raw_total_norm (до clip)
  post_total_norm (после clip)
  отдельно q_head grad норма
  effective_step = q_lr * post_norm

Использование:
  py tools/q_grad_clip_audit.py \
    --checkpoint models/test79seed1/multi_checkpoint_iter_100.pt \
    --out-json models/test79seed1/q_grad_clip_audit_iter_100.json \
    --out-md sizing_reports/82-q-grad-clip-audit.md
"""
import argparse
import json
import math
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from tools.checkpoint_tools import load_full_checkpoint

ACTION_NAMES = {0: 'fold', 1: 'check', 2: 'call', 3: 'raise'}


def _collect_grad_norm(model):
    """Собирает L2 норму градиентов всех параметров модели."""
    total_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            total_norm += p.grad.data.norm(2).item() ** 2
    return math.sqrt(total_norm)


def _collect_head_grad_norm(model):
    """L2 норма градиента только q_head слоя."""
    total = 0.0
    for name, p in model.named_parameters():
        if 'q_head' in name and p.grad is not None:
            total += p.grad.data.norm(2).item() ** 2
    return math.sqrt(total)


def _get_mask_for_scenario(actions, terminals, scenario_name):
    """Возвращает boolean mask для сценария батча."""
    if scenario_name == 'mixed_natural':
        return np.ones(len(actions), dtype=bool)
    elif scenario_name == 'terminal_only':
        return terminals > 0.5
    elif scenario_name == 'bootstrap_only':
        return terminals < 0.5
    elif scenario_name == 'raise_only':
        return actions == 3
    raise ValueError(f'unknown scenario: {scenario_name}')


def run_scenario(nets, scenario_name, batch_size, reward_unit, mix_enabled, uniform_mix,
                 max_norm, q_lr, n_batches, device='cpu'):
    """Прогоняет один сценарий батча n_batches раз, усредняя метрики."""
    actions_all = np.asarray(nets['q_buffer_actions'])
    terminals_all = np.asarray(nets['q_buffer_terminals']).astype(np.float32)
    rewards_all = np.asarray(nets['q_buffer_rewards'], dtype=np.float32)
    states_all = np.asarray(nets['q_buffer_states'], dtype=np.float32)
    next_st_all = np.asarray(nets['q_buffer_next_states'], dtype=np.float32)
    next_pol_all = np.asarray(nets['q_buffer_next_policy_states'], dtype=np.float32)
    next_msk_all = np.asarray(nets['q_buffer_next_masks'], dtype=np.float32)

    global_mask = _get_mask_for_scenario(actions_all, terminals_all, scenario_name)
    global_indices = np.where(global_mask)[0]

    if len(global_indices) < batch_size:
        return {'error': f'scenario={scenario_name} has only {len(global_indices)} samples, need {batch_size}'}

    q_net = nets['q_net']
    q_target = nets.get('q_target_net')
    adv_net = nets.get('advantage_net')
    num_actions = nets.get('num_actions', 4)

    accum = {
        'target_abs_mean': [],
        'target_abs_max': [],
        'raw_total_norm': [],
        'post_total_norm': [],
        'head_raw_norm': [],
        'head_post_norm': [],
    }

    rng = np.random.RandomState(42)
    for _ in range(n_batches):
        idx = rng.choice(global_indices, batch_size, replace=False)

        states_t = torch.from_numpy(states_all[idx].copy()).float().to(device)
        actions_t = torch.from_numpy(actions_all[idx].copy()).long().to(device)
        rewards_t = torch.from_numpy(rewards_all[idx].copy()).float().to(device) / reward_unit
        next_st_t = torch.from_numpy(next_st_all[idx].copy()).float().to(device)
        next_pol_t = torch.from_numpy(next_pol_all[idx].copy()).float().to(device)
        next_msk_t = torch.from_numpy(next_msk_all[idx].copy()).float().to(device)
        terminals_t = torch.from_numpy(terminals_all[idx].copy()).float().to(device)

        # Bootstrap targets — 1-в-1 с train_q_network
        with torch.no_grad():
            next_adv = adv_net(next_pol_t)
            next_pos = torch.clamp(next_adv[:, :num_actions], min=0) * next_msk_t
            next_sum = next_pos.sum(dim=1, keepdim=True)
            uniform_strat = next_msk_t / next_msk_t.sum(dim=1, keepdim=True).clamp(min=1)
            next_strat = torch.where(next_sum > 0, next_pos / next_sum, uniform_strat)

            if mix_enabled and q_target is not None:
                backup_strat = (1.0 - uniform_mix) * next_strat + uniform_mix * uniform_strat
            else:
                backup_strat = next_strat

            if q_target is not None:
                nq = q_target(next_st_t)
            else:
                nq = q_net(next_st_t)
            next_v = (backup_strat * nq).sum(dim=1)
            targets = rewards_t + (1.0 - terminals_t) * next_v

        accum['target_abs_mean'].append(float(targets.abs().mean().item()))
        accum['target_abs_max'].append(float(targets.abs().max().item()))

        # Forward
        current_q = q_net(states_t)
        q_selected = current_q.gather(1, actions_t.unsqueeze(1)).squeeze(1)
        loss = F.mse_loss(q_selected, targets.detach())

        # Backward
        q_net.zero_grad()
        loss.backward()
        raw_norm = _collect_grad_norm(q_net)
        head_raw = _collect_head_grad_norm(q_net)
        accum['raw_total_norm'].append(raw_norm)
        accum['head_raw_norm'].append(head_raw)

        # Clip
        torch.nn.utils.clip_grad_norm_(q_net.parameters(), max_norm=max_norm)
        post_norm = _collect_grad_norm(q_net)
        head_post = _collect_head_grad_norm(q_net)
        accum['post_total_norm'].append(post_norm)
        accum['head_post_norm'].append(head_post)

        q_net.zero_grad()

    result = {
        'scenario': scenario_name,
        'n_batches': n_batches,
        'batch_size': batch_size,
        'reward_unit': reward_unit,
        'max_norm': max_norm,
        'q_lr': q_lr,
    }
    for key in ['target_abs_mean', 'target_abs_max', 'raw_total_norm', 'post_total_norm',
                'head_raw_norm', 'head_post_norm']:
        arr = np.array(accum[key], dtype=np.float64)
        result[key] = round(float(arr.mean()), 6)
        result[f'{key}_std'] = round(float(arr.std()), 6)

    result['throttle_factor'] = round(result['raw_total_norm'] / max(result['post_total_norm'], 1e-8), 2)
    result['head_throttle_factor'] = round(result['head_raw_norm'] / max(result['head_post_norm'], 1e-8), 2)
    result['effective_step_size'] = round(q_lr * result['post_total_norm'], 8)

    return result


def run_mixed_decomposition(nets, batch_size, reward_unit, mix_enabled, uniform_mix,
                            max_norm, q_lr, n_batches, device='cpu'):
    """Внутри natural-батча разлагает вклад terminal vs bootstrap в raw grad."""
    actions_all = np.asarray(nets['q_buffer_actions'])
    terminals_all = np.asarray(nets['q_buffer_terminals']).astype(np.float32)
    rewards_all = np.asarray(nets['q_buffer_rewards'], dtype=np.float32)
    states_all = np.asarray(nets['q_buffer_states'], dtype=np.float32)
    next_st_all = np.asarray(nets['q_buffer_next_states'], dtype=np.float32)
    next_pol_all = np.asarray(nets['q_buffer_next_policy_states'], dtype=np.float32)
    next_msk_all = np.asarray(nets['q_buffer_next_masks'], dtype=np.float32)

    q_net = nets['q_net']
    q_target = nets.get('q_target_net')
    adv_net = nets.get('advantage_net')
    num_actions = nets.get('num_actions', 4)

    results = []
    rng = np.random.RandomState(42)
    for _ in range(n_batches):
        idx = rng.choice(len(actions_all), batch_size, replace=False)
        term_mask = (terminals_all[idx] > 0.5)
        bs_mask = ~term_mask
        n_term = int(term_mask.sum())
        n_bs = int(bs_mask.sum())

        states_t = torch.from_numpy(states_all[idx].copy()).float().to(device)
        actions_t = torch.from_numpy(actions_all[idx].copy()).long().to(device)
        rewards_t = torch.from_numpy(rewards_all[idx].copy()).float().to(device) / reward_unit
        next_st_t = torch.from_numpy(next_st_all[idx].copy()).float().to(device)
        next_pol_t = torch.from_numpy(next_pol_all[idx].copy()).float().to(device)
        next_msk_t = torch.from_numpy(next_msk_all[idx].copy()).float().to(device)
        terminals_t = torch.from_numpy(terminals_all[idx].copy()).float().to(device)

        with torch.no_grad():
            next_adv = adv_net(next_pol_t)
            next_pos = torch.clamp(next_adv[:, :num_actions], min=0) * next_msk_t
            next_sum = next_pos.sum(dim=1, keepdim=True)
            uniform_strat = next_msk_t / next_msk_t.sum(dim=1, keepdim=True).clamp(min=1)
            next_strat = torch.where(next_sum > 0, next_pos / next_sum, uniform_strat)
            if mix_enabled and q_target is not None:
                backup_strat = (1.0 - uniform_mix) * next_strat + uniform_mix * uniform_strat
            else:
                backup_strat = next_strat
            if q_target is not None:
                nq = q_target(next_st_t)
            else:
                nq = q_net(next_st_t)
            next_v = (backup_strat * nq).sum(dim=1)
            targets = rewards_t + (1.0 - terminals_t) * next_v

        # 1. Full batch grad — свежий forward
        q_net.zero_grad()
        q_full = q_net(states_t).gather(1, actions_t.unsqueeze(1)).squeeze(1)
        loss_full = F.mse_loss(q_full, targets.detach())
        loss_full.backward()
        full_grad_norm = _collect_head_grad_norm(q_net)
        q_net.zero_grad()

        # 2. Terminal-only subset grad — свежий forward на subset индексах
        term_grad_norm = 0.0
        if n_term > 0:
            idx_t = torch.from_numpy(np.where(term_mask)[0]).long().to(device)
            q_term = q_net(states_t[idx_t]).gather(1, actions_t[idx_t].unsqueeze(1)).squeeze(1)
            loss_term = F.mse_loss(q_term, targets[idx_t].detach())
            loss_term.backward()
            term_grad_norm = _collect_head_grad_norm(q_net)
            q_net.zero_grad()

        # 3. Bootstrap-only subset grad — свежий forward на subset индексах
        bs_grad_norm = 0.0
        if n_bs > 0:
            idx_b = torch.from_numpy(np.where(bs_mask)[0]).long().to(device)
            q_bs = q_net(states_t[idx_b]).gather(1, actions_t[idx_b].unsqueeze(1)).squeeze(1)
            loss_bs = F.mse_loss(q_bs, targets[idx_b].detach())
            loss_bs.backward()
            bs_grad_norm = _collect_head_grad_norm(q_net)
            q_net.zero_grad()

        results.append({
            'n_terminal': n_term,
            'n_bootstrap': n_bs,
            'terminal_target_abs_mean': float(targets[term_mask].abs().mean().item()) if n_term > 0 else 0.0,
            'bootstrap_target_abs_mean': float(targets[bs_mask].abs().mean().item()) if n_bs > 0 else 0.0,
            'full_grad_norm': full_grad_norm,
            'terminal_grad_norm': term_grad_norm,
            'bootstrap_grad_norm': bs_grad_norm,
        })
        q_net.zero_grad()

    return results


def main():
    parser = argparse.ArgumentParser(description='Action-Q grad-clip / scale audit')
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--out-json', default=None, help='авто: peer-дир + json по имени чекпоинта')
    parser.add_argument('--out-md', default=None, help='авто: peer-дир + .md по имени чекпоинта')
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--n-batches', type=int, default=20)
    parser.add_argument('--max-norm', type=float, default=1.0)
    parser.add_argument('--big-blind', type=float, default=None, help='override reward_unit')
    parser.add_argument('--uniform-mix', type=float, default=None, help='override mix')
    parser.add_argument('--no-mix', action='store_true')
    args = parser.parse_args()

    # Авто-пути: peer-директория от чекпоинта, префикс по имени файла
    ckpt_dir = os.path.dirname(os.path.abspath(args.checkpoint))
    ckpt_stem = os.path.splitext(os.path.basename(args.checkpoint))[0]
    bb_suffix = f'_bb{int(args.big_blind)}' if args.big_blind else ''
    if args.out_json is None:
        args.out_json = os.path.join(ckpt_dir, f'{ckpt_stem}_q_grad_clip_audit{bb_suffix}.json')
    if args.out_md is None:
        args.out_md = os.path.join(ckpt_dir, f'{ckpt_stem}_q_grad_clip_audit{bb_suffix}.md')

    print(f'Loading checkpoint: {args.checkpoint}')
    nets = load_full_checkpoint(args.checkpoint, device=args.device)

    if 'q_net' not in nets or 'q_buffer_actions' not in nets:
        print('ERROR: checkpoint не содержит q_net или q_buffer')
        sys.exit(1)

    cfg = nets.get('config', {}) or {}
    reward_unit = args.big_blind if args.big_blind is not None else float(cfg.get('big_blind', nets.get('big_blind', 2.0)))
    mix_enabled = not args.no_mix and cfg.get('q_bootstrap_policy_mix_enabled', False)
    uniform_mix = float(args.uniform_mix if args.uniform_mix is not None
                        else cfg.get('q_bootstrap_policy_uniform_mix', 0.15))
    q_lr = float(cfg.get('q_lr', cfg.get('lr', 0.001)))
    max_norm = args.max_norm

    print(f'Config: reward_unit={reward_unit}, mix_enabled={mix_enabled}, uniform_mix={uniform_mix}, '
          f'q_lr={q_lr}, max_norm={max_norm}')

    scenarios = ['mixed_natural', 'terminal_only', 'bootstrap_only', 'raise_only']
    results = {}
    for sc in scenarios:
        print(f'Running scenario: {sc}')
        r = run_scenario(nets, sc, args.batch_size, reward_unit, mix_enabled, uniform_mix,
                         max_norm, q_lr, args.n_batches, device=args.device)
        results[sc] = r

    # Mixed decomposition
    print('Running mixed decomposition...')
    decomp = run_mixed_decomposition(nets, args.batch_size, reward_unit, mix_enabled, uniform_mix,
                                     max_norm, q_lr, args.n_batches, device=args.device)

    # Verdict
    term = results.get('terminal_only', {})
    bs = results.get('bootstrap_only', {})
    raw_grad_ratio = term.get('raw_total_norm', 0) / max(bs.get('raw_total_norm', 1e-8), 1e-8)
    post_ratio = term.get('post_total_norm', 0) / max(bs.get('post_total_norm', 1e-8), 1e-8)
    term_throttle = term.get('throttle_factor', 0)
    bs_throttle = bs.get('throttle_factor', 0)

    # Усредним decomposition
    term_grads = [d['terminal_grad_norm'] for d in decomp if d['terminal_grad_norm'] > 0]
    bs_grads = [d['bootstrap_grad_norm'] for d in decomp if d['bootstrap_grad_norm'] > 0]
    term_target_abs = [d['terminal_target_abs_mean'] for d in decomp if d['n_terminal'] > 0]
    bs_target_abs = [d['bootstrap_target_abs_mean'] for d in decomp if d['n_bootstrap'] > 0]

    flags = {
        'terminal_raw_grad_much_larger': raw_grad_ratio > 5.0,
        'post_clip_steps_comparable': 0.5 < post_ratio < 2.0,
        'terminal_grad_heavily_throttled': term_throttle > 5.0,
        'bootstrap_grad_barely_clipped': bs_throttle < 1.5,
    }

    flags['scale_bottleneck_confirmed'] = all([
        flags['terminal_raw_grad_much_larger'],
        flags['post_clip_steps_comparable'],
        flags['terminal_grad_heavily_throttled'],
        flags['bootstrap_grad_barely_clipped'],
    ])

    output = {
        'checkpoint': os.path.basename(args.checkpoint),
        'config': {
            'reward_unit': reward_unit,
            'mix_enabled': mix_enabled,
            'uniform_mix': uniform_mix,
            'q_lr': q_lr,
            'max_norm': max_norm,
            'batch_size': args.batch_size,
            'n_batches': args.n_batches,
        },
        'scenarios': results,
        'decomposition_summary': {
            'n_batches': len(decomp),
            'terminal_grad_norm_mean': round(float(np.mean(term_grads)), 4) if term_grads else None,
            'bootstrap_grad_norm_mean': round(float(np.mean(bs_grads)), 4) if bs_grads else None,
            'terminal_target_abs_mean': round(float(np.mean(term_target_abs)), 4) if term_target_abs else None,
            'bootstrap_target_abs_mean': round(float(np.mean(bs_target_abs)), 4) if bs_target_abs else None,
            'grad_ratio_terminal_vs_bootstrap': round(
                float(np.mean(term_grads) / max(np.mean(bs_grads), 1e-8)), 2) if term_grads and bs_grads else None,
        },
        'verdict': {
            'raw_grad_ratio_terminal_vs_bootstrap': round(raw_grad_ratio, 2),
            'post_clip_ratio_terminal_vs_bootstrap': round(post_ratio, 2),
            'throttle_factor_terminal': round(term_throttle, 2),
            'throttle_factor_bootstrap': round(bs_throttle, 2),
            'flags': flags,
        },
        'decomposition_raw': decomp,
    }

    with open(args.out_json, 'w') as f:
        json.dump(output, f, indent=2)
    print(f'JSON saved to {args.out_json}')

    # Markdown
    if args.out_md:
        md = _build_md(output)
        with open(args.out_md, 'w', encoding='utf-8') as f:
            f.write(md)
        print(f'Markdown saved to {args.out_md}')


def _build_md(output):
    cfg = output['config']
    v = output['verdict']
    flags = v['flags']
    dec = output['decomposition_summary']

    lines = [
        f'# Action-Q Grad-Clip & Scale Audit — iter {output.get("iteration", "?")}',
        '',
        f'> **Checkpoint:** {output["checkpoint"]}',
        f'> **Config:** reward_unit={cfg["reward_unit"]}, mix={cfg["mix_enabled"]}@{cfg["uniform_mix"]}, '
        f'lr={cfg["q_lr"]}, max_norm={cfg["max_norm"]}',
        '',
        '---',
        '',
        '## 1. Verdict',
        '',
        '| Flag | Value |',
        '|---|---|',
        f'| terminal_raw_grad_much_larger (ratio > 5) | **{flags["terminal_raw_grad_much_larger"]}** (ratio={v["raw_grad_ratio_terminal_vs_bootstrap"]}) |',
        f'| post_clip_steps_comparable (0.5 < ratio < 2) | **{flags["post_clip_steps_comparable"]}** (ratio={v["post_clip_ratio_terminal_vs_bootstrap"]}) |',
        f'| terminal_grad_heavily_throttled (throttle > 5) | **{flags["terminal_grad_heavily_throttled"]}** (throttle={v["throttle_factor_terminal"]}) |',
        f'| bootstrap_grad_barely_clipped (throttle < 1.5) | **{flags["bootstrap_grad_barely_clipped"]}** (throttle={v["throttle_factor_bootstrap"]}) |',
        f'| **scale_bottleneck_confirmed** | **{flags["scale_bottleneck_confirmed"]}** |',
        '',
        '---',
        '',
        '## 2. Scenario Comparison',
        '',
        '| Scenario | Target abs(mean) | Target abs(max) | Raw grad | Post grad | Throttle | Eff step |',
        '|---|---|---|---|---|---|---|',
    ]

    for sc_name, sc in output['scenarios'].items():
        if 'error' in sc:
            lines.append(f'| {sc_name} | {sc["error"]} | - | - | - | - | - |')
            continue
        lines.append(
            f'| {sc_name} | {sc["target_abs_mean"]:.4f} | {sc["target_abs_max"]:.4f} | '
            f'{sc["raw_total_norm"]:.4f} | {sc["post_total_norm"]:.4f} | '
            f'{sc["throttle_factor"]} | {sc["effective_step_size"]:.6f} |'
        )

    lines += [
        '',
        '---',
        '',
        '## 3. Mixed Decomposition (terminal vs bootstrap inside natural batch)',
        '',
    ]
    if dec.get('terminal_grad_norm_mean') is not None:
        lines += [
            f'- terminal target abs(mean): {dec["terminal_target_abs_mean"]}',
            f'- bootstrap target abs(mean): {dec["bootstrap_target_abs_mean"]}',
            f'- terminal grad norm: {dec["terminal_grad_norm_mean"]}',
            f'- bootstrap grad norm: {dec["bootstrap_grad_norm_mean"]}',
            f'- grad ratio terminal/bootstrap: {dec["grad_ratio_terminal_vs_bootstrap"]}×',
        ]
    else:
        lines.append('No terminal samples in batches — cannot compare.')

    lines += [
        '',
        '---',
        '',
        '## 4. Interpretation',
        '',
    ]
    if flags['scale_bottleneck_confirmed']:
        lines += [
            '**Scale + grad-clip bottleneck CONFIRMED.**',
            '',
            '- Terminal targets создают raw grad в разы больше bootstrap.',
            '- `clip_grad_norm_(1.0)` обрезает terminal до того же масштаба, что и bootstrap.',
            '- Bootstrap почти не режется clip-ом.',
            '- terminal и bootstrap двигают веса на одинаковый effective step, несмотря на разный масштаб target.',
            '',
            '**→ Перед n-step/MC для raise нужно починить масштаб action-Q.**',
            '- Увеличить `q_reward_scale` (не `bb`=2.0, а ~200-500 или pot-relative).',
            '- Или z-score/Pop-Art targets.',
            '- После scale-fix повторить этот audit — throttle_factor_terminal должен упасть к ~1.',
        ]
    else:
        lines += [
            '**Scale bottleneck NOT confirmed.**',
            '',
            'Причина сжатия Q в ±1 не в grad-clip. Возможно:',
            '- bootstrap majority просто статистически доминирует loss (реже terminal → меньше weight).',
            '- loss weighting / batch composition.',
            '- выходной диапазон сети ограничен инициализацией/архитектурой.',
            '',
            'Нужна дополнительная диагностика.',
        ]

    return '\n'.join(lines)


if __name__ == '__main__':
    main()
