"""Step 3: Диагностика механизма max_norm=100 через цепочку q_net → advantage → политика.

Сравнивает test79seed1 (max_norm=1.0, коллапс) vs test80bseed1 (max_norm=100, OK).
Проверяет гипотезу: max_norm=100 улучшает Q-дифференциацию между действиями,
что меняет advantage-цели через control variate / bootstrap канал.
"""
import argparse
import json
import numpy as np
import torch
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tools.checkpoint_tools import load_full_checkpoint


def forward_chain(nets, states_t, actions_t, terminals_t):
    """Полный forward-pass: Q → advantage → стратегия для diagnostic comparison."""
    device = states_t.device
    qnet = nets['q_net']
    advnet = nets['advantage_net']
    stratnet = nets.get('strategy_net')

    with torch.no_grad():
        # Q-values
        q_all = qnet(states_t)
        q_selected = q_all.gather(1, actions_t.unsqueeze(1)).squeeze(1)

        # Advantages
        adv_all = advnet(states_t)
        adv_pos = torch.clamp(adv_all, min=0)

        # Strategy from advantages (simplified regret matching)
        adv_sum = adv_pos.sum(dim=1, keepdim=True)
        uniform = torch.ones_like(adv_pos) / adv_pos.shape[1]
        strategy = torch.where(adv_sum > 0, adv_pos / adv_sum, uniform)

        # Per-action breakdown
        q_fold = q_all[:, 0]
        q_checkcall = q_all[:, 1:3].mean(dim=1)
        q_raise = q_all[:, 3]

        adv_fold = adv_all[:, 0]
        adv_checkcall = adv_all[:, 1:3].mean(dim=1)
        adv_raise = adv_all[:, 3]

        strat_fold = strategy[:, 0]
        strat_checkcall = strategy[:, 1:3].sum(dim=1)
        strat_raise = strategy[:, 3]

    return {
        'q_fold': q_fold, 'q_checkcall': q_checkcall, 'q_raise': q_raise,
        'q_selected': q_selected,
        'adv_fold': adv_fold, 'adv_checkcall': adv_checkcall, 'adv_raise': adv_raise,
        'strat_fold': strat_fold, 'strat_checkcall': strat_checkcall, 'strat_raise': strat_raise,
    }


def main():
    parser = argparse.ArgumentParser(description='Step 3: механизм max_norm=100')
    parser.add_argument('--base', default='models/test79seed1/multi_checkpoint_iter_100.pt',
                        help='Base checkpoint (max_norm=1.0)')
    parser.add_argument('--target', default='models/test80bseed1/multi_checkpoint_iter_100.pt',
                        help='Target checkpoint (max_norm=100)')
    parser.add_argument('--n-samples', type=int, default=2000)
    parser.add_argument('--device', default='cpu')
    args = parser.parse_args()

    print(f'Loading base: {args.base}')
    nets_base = load_full_checkpoint(args.base, device=args.device)
    print(f'Loading target: {args.target}')
    nets_target = load_full_checkpoint(args.target, device=args.device)

    if 'q_buffer_states' not in nets_base or 'q_buffer_actions' not in nets_base:
        print('ERROR: base checkpoint missing q_buffer')
        sys.exit(1)
    if 'q_buffer_states' not in nets_target or 'q_buffer_actions' not in nets_target:
        print('ERROR: target checkpoint missing q_buffer')
        sys.exit(1)

    # Take first N samples from target's q_buffer
    n = min(args.n_samples, len(nets_target['q_buffer_states']))
    print(f'Using {n} samples from q_buffer')

    states_np = nets_target['q_buffer_states'][:n].astype(np.float32)
    actions_np = nets_target['q_buffer_actions'][:n]
    terminals_np = nets_target.get('q_buffer_terminals', np.zeros(n))[:n]

    states_t = torch.from_numpy(states_np).to(args.device)
    actions_t = torch.from_numpy(actions_np).long().to(args.device)
    terminals_t = torch.from_numpy(terminals_np).to(args.device)

    r_base = forward_chain(nets_base, states_t, actions_t, terminals_t)
    r_target = forward_chain(nets_target, states_t, actions_t, terminals_t)

    term_mask = terminals_t.bool()
    boot_mask = ~term_mask

    # === Diagnostic report ===
    print('\n' + '=' * 60)
    print('  Step 3: Q → Advantage → Strategy Chain Analysis')
    print('=' * 60)

    print(f'\n--- Q-values (all samples, n={n}) ---')
    for label in ['q_fold', 'q_checkcall', 'q_raise', 'q_selected']:
        b_mean = r_base[label].mean().item()
        t_mean = r_target[label].mean().item()
        ratio = t_mean / max(abs(b_mean), 1e-8)
        print(f'  {label:20s}: base={b_mean:8.4f}  target={t_mean:8.4f}  ratio={ratio:6.2f}x')

    print(f'\n--- Q-values (terminal only, n={term_mask.sum().item()}) ---')
    for label in ['q_fold', 'q_checkcall', 'q_raise']:
        b_mean = r_base[label][term_mask].mean().item()
        t_mean = r_target[label][term_mask].mean().item()
        ratio = t_mean / max(abs(b_mean), 1e-8)
        print(f'  {label:20s}: base={b_mean:8.4f}  target={t_mean:8.4f}  ratio={ratio:6.2f}x')

    print(f'\n--- Q-values (bootstrap only, n={boot_mask.sum().item()}) ---')
    for label in ['q_fold', 'q_checkcall', 'q_raise']:
        b_mean = r_base[label][boot_mask].mean().item()
        t_mean = r_target[label][boot_mask].mean().item()
        ratio = t_mean / max(abs(b_mean), 1e-8)
        print(f'  {label:20s}: base={b_mean:8.4f}  target={t_mean:8.4f}  ratio={ratio:6.2f}x')

    print(f'\n--- Q-differentiation (all samples) ---')
    diff_labels = [
        ('q_raise - q_fold', 'q_raise', 'q_fold'),
        ('q_raise - q_checkcall', 'q_raise', 'q_checkcall'),
        ('q_fold - q_checkcall', 'q_fold', 'q_checkcall'),
    ]
    for name, a, b in diff_labels:
        base_diff = (r_base[a] - r_base[b]).mean().item()
        target_diff = (r_target[a] - r_target[b]).mean().item()
        print(f'  {name:25s}: base={base_diff:+8.4f}  target={target_diff:+8.4f}')

    print(f'\n--- Advantages (all samples) ---')
    for label in ['adv_fold', 'adv_checkcall', 'adv_raise']:
        b_mean = r_base[label].mean().item()
        t_mean = r_target[label].mean().item()
        print(f'  {label:20s}: base={b_mean:8.4f}  target={t_mean:8.4f}')

    print(f'\n--- Strategy (all samples) ---')
    for label in ['strat_fold', 'strat_checkcall', 'strat_raise']:
        b_mean = r_base[label].mean().item()
        t_mean = r_target[label].mean().item()
        print(f'  {label:20s}: base={b_mean:8.4f}  target={t_mean:8.4f}')

    # Correlation: does Q-differentiation predict strategy?
    print(f'\n--- Correlation Q-raise-minus-fold vs strategy-raise ---')
    base_q_diff = r_base['q_raise'] - r_base['q_fold']
    target_q_diff = r_target['q_raise'] - r_target['q_fold']
    base_corr = np.corrcoef(base_q_diff.cpu().numpy(), r_base['strat_raise'].cpu().numpy())[0, 1]
    target_corr = np.corrcoef(target_q_diff.cpu().numpy(), r_target['strat_raise'].cpu().numpy())[0, 1]
    print(f'  base corr:   {base_corr:.4f}')
    print(f'  target corr: {target_corr:.4f}')

    print('\nDone.')


if __name__ == '__main__':
    main()
