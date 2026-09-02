"""Policy divergence diagnostic for multi-agent checkpoints.

Сравнивает per-seat advantage-сети на фиксированном наборе стейтов.
Показывает, насколько расходятся политики 5 сетей.

Usage:
    py -m tools.policy_divergence models/test98seed1N5_5/multi_checkpoint_iter_500.pt --num-states 300
"""

import argparse
import sys
import random

import numpy as np
import torch
import torch.nn.functional as F

from tools.checkpoint_tools import sample_states
from src.core.model import PokerNetwork, encode_state_with_position


def load_multi_agent_nets(checkpoint_path, device='cpu'):
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg = ckpt.get('config', {}) or {}
    hidden_size = cfg.get('hidden_size', 256)
    num_actions = cfg.get('num_actions', 4)
    input_size = ckpt['strategy_net']['base.0.weight'].shape[1]

    advantage_nets = []
    if 'advantage_nets' in ckpt and ckpt['advantage_nets']:
        for sd in ckpt['advantage_nets']:
            net = PokerNetwork(input_size=input_size, hidden_size=hidden_size, num_actions=num_actions).to(device)
            net.load_state_dict(sd, strict=False)
            net.eval()
            advantage_nets.append(net)
    elif 'advantage_net' in ckpt:
        sd = ckpt['advantage_net']
        for _ in range(5):
            net = PokerNetwork(input_size=input_size, hidden_size=hidden_size, num_actions=num_actions).to(device)
            net.load_state_dict(sd, strict=False)
            net.eval()
            advantage_nets.append(net)
        print("  INFO: v2 чекпоинт — все сети идентичны (divergence=0 по определению)")
    else:
        print("  ERROR: нет advantage_nets и нет advantage_net в чекпоинте")
        sys.exit(1)

    print(f"  Загружено {len(advantage_nets)} advantage-сетей (input_size={input_size})")
    return advantage_nets, input_size, num_actions


def compute_divergence(nets, states, device='cpu'):
    n_nets = len(nets)
    n_states = len(states)

    all_logits = []
    for net_idx, net in enumerate(nets):
        net_logits = np.zeros((n_states, 4), dtype=np.float32)
        for i, state in enumerate(states):
            player_id = int(state.current_player) % 5
            encoded = encode_state_with_position(state, player_id).astype(np.float32)
            tensor = torch.from_numpy(encoded).unsqueeze(0).to(device)
            with torch.inference_mode():
                out = net(tensor)
                logits = out[0] if isinstance(out, tuple) else out
            net_logits[i] = logits[0].cpu().numpy()
        all_logits.append(net_logits)

    pairwise_kl = np.zeros((n_nets, n_nets), dtype=np.float32)
    pairwise_agreement = np.zeros((n_nets, n_nets), dtype=np.float32)
    per_net_entropy = np.zeros(n_nets, dtype=np.float32)
    per_net_argmax = np.zeros((n_nets, 4), dtype=np.float32)

    for i in range(n_nets):
        probs_i = np.array([F.softmax(torch.tensor(l), dim=0).numpy() for l in all_logits[i]])
        per_net_entropy[i] = float(np.mean(-np.sum(probs_i * np.log(probs_i + 1e-10), axis=1)))
        per_net_argmax[i] = np.bincount(np.argmax(probs_i, axis=1), minlength=4).astype(np.float32)
        per_net_argmax[i] /= n_states

        for j in range(n_nets):
            if i >= j:
                continue
            probs_j = np.array([F.softmax(torch.tensor(l), dim=0).numpy() for l in all_logits[j]])
            probs_i_clip = np.clip(probs_i, 1e-10, 1.0)
            probs_j_clip = np.clip(probs_j, 1e-10, 1.0)
            kl_ij = float(np.mean(np.sum(probs_i_clip * np.log(probs_i_clip / probs_j_clip), axis=1)))
            kl_ji = float(np.mean(np.sum(probs_j_clip * np.log(probs_j_clip / probs_i_clip), axis=1)))
            pairwise_kl[i, j] = 0.5 * (kl_ij + kl_ji)
            pairwise_agreement[i, j] = float(np.mean(np.argmax(probs_i, axis=1) == np.argmax(probs_j, axis=1)))

    return pairwise_kl, pairwise_agreement, per_net_entropy, per_net_argmax


def main():
    parser = argparse.ArgumentParser(description='Multi-agent policy divergence diagnostic.')
    parser.add_argument('checkpoint', help='Путь к чекпоинту')
    parser.add_argument('--num-states', type=int, default=300)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--device', default='cpu')
    args = parser.parse_args()

    print(f"=== Policy Divergence Diagnostic ===")
    print(f"Чекпоинт: {args.checkpoint}")
    print(f"Стейтов: {args.num_states}, seed={args.seed}")

    nets, input_size, num_actions = load_multi_agent_nets(args.checkpoint, args.device)
    states = sample_states(args.num_states, seed=args.seed)
    print(f"  Сгенерировано стейтов: {len(states)}")

    pairwise_kl, pairwise_agreement, per_net_entropy, per_net_argmax = compute_divergence(
        nets, states, args.device
    )

    n_nets = len(nets)
    actions = ['Fold', 'Check', 'Call', 'Raise']

    print(f"\n{'='*75}")
    print(f"Per-network entropy")
    print(f"{'Сеть':>6}  {'Entropy':>8}  {'Fold':>6} {'Check':>6} {'Call':>6} {'Raise':>6}")
    for i in range(n_nets):
        print(f"{i:>6}  {per_net_entropy[i]:8.4f}  "
              f"{per_net_argmax[i][0]:6.1%} {per_net_argmax[i][1]:6.1%} "
              f"{per_net_argmax[i][2]:6.1%} {per_net_argmax[i][3]:6.1%}")

    print(f"\n{'='*75}")
    print(f"Pairwise KL divergence (0 = идентичны)")
    header = "       " + "".join(f"  net{i}  " for i in range(n_nets))
    print(header)
    for i in range(n_nets):
        row = f"  net{i} "
        for j in range(n_nets):
            if i == j:
                row += f"    -    "
            elif i < j:
                row += f" {pairwise_kl[i, j]:8.5f}"
            else:
                row += f" {pairwise_kl[j, i]:8.5f}"
        print(row)

    mean_kl = float(np.mean([pairwise_kl[i, j] for i in range(n_nets) for j in range(i + 1, n_nets)]))
    print(f"\nMean pairwise KL: {mean_kl:.5f}")

    print(f"\n{'='*75}")
    print(f"Pairwise action agreement (% совпадения argmax)")
    print(header)
    for i in range(n_nets):
        row = f"  net{i} "
        for j in range(n_nets):
            if i == j:
                row += f"    -    "
            elif i < j:
                row += f" {pairwise_agreement[i, j]:8.2%}"
            else:
                row += f" {pairwise_agreement[j, i]:8.2%}"
        print(row)

    mean_agreement = float(np.mean([pairwise_agreement[i, j] for i in range(n_nets) for j in range(i + 1, n_nets)]))
    print(f"\nMean action agreement: {mean_agreement:.2%}")

    print(f"\n{'='*75}")
    print(f"Интерпретация:")
    if mean_kl < 0.05 and mean_agreement > 0.90:
        print(f"  Сети ПРАКТИЧЕСКИ ИДЕНТИЧНЫ: дивергенции нет, пересадка бесполезна.")
    elif mean_kl < 0.10:
        print(f"  Сети СЛАБО РАЗЛИЧАЮТСЯ: малая дивергенция, пересадка вероятно избыточна.")
    elif mean_kl < 0.25:
        print(f"  Сети УМЕРЕННО РАЗЛИЧАЮТСЯ: политики заметно разные, пересадка МОЖЕТ дать улучшение.")
    else:
        print(f"  Сети СИЛЬНО РАЗЛИЧАЮТСЯ: политики существенно разные, пересадка РЕКОМЕНДУЕТСЯ.")


if __name__ == '__main__':
    main()
