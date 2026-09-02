"""On-policy policy divergence diagnostic.

Генерирует стейты через CFR-траверс (реальные игровые узлы),
а не через синтетический sample_states.

Usage:
    py -m tools.policy_divergence_onpolicy models/test98seed1N5_5/multi_checkpoint_iter_500.pt --num-traversals 50
"""

import argparse
import random
import sys
import numpy as np
import torch
import torch.nn.functional as F

import src.core.deep_cfr
from src.core.model import PokerNetwork, encode_state_with_position


def _load_nets(checkpoint_path, device='cpu'):
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg = ckpt.get('config', {}) or {}
    hidden_size = cfg.get('hidden_size', 256)
    num_actions = cfg.get('num_actions', 4)
    input_size = ckpt['strategy_net']['base.0.weight'].shape[1]
    num_players = cfg.get('num_players', 6)

    nets = []
    if 'advantage_nets' in ckpt and ckpt['advantage_nets']:
        for sd in ckpt['advantage_nets']:
            net = PokerNetwork(input_size=input_size, hidden_size=hidden_size, num_actions=num_actions).to(device)
            net.load_state_dict(sd, strict=False)
            net.eval()
            nets.append(net)
    else:
        sd = ckpt['advantage_net']
        for _ in range(5):
            net = PokerNetwork(input_size=input_size, hidden_size=hidden_size, num_actions=num_actions).to(device)
            net.load_state_dict(sd, strict=False)
            net.eval()
            nets.append(net)
    print(f"  Загружено {len(nets)} сетей")
    return nets, input_size, num_actions, num_players


def _choose_action_advantage(net, state, player_id, device='cpu'):
    """Выбор действия через advantage-сеть без стратегии (чистый regret-matching)."""
    import pokers as pkrs
    encoded = encode_state_with_position(state, player_id).astype(np.float32)
    tensor = torch.from_numpy(encoded).unsqueeze(0).to(device)
    with torch.inference_mode():
        out = net(tensor)
        logits = out[0] if isinstance(out, tuple) else out
    logits_np = logits[0].cpu().numpy()

    legal_mask = np.zeros(4)
    for a in state.legal_actions:
        legal_mask[int(a)] = 1
    masked = np.where(legal_mask > 0, logits_np, -1e20)
    probs = np.exp(masked - masked.max()) / np.sum(np.exp(masked - masked.max()) + 1e-10)

    action_map = {
        0: pkrs.ActionEnum.Fold,
        1: pkrs.ActionEnum.Check,
        2: pkrs.ActionEnum.Call,
        3: pkrs.ActionEnum.Raise,
    }
    action_idx = np.random.choice(4, p=probs)
    action_enum = action_map[action_idx]
    if action_idx == 3:
        bet = max(float(state.min_bet), float(state.min_bet) * 0.5)
        return pkrs.Action(action_enum, bet)
    return pkrs.Action(action_enum)


def collect_traversal_states(nets, num_traversals, traversing_player, device='cpu'):
    """Собирает action-узлы из реальных CFR-траверсов."""
    import pokers as pkrs
    states_collected = []
    n_players = 6

    for _ in range(num_traversals):
        state = pkrs.State.from_seed(
            n_players=n_players,
            button=random.randint(0, n_players - 1),
            sb=1, bb=2, stake=200.0,
            seed=random.randint(0, 1_000_000),
        )

        depth = 0
        while not state.final_state and depth < 100:
            depth += 1

            current = int(state.current_player)
            legal_actions = list(state.legal_actions)
            if not legal_actions:
                break

            # Collect state if it's an action node with raise legal
            if pkrs.ActionEnum.Raise in state.legal_actions and not state.final_state:
                states_collected.append(state)

            # Opponent (random agent at seat 5) plays random
            if current == n_players - 1:
                action = random.choice(legal_actions)
                if action == pkrs.ActionEnum.Raise:
                    bet = max(float(state.min_bet), float(state.min_bet) * random.uniform(0.5, 2.5))
                    action = pkrs.Action(action, bet)
                else:
                    action = pkrs.Action(action)
            else:
                # Use the appropriate network
                net = nets[current % len(nets)]
                player_id = current
                action = _choose_action_advantage(net, state, player_id, device)

            state = state.apply_action(action)
            if state.status != pkrs.StateStatus.Ok:
                break

    return states_collected


def compute_divergence_on_states(nets, states, device='cpu'):
    n_nets = len(nets)
    n_states = len(states)

    all_logits = []
    for net in nets:
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

    for i in range(n_nets):
        probs_i = np.array([F.softmax(torch.tensor(l), dim=0).numpy() for l in all_logits[i]])
        per_net_entropy[i] = float(np.mean(-np.sum(probs_i * np.log(probs_i + 1e-10), axis=1)))

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

    return pairwise_kl, pairwise_agreement, per_net_entropy


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('checkpoint')
    parser.add_argument('--num-traversals', type=int, default=50)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--device', default='cpu')
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    print(f"=== On-Policy Divergence Diagnostic ===")
    print(f"Чекпоинт: {args.checkpoint}")
    print(f"Траверсов: {args.num_traversals}, seed={args.seed}")

    nets, input_size, num_actions, num_players = _load_nets(args.checkpoint, args.device)

    # Collect on-policy states from all 5 traversing perspectives
    all_states = []
    for tp in range(5):
        states = collect_traversal_states(nets, args.num_traversals, tp, args.device)
        all_states.extend(states)
        print(f"  traversing {tp}: собрано {len(states)} стейтов")

    print(f"  Всего on-policy стейтов: {len(all_states)}")

    if len(all_states) < 50:
        print("  СЛИШКОМ МАЛО стейтов для надёжного замера")
        sys.exit(1)

    pairwise_kl, pairwise_agreement, per_net_entropy = compute_divergence_on_states(
        nets, all_states, args.device
    )

    n_nets = len(nets)
    actions = ['Fold', 'Check', 'Call', 'Raise']

    print(f"\n{'='*75}")
    print(f"Per-network entropy (on-policy)")
    print(f"{'Сеть':>6}  {'Entropy':>8}")
    for i in range(n_nets):
        print(f"{i:>6}  {per_net_entropy[i]:8.4f}")

    header = "       " + "".join(f"  net{i}  " for i in range(n_nets))

    print(f"\n{'='*75}")
    print(f"Pairwise KL divergence (on-policy)")
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
    print(f"\nMean pairwise KL (on-policy): {mean_kl:.5f}")

    print(f"\n{'='*75}")
    print(f"Pairwise action agreement (on-policy)")
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
    print(f"\nMean action agreement (on-policy): {mean_agreement:.2%}")

    print(f"\n{'='*75}")
    print(f"Сравнение синтетика vs on-policy:")
    print(f"  Данный замер — on-policy (реальные игровые стейты)")
    print(f"  Для сравнения с синтетикой запусти tools.policy_divergence")


if __name__ == '__main__':
    main()
