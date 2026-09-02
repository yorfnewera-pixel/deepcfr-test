"""Peek at sizing network predictions for a specific poker state.

Загружает чекпоинт, генерирует стейты, показывает 15-way sizing-распределение
для выбранного стейта.

Usage:
    py -m tools.peek_sizing models/test102/multi_checkpoint_iter_700_light.pt --state-idx 3
    py -m tools.peek_sizing models/test102/multi_checkpoint_iter_700_light.pt --state-idx 0 --seed 42 --num-states 20
"""

import argparse
import random
import numpy as np
import torch
import torch.nn.functional as F

from tools.checkpoint_tools import sample_states, load_full_checkpoint
from src.core.model import StrategySizingNet, encode_state
from src.core.deep_cfr import regret_matching_anchors


def describe_state(state, player_id=0):
    """Человекочитаемое описание стейта."""
    rank_names = ['2', '3', '4', '5', '6', '7', '8', '9', 'T', 'J', 'Q', 'K', 'A']
    suit_names = ['♠', '♥', '♦', '♣']
    stage_names = ['Префлоп', 'Флоп', 'Тёрн', 'Ривер', 'Шоудаун']

    hand = state.players_state[player_id].hand
    hand_str = ' '.join(
        f"{rank_names[int(c.rank)]}{suit_names[int(c.suit)]}" for c in hand
    )
    board_str = ' '.join(
        f"{rank_names[int(c.rank)]}{suit_names[int(c.suit)]}" for c in state.public_cards
    )
    stage = stage_names[int(state.stage)] if int(state.stage) < len(stage_names) else '?'
    pot = float(state.pot)
    stake = float(state.players_state[player_id].stake)
    button = int(state.button)

    pos_desc = []
    for i in range(len(state.players_state)):
        p = (player_id + i) % len(state.players_state)
        if i == 0:
            pos_desc.append(f"Я(стек={float(state.players_state[p].stake):.0f})")
        else:
            pos_desc.append(f"P{p}({float(state.players_state[p].stake):.0f})")
        if p == button:
            pos_desc[-1] += '[BTN]'

    print(f"Рука:     {hand_str}")
    print(f"Доска:    {board_str or '—'}")
    print(f"Стадия:   {stage}")
    print(f"Пот:      {pot:.1f}")
    print(f"Стек:     {stake:.1f}")
    print(f"Позиция:  {' → '.join(pos_desc)}")
    print(f"Легально: {[str(a)[11:] for a in state.legal_actions]}")
    print(f"Мин.бет:  {float(state.min_bet):.1f}")


def main():
    parser = argparse.ArgumentParser(description='Peek at sizing network predictions.')
    parser.add_argument('checkpoint', help='Путь к чекпоинту (.pt)')
    parser.add_argument('--state-idx', type=int, default=0, help='Индекс стейта (0..num-states-1)')
    parser.add_argument('--seed', type=int, default=1, help='Seed для генерации стейтов')
    parser.add_argument('--num-states', type=int, default=10, help='Количество генерируемых стейтов')
    parser.add_argument('--show-action', action='store_true', help='Также показать action-голову (Fold/Check/Call/Raise)')
    parser.add_argument('--device', default='cpu')
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    print(f"=== Peek Sizing: {args.checkpoint} ===\n")

    nets = load_full_checkpoint(args.checkpoint, args.device)
    cfg = nets.get('config', {})
    anchors = nets.get('anchors', [0.1, 0.25, 0.33, 0.5, 0.66, 0.75, 1.0,
                                   1.25, 1.5, 1.75, 2.0, 2.25, 2.5, 2.75, 3.0])
    sizing_hidden = cfg.get('sizing_hidden_size', 128)
    sizing_input_size = nets.get('sizing_input_size',
                                  nets['strategy_sizing_net'].base[0].in_features)

    states = sample_states(args.num_states, seed=args.seed)
    if args.state_idx >= len(states):
        print(f"ERROR: state_idx={args.state_idx} >= num_states={len(states)}")
        return

    state = states[args.state_idx]
    player_id = int(state.current_player)
    print(f"--- State {args.state_idx}/{len(states)} (player={player_id}) ---")
    describe_state(state, player_id)

    use_ma = nets.get('use_multi_agent', False)
    from src.core.model import encode_state_with_position
    encode_fn = encode_state_with_position if use_ma else encode_state
    encoded = encode_fn(state, player_id).astype(np.float32)
    tensor = torch.from_numpy(encoded).unsqueeze(0).to(args.device)

    with torch.inference_mode():
        slot_logits_t, _scalar_bet = nets['strategy_sizing_net'](tensor)
        slot_logits = slot_logits_t[0].cpu().numpy()
    probs = F.softmax(torch.tensor(slot_logits), dim=0).numpy()

    try:
        import pokers as pkrs
        avail, _ = _anchor_availability_stub(state, anchors)
    except Exception:
        avail = np.ones(len(anchors), dtype=bool)

    print(f"\n{'Anchor':>8}  {'Logit':>8}  {'Softmax%':>9}  {'Доступен':>9}")
    print("-" * 42)
    for i, a in enumerate(anchors):
        marker = "✅" if avail[i] else "❌"
        print(f"{a:>8.2f}  {slot_logits[i]:>+8.3f}  {probs[i]*100:>8.2f}%  {marker:>9}")

    bucket_groups_raw = cfg.get('sizing_bucket_groups', [[0.1,0.25,0.33,0.5,0.66],[0.75,1.0,1.25,1.5,1.75],[2.0,2.25,2.5,2.75,3.0]])
    bucket_groups = []
    for group in bucket_groups_raw:
        indices = []
        for sz in group:
            best_i = min(range(len(anchors)), key=lambda i: abs(anchors[i] - float(sz)))
            indices.append(best_i)
        bucket_groups.append(sorted(set(indices)))
    bucket_names = ['Small', 'Medium', 'Large']
    bucket_probs_softmax = F.softmax(torch.tensor(bucket_logits), dim=0).numpy()
    print(f"\n{'Bucket':>8}  {'Logit':>8}  {'Softmax%':>9}  {'Mass%':>8}")
    print("-" * 42)
    for b, indices in enumerate(bucket_groups):
        mass = float(probs[indices].sum()) * 100
        print(f"{bucket_names[b]:>8}  {bucket_logits[b]:>+8.3f}  {bucket_probs_softmax[b]*100:>8.2f}%  {mass:>7.2f}%")

    print("\nRegret-matching (min_prob=0.003):")
    rm_probs = regret_matching_anchors(slot_logits, min_prob=0.003)
    for i, a in enumerate(anchors):
        if rm_probs[i] > 0.001:
            print(f"  {a:.2f}: {rm_probs[i]*100:.2f}%")

    if getattr(args, 'show_action', False) and 'strategy_net' in nets:
        action_names = ['Fold', 'Check', 'Call', 'Raise']
        legal_set = {int(a) for a in state.legal_actions}
        with torch.inference_mode():
            a_out = nets['strategy_net'](tensor)
            a_logits = a_out[0][0].cpu().numpy() if isinstance(a_out, tuple) else a_out[0].cpu().numpy()
        a_probs = F.softmax(torch.tensor(a_logits), dim=0).numpy()
        print(f"\n{'Action':>8}  {'Logit':>8}  {'Softmax%':>9}  {'Легально':>9}")
        print("-" * 42)
        for i, name in enumerate(action_names):
            if i >= len(a_logits):
                break
            legal = "✅" if i in legal_set else "❌"
            print(f"{name:>8}  {a_logits[i]:>+8.3f}  {a_probs[i]*100:>8.2f}%  {legal:>9}")


def _anchor_availability_stub(state, anchors):
    """Упрощённая проверка доступности анкеров."""
    import pokers as pkrs
    min_bet = float(state.min_bet)
    pot = float(state.pot)
    stake = float(state.players_state[int(state.current_player)].stake)
    avail = np.ones(len(anchors), dtype=bool)
    allin_idx = -1
    for i, a in enumerate(anchors):
        bet_amount = a * pot
        if bet_amount < min_bet and a != anchors[-1]:
            avail[i] = False
        if bet_amount > stake:
            avail[i] = False
        if avail[i] and allin_idx < 0:
            allin_idx = i
    return avail, allin_idx


if __name__ == '__main__':
    main()
