"""Диагностика: почему модель фолдит AA префлоп?

Проверяет:
1. legal_actions кодировку (Fold+Check одновременно?)
2. choose_action маскирование
3. Выход сети для AA сценария
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pokers as pkrs
import numpy as np
from src.core.model import encode_state

def find_aa_hand(seed_start=0, max_seeds=5000):
    """Ищем раздачу где у player 0 карманные тузы"""
    for seed in range(seed_start, seed_start + max_seeds):
        state = pkrs.State.from_seed(n_players=6, button=0, sb=1.0, bb=2.0, stake=200.0, seed=seed)
        hand = state.players_state[0].hand
        ranks = [int(c.rank) for c in hand]
        if ranks == [12, 12]:
            return seed, state
    return None, None

def diag_legal_actions():
    """Проверяем какие legal_actions pokers даёт когда Check доступен"""
    print("=" * 60)
    print("ДИАГНОЗ 1: legal_actions когда Check доступен")
    print("=" * 60)

    for seed in range(50):
        state = pkrs.State.from_seed(n_players=6, button=0, sb=1.0, bb=2.0, stake=200.0, seed=seed)
        cp = state.current_player

        has_check = pkrs.ActionEnum.Check in state.legal_actions
        has_fold = pkrs.ActionEnum.Fold in state.legal_actions

        if has_check and has_fold:
            print(f"  seed={seed}, player={cp}: Fold+Check ОДНОВРЕМЕННО! "
                  f"legal={[int(a) for a in state.legal_actions]}")
            break
    else:
        print("  Не найдено Fold+Check в первых 50 seed (префлоп обычно Fold+Call+Raise)")

    # После лимпов — BB с чеком
    for seed in range(200):
        state = pkrs.State.from_seed(n_players=6, button=0, sb=1.0, bb=2.0, stake=200.0, seed=seed)
        steps = 0
        while not state.final_state and steps < 20:
            cp = state.current_player
            has_check = pkrs.ActionEnum.Check in state.legal_actions
            has_fold = pkrs.ActionEnum.Fold in state.legal_actions

            if has_check and has_fold:
                print(f"  seed={seed}, step={steps}, player={cp}: "
                      f"Fold+Check ОДНОВРЕМЕННО! legal={[int(a) for a in state.legal_actions]}")
                print(f"  → get_legal_action_types вернёт [0, 1, ...] — Fold НЕ замаскирован!")
                return state

            if pkrs.ActionEnum.Check in state.legal_actions:
                state = state.apply_action(pkrs.Action(pkrs.ActionEnum.Check))
            elif pkrs.ActionEnum.Call in state.legal_actions:
                state = state.apply_action(pkrs.Action(pkrs.ActionEnum.Call))
            elif pkrs.ActionEnum.Fold in state.legal_actions:
                state = state.apply_action(pkrs.Action(pkrs.ActionEnum.Fold))
            else:
                state = state.apply_action(pkrs.Action(pkrs.ActionEnum.Raise, 4.0))
            steps += 1

    print("  Не удалось найти Fold+Check ситуацию")
    return None

def diag_aa_scenario():
    """Тестируем сеть на AA префлоп против рейза"""
    print("\n" + "=" * 60)
    print("ДИАГНОЗ 2: AA префлоп против рейза — что выбирает сеть?")
    print("=" * 60)

    import torch
    import torch.nn.functional as F
    from src.core.deep_cfr import DeepCFRAgent

    import glob
    checkpoint_paths = sorted(glob.glob("models/multi/*_light.pt"))
    latest = checkpoint_paths[-1] if checkpoint_paths else None
    if latest:
        print(f"  Чекпоинт: {latest}")

    if not latest or not os.path.isfile(latest):
        print("  Чекпоинт не найден — проверяем только encode_state")
        diag_encode_only()
        return

    try:
        agent = DeepCFRAgent(player_id=0, device='cpu')
        ckpt = torch.load(latest, map_location='cpu', weights_only=False)
        iter_num = ckpt.get('iteration', '?')
        input_size = ckpt.get('strategy_net', {}).get('base.0.weight', torch.zeros(1)).shape[1]
        print(f"  Итерация: {iter_num}, input_size: {input_size}")
    except Exception as e:
        print(f"  Ошибка загрузки агента: {e}")
        diag_encode_only()
        return

    # Ситуация: player 0 имеет AA, кто-то зарейзил
    seed, state = find_aa_hand()
    if seed is None:
        print("  Не удалось найти AA руку")
        return

    print(f"  seed={seed}, hand={[f'{c.rank}:{c.suit}' for c in state.players_state[0].hand]}")
    print(f"  stage={state.stage}, pot={state.pot}, legal={[int(a) for a in state.legal_actions]}")

    # Симулируем: все лимпают, потом кто-то рейзит
    actions_taken = 0
    while not state.final_state and actions_taken < 10:
        cp = state.current_player
        if cp == 0:
            break  # Наш ход с AA
        if pkrs.ActionEnum.Raise in state.legal_actions and actions_taken >= 3:
            # Кто-то рейзит
            state = state.apply_action(pkrs.Action(pkrs.ActionEnum.Raise, 6.0))
        elif pkrs.ActionEnum.Call in state.legal_actions:
            state = state.apply_action(pkrs.Action(pkrs.ActionEnum.Call))
        elif pkrs.ActionEnum.Check in state.legal_actions:
            state = state.apply_action(pkrs.Action(pkrs.ActionEnum.Check))
        else:
            state = state.apply_action(pkrs.Action(pkrs.ActionEnum.Fold))
        actions_taken += 1

    if state.final_state:
        print("  Раздача закончилась до нашего хода")
        return

    print(f"  Перед нашим ходом: stage={state.stage}, pot={state.pot}")
    print(f"  legal_actions={[int(a) for a in state.legal_actions]}")
    print(f"  min_bet={state.min_bet}, bet_chips={state.players_state[0].bet_chips}")

    legal_action_types = agent.get_legal_action_types(state)
    print(f"  get_legal_action_types → {legal_action_types}")
    print(f"    0=Fold, 1=Check/Call, 2=Raise")

    has_check = pkrs.ActionEnum.Check in state.legal_actions
    has_call = pkrs.ActionEnum.Call in state.legal_actions
    has_fold = pkrs.ActionEnum.Fold in state.legal_actions
    print(f"  Check={has_check}, Call={has_call}, Fold={has_fold}")

    if has_fold and (has_check or has_call):
        print("  ⚠️ Fold доступен но Check/Call тоже — Fold должен быть замаскирован!")

    # Прогоняем через сеть
    encoded = encode_state(state, 0)
    print(f"  encode_state длина: {len(encoded)}")

    if len(encoded) != input_size:
        print(f"  ❌ НЕСОВМЕСТИМОСТЬ: encode_state={len(encoded)}, чекпоинт={input_size}")
        return

    state_tensor = torch.FloatTensor(encoded).unsqueeze(0)
    with torch.inference_mode():
        logits, z_mean, _, _ = agent.strategy_net(state_tensor)
        probs = F.softmax(logits, dim=1)[0].cpu().numpy()

    print(f"\n  Сырые вероятности сети (softmax):")
    print(f"    Fold:  {probs[0]:.4f}")
    print(f"    C/Call:{probs[1]:.4f}")
    print(f"    Raise: {probs[2]:.4f}")

    # Без маскирования Fold
    legal_probs_raw = np.array([probs[a] for a in legal_action_types])
    legal_probs_raw = legal_probs_raw / legal_probs_raw.sum()
    print(f"\n  Без маскирования Fold (текущее поведение choose_action):")
    for i, a in enumerate(legal_action_types):
        labels = {0: "Fold", 1: "Check/Call", 2: "Raise"}
        print(f"    {labels[a]}: {legal_probs_raw[i]:.4f}")

    # С маскированием Fold когда Check/Call доступен
    masked_legal = [a for a in legal_action_types if not (a == 0 and (has_check or has_call))]
    if masked_legal:
        legal_probs_masked = np.array([probs[a] for a in masked_legal])
        legal_probs_masked = legal_probs_masked / legal_probs_masked.sum()
        print(f"\n  С маскированием Fold (inference/core.py логика):")
        for i, a in enumerate(masked_legal):
            labels = {0: "Fold", 1: "Check/Call", 2: "Raise"}
            print(f"    {labels[a]}: {legal_probs_masked[i]:.4f}")

    # Запускаем 100 раз
    fold_count = 0
    for _ in range(1000):
        action = agent.choose_action(state)
        if action.action == pkrs.ActionEnum.Fold:
            fold_count += 1

    print(f"\n  1000 вызовов choose_action: Fold={fold_count}, "
          f"дробь={fold_count/10:.1f}%")
    if fold_count > 0:
        print("  ❌ МОДЕЛЬ ФОЛДИТ AA! Fold не замаскирован когда Call/Check доступен")
    else:
        print("  ✅ Модель не фолдит AA")

def diag_encode_only():
    """Проверяем encode_state для AA без загрузки чекпоинта"""
    seed, state = find_aa_hand()
    if seed is None:
        print("  Не удалось найти AA руку")
        return

    print(f"  seed={seed}, hand={[f'{c.rank}:{c.suit}' for c in state.players_state[0].hand]}")
    enc = encode_state(state, 0)
    print(f"  encode_state длина: {len(enc)}")

    # Проверяем legal_actions encoding
    legal_idx = 52 + 52 + 5 + 1 + 6 + 6 + 6*4 + 1 + 1
    legal_enc = enc[legal_idx:legal_idx+4]
    print(f"  legal_actions_enc: {legal_enc} (позиции {legal_idx}-{legal_idx+3})")
    print(f"  pokers legal_actions: {[int(a) for a in state.legal_actions]}")

if __name__ == "__main__":
    diag_legal_actions()
    diag_aa_scenario()
