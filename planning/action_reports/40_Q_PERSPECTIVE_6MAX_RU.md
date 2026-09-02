# Баг-репорт #40: Q-сеть perspective неоднородна — нарушена семантика для 6-max

## Серьёзность: HIGH

## Категория: Архитектурный баг / Semantic correctness

## Дата обнаружения: 2026-05-15

---

## Описание

Q-сеть (QValueNetwork) используется как baseline для variance reduction в DeepDCFR+.  
В статье DeepDCFR для 2-player zero-sum Q-сеть предсказывает value player 1, а для player 2 используется `Q2 = -Q1`.  
В 6-max это не работает: Q-сеть должна предсказывать value **traversing player**, независимо от acting player.

**Текущее поведение (баг):**

1. **Opponent branch** (`cfr_traverse_multi`): Q-сеть получает `encode_state(state, current_player)` — perspective **opponent**, а не traversing player. Target при этом — reward **traversing player**. Input и target в разных перспективах.

2. **next_state в q_buffer**: `encode_state(new_state, new_state.current_player)` — perspective следующего acting player, а не traversing player. Q-target в `train_q_network` использует `advantage_net(next_state)` для strategy bootstrap — но advantage_net на opponent-perspective state предсказывает advantages **opponent**, а не traversing player.

3. **train_q_network**: `advantage_net(next_states_t)` и `q_net(next_states_t)` работают с одними и теми же states. Но advantage нужно perspective acting player (для strategy), а Q-value — perspective traversing player (для value target). Конфликт.

**Итог:** Q-сеть обучается на смеси perspective — иногда input = acting player, иногда = traversing player, target всегда traversing player. Это создаёт неоднородную регрессионную задачу, снижающую качество Q-baseline и увеличивающую variance OS-оценок.

---

## Затронутые файлы

- `src/core/deep_cfr.py` — `QValueBuffer`, `cfr_traverse_multi` (opponent branch), `_cfr_traverse_multi_outcome_node`, `train_q_network`
- `config.yaml` — `q_buffer_size`
- `src/utils/config.py` — default `q_buffer_size`
- `tests/test_hybrid_outcome_sampling.py` — `q_buffer.add` calls
- `src/training/train.py` — default в логе

---

## Фикс

### 1. Q-buffer size: 50000 → 300000

Увеличен буфер replay для лучшего покрытия пространства состояний 6-max.

**Файлы:** `config.yaml`, `src/utils/config.py`, `src/training/train.py`

### 2. QValueBuffer: добавлено next_policy_states

Буфер теперь хранит два отдельных next-state:
- `next_state` — `encode_state(new_state, traversing_player)` → для Q-value bootstrap
- `next_policy_state` — `encode_state(new_state, new_state.current_player)` → для strategy bootstrap

**Файл:** `src/core/deep_cfr.py` — класс `QValueBuffer`

### 3. Opponent branch: разделены Q и policy encoding

```python
# ДО (баг): Q-сеть получала opponent-perspective input
q_values = self.q_net(opp_state_tensor)[0].cpu().numpy()

# ПОСЛЕ: Q-сеть получает traversing-player-perspective input
q_encoded = encode_state(state, traversing_player)
q_state_tensor = torch.from_numpy(
    q_encoded.astype(np.float32, copy=False)
).unsqueeze(0).to(self.device)
q_values = self.q_net(q_state_tensor)[0].cpu().numpy()
```

Opponent strategy всё ещё считается из `opp_state_tensor` (perspective `current_player`).  
Q-baseline теперь из `q_state_tensor` (perspective `traversing_player`).

**Файл:** `src/core/deep_cfr.py` — `cfr_traverse_multi`, opponent branch (~L1086-1168)

### 4. OS node: next_state perspective исправлен

```python
# ДО (баг):
next_encoded = encode_state(new_state, new_state.current_player)

# ПОСЛЕ:
next_q_encoded = encode_state(new_state, traversing_player)
next_policy_encoded = encode_state(new_state, new_state.current_player)
```

**Файл:** `src/core/deep_cfr.py` — `_cfr_traverse_multi_outcome_node` (~L950-970)

### 5. train_q_network: разделены Q и policy для next-state

```python
# ДО (баг): один next_state для advantage и Q
next_adv, _, _, _ = self.advantage_net(next_states_t)
next_q = self.q_net(next_states_t)

# ПОСЛЕ: advantage из policy perspective, Q из target perspective
next_adv, _, _, _ = self.advantage_net(next_policy_states_t)
next_q = self.q_net(next_states_t)
```

**Файл:** `src/core/deep_cfr.py` — `train_q_network`

### 6. Тесты обновлены

`q_buffer.add()` теперь требует `next_policy_state` — все 3 вызова в `test_hybrid_outcome_sampling.py` обновлены.

---

## Критерии успеха

1. Q-buffer capacity = 300000
2. Q-сеть всегда получает state encoded from `traversing_player` perspective
3. Opponent strategy всё ещё считается from `current_player` perspective
4. Q target strategy считается from `new_state.current_player` perspective (через `next_policy_states`)
5. `train_q_network` не падает
6. OS diagnostics не становятся NaN/inf
7. `os_q_baseline_abs_error_mean` со временем падает или хотя бы не взрывается
8. Checkpoint совместимость: q_buffer не сохраняется в чекпоинтах — проблем нет

---

## Что НЕ менялось

- Архитектура QValueNetwork
- Reward semantics
- Advantage target formula
- Opponent strategy computation logic (только разделение encoding)
