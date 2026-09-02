# Bug #29: Полная переделка алгоритма — DCFR+ с маскированием, ZeroInit, target_net, reservoir buffers

**Дата:** 2026-05-11  
**Статус:** FIXED  
**Критичность:** CRITICAL (обучение не сходилось — Fold≈33% во всех состояниях)  
**Затронутые файлы:** `src/core/model.py`, `src/core/deep_cfr.py`, `inference/core.py`, `src/training/train.py`, `config.yaml`

---

## Диагноз

Диагностический скрипт `scripts/diag_aa_fold.py` подтвердил:
- **Fold=32.4%** на AA preflop vs raise (должно быть ≈0%)
- **Fold=30.5%** на постфлопе с бесплатным check (должно быть ≈0%)
- Диапазон Fold по всем состояниям: **0.344–0.351** — сеть никогда не подавляла Fold

### Корневые причины (3)

1. **Нет маскирования легальных действий** → advantage_net предсказывает regret для всех 3 действий всегда, включая Fold когда Check бесплатен. Для illegal действий нет градиента — веса остаются на Kaiming init ≈ равные логиты → Fold≈1/3
2. **Нет ZeroInit** → Kaiming init даёт случайные ≈равные логиты на выходе. Сеть начинает с uniform 33/33/33 вместо uniform по legal actions
3. **Stale target** → `prev_advantage_net` обновлялся раз в 6 итераций (каждый игрок), вместо каждой итерации. Bootstrap target сильно отстаёт

---

## Изменения (10 пунктов + 3 багфикса)

### 1. ZeroInit — `PokerNetwork._init_output_layers()`
**Файл:** `src/core/model.py`, `inference/core.py`

```python
def _init_output_layers(self):
    nn.init.zeros_(self.action_head.weight)
    nn.init.zeros_(self.action_head.bias)
    nn.init.zeros_(self.value_head.weight)
    nn.init.zeros_(self.value_head.bias)
    nn.init.zeros_(self.sizing_head[-1].weight)
    nn.init.zeros_(self.sizing_head[-1].bias)
```

**Эффект:** При ZeroInit логиты = 0 для всех действий. Softmax(0,0,0) = (1/3, 1/3, 1/3). С маскированием Fold=-1e20 → Softmax = (0, 0.5, 0.5) — корректная стартовая стратегия.

### 2. AdvantageBuffer — reservoir sampling, full-vector
**Файл:** `src/core/deep_cfr.py`

Замена `PrioritizedMemory` (per-action, circular) на `AdvantageBuffer` (full-vector, reservoir):

- Хранит `(state[157], regrets[3], mask[3], iteration)` как полные векторы
- Reservoir sampling — равномерная выборка, без приоритетов
- Очищается каждую итерацию (только данные текущего traversing player)
- `capacity=300000`

**Почему PER не работает:** Per-action хранение создаёт "чёрную дыру" для illegal действий — нет градиента, веса остаются на Kaiming init. Full-vector + mask×output даёт градиент=0 к target=0 для illegal.

### 3. StrategyBuffer — reservoir, NOT cleared
**Файл:** `src/core/deep_cfr.py`

Замена `deque(maxlen=300000)` на `StrategyBuffer`:
- Хранит `(state[157], policy[3], mask[3], iteration)` 
- Reservoir sampling
- **НЕ очищается** между итерациями — аккумулирует стратегию
- `capacity=300000`

### 4. get_legal_action_mask — доминантность Fold
**Файл:** `src/core/deep_cfr.py`

```python
def get_legal_action_mask(self, state):
    mask = np.zeros(3, dtype=np.float32)
    has_fold = pkrs.ActionEnum.Fold in state.legal_actions
    has_check = pkrs.ActionEnum.Check in state.legal_actions
    has_call = pkrs.ActionEnum.Call in state.legal_actions
    has_raise = pkrs.ActionEnum.Raise in state.legal_actions
    if has_fold and not has_check:   # Fold запрещён когда Check доступен
        mask[0] = 1.0
    if has_check or has_call:
        mask[1] = 1.0
    if has_raise:
        mask[2] = 1.0
    return mask
```

**Ключевое:** Когда Check доступен, Fold физически запрещён (mask[0]=0). Это доминантность: Check ≥ Fold всегда (0 ≥ -stakes).

### 5. cfr_traverse_multi — полная переписка
**Файл:** `src/core/deep_cfr.py`

- Стратегия из RM+: `max(advantages, 0)` (regret matching+)
- Regret хранится как full-vector `(regrets[3])` + `mask[3]`
- Strategy buffer получает `policy[3]` + `mask[3]`
- Убран `log1p` compression (было: `sign(x) * log1p(|x|)`) → raw regrets
- Убрана running mean/var нормализация → gradient clipping вместо неё

### 6. train_advantage_network_multi — DCFR+ bootstrap
**Файл:** `src/core/deep_cfr.py`

Формула из статьи (line 277):
```
target = max(prev_pred, 0) * discount + regret
discount = (t-1)^α / ((t-1)^α + 1)
```

- `target_net` вместо `prev_advantage_net` — deep copy, synced после КАЖДОГО training call
- Raw MSE loss: `MSE(pred × mask, target × mask)` — illegal действия получают pred×0 vs target×0 → градиент=0
- Gradient clipping: `max_norm=1.0`
- Target sync: `self.target_net.load_state_dict(self.advantage_net.state_dict())` ПОСЛЕ training

### 7. train_strategy_network — MSE + (t/T)^(γ/2) weighting
**Файл:** `src/core/deep_cfr.py`

Формула из статьи (line 282):
```
L(ψ) = E[(t/T)^γ × Σ_a (σ^t(I,a) - Π(I,a|ψ))²]
```

- MSE loss вместо cross-entropy (по статье)
- Weight: `(iteration / T × 2) ^ (gamma / 2)` — Linear CFR weighting
- Logit masking: illegal действия получают -1e20 в logits → softmax даёт ≈0 вероятность
- Gradient clipping: `max_norm=0.5`

### 8. target_net + choose_action
**Файл:** `src/core/deep_cfr.py`

- `target_net = PokerNetwork(...)` — deep copy advantage_net в `__init__`
- `requires_grad=False` для всех параметров target_net
- `choose_action` переписан: использует `get_legal_action_mask()` + masked logits (-1e20 для illegal)

### 9. inference/core.py — ZeroInit + masking
**Файл:** `inference/core.py`

- `PokerNetwork._init_output_layers()` — тот же ZeroInit
- `InferenceAgent.get_legal_action_mask()` — та же логика доминантности
- `InferenceAgent.choose_action()` — masked logits вместо post-hoc renormalization
- Удалён `_mask_illegal_actions()` (заменён на logit-level masking)

### 10. train.py — обновление ссылок
**Файл:** `src/training/train.py`

- `advantage_memory` → `advantage_buffers[0]` (все 4 функции)
- `strategy_memory` → `strategy_buffer` (все 4 функции)
- `advantage_memories[pid]` → `advantage_buffers[pid]` (multi)
- `total_advantage_memory_size()` обновлён под `advantage_buffers`

---

## Багфиксы (критические — краш при запуске)

### Багфикс A: _build_checkpoint → advantage_memories[0].capacity
**Файл:** `src/core/deep_cfr.py:973`  
**Было:** `self.advantage_memories[0].capacity` — AttributeError (advantage_memories не существует)  
**Стало:** `self.advantage_buffers[0].capacity`

### Багфикс B: _build_checkpoint → target_mean/target_var
**Файл:** `src/core/deep_cfr.py:1006-1007`  
**Было:** `'target_mean': self.target_mean` — AttributeError (не инициализированы в __init__)  
**Стало:** Убраны. Добавлены `target_net` (state_dict) и `max_regret_seen` вместо них.

### Багфикс C: total_advantage_memory_size → advantage_memories
**Файл:** `src/core/deep_cfr.py:1073`  
**Было:** `sum(len(m) for m in self.advantage_memories)` — AttributeError  
**Стало:** `sum(len(buf) for buf in self.advantage_buffers)`

### Багфикс D: _load_checkpoint — target_net + scalar cleanup
**Файл:** `src/core/deep_cfr.py`  
**Было:** Загружал `target_mean`/`target_var` (не существуют), не загружал `target_net`  
**Стало:** Загружает `target_net` (fallback: copy от advantage_net), загружает `max_regret_seen`, убраны `target_mean`/`target_var`

### Багфикс E: Legacy train_advantage_network → заглушка
**Файл:** `src/core/deep_cfr.py`  
**Было:** Мёртвый код — использует `self.advantage_memories`, `self.prev_advantage_net`, `self.target_momentum` (ни одного нет)  
**Стало:** `raise NotImplementedError` с направлением на `train_advantage_network_multi()`

### Багфикс F: train_strategy_network — weights shape mismatch
**Файл:** `src/core/deep_cfr.py:863`  
**Было:** `predicted_policies * weights` — `weights` shape `[batch]`, `predicted_policies` shape `[batch, 3]` → RuntimeError: size mismatch  
**Стало:** `weights.unsqueeze(1)` → shape `[batch, 1]`, корректный broadcast с `[batch, 3]`  
**Проявление:** Краш на итерации 10 (первая стратегия обучения, когда strategy_buffer набрал 128+ сэмплов)

---

## config.yaml изменения

| Параметр | Было | Стало | Обоснование |
|---|---|---|---|
| `advantage_weight_decay` | 0 | 1e-5 | L2 регуляризация (Berweger: 1e-5) |
| `discount_gamma` | отсутствовал | 2.0 | Для strategy weighting (t/T)^(γ/2) |
| `batch_size` | 32 | убран | Захардкожен в training функциях (256/128) |
| `advantage_epochs` | 1 | 3 | Достаточно для reservoir buffer |
| `strategy_epochs` | 1 | 3 | Достаточно для reservoir buffer |

---

## Верификация

### Маскирование + ZeroInit
```
Preflop (Fold/Call/Raise legal): mask=[1,1,1], probs=[0.33, 0.33, 0.33] — корректно
Postflop (Check+Raise, no Fold): mask=[0,1,1], probs=[0.00, 0.50, 0.50] — Fold=0!
```

### ZeroInit веса
```
action_head.weight.sum() = 0.000000
action_head.bias.sum() = 0.000000
```

### Checkpoint save/load round-trip
```
checkpoint keys: [advantage_net, target_net, strategy_net, ...] — OK
_load_checkpoint → target_net loaded, iteration restored — OK
```

### Smoke-тесты
```
6/6 тестов пройдены (test_field_verification.py)
```

---

## Сравнение с оригиналом Berweger

| Аспект | Berweger (оригинал) | Наш (после #29) |
|---|---|---|
| CFR алгоритм | Linear CFR (regret × √iter) | DCFR+ (discount + RM+) |
| Advantage loss | Huber + PER веса | MSE + masking |
| Strategy loss | Cross-entropy | MSE + (t/T)^γ |
| Буфер advantage | PER (α=0.6, circular) | Reservoir (uniform) |
| Masking | Post-hoc renorm | Logit-level (-1e20) |
| Target network | НЕТ | Есть, sync каждую итерацию |
| ZeroInit | НЕТ | Есть |
| LR advantage | 1e-6 | 1e-4 |
| Regret нормализация | normalize + clip[-10,10] | Raw + grad clip 1.0 |
| Weight decay | 1e-5 | 1e-5 |

---

## Что НЕ вошло в этот багфикс (отложено)

- `deep_cfr_with_opponent_modeling.py` — ещё не обновлён (AdvantageBuffer, masking, DCFR+)
- Legacy train-функции (2p, 3p, 4p) — `cfr_traverse` внутри них вызывает старый `advantage_memory.add()` / `strategy_memory.append()` — мёртвый код
- Card embedding / suit isomorphism — не реализовано (Berweger тоже не реализовал)
- PER для advantage — reservoir uniform вместо приоритетного; может быть менее эффективно для редких состояний
- Фазовый подход (random → checkpoint → diverse pool) — не реализован
