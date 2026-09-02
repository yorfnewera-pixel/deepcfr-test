# Баг-репорт: Архитектурная миграция на Soft-CFR (P1-P9)

**Проект**: deepcfr-test  
**Дата**: 2026-05-07  
**Серьёзность**: Critical (архитектурный)  
**Статус**: Fixed  

---

## Описание

Замена классической табличной логики CFR на нейросетевую логику Soft-CFR, устойчивую к шуму и отрицательным значениям регрета в 6-max NL Hold'em.

**Ключевой тезис**: «Мы не фиксим баги, мы переводим архитектуру с классической (табличной) логики CFR на нейросетевую логику Soft-CFR, которая устойчива к шуму и отрицательным значениям регрета».

---

## Корневые причины (устранены)

### P1: clip[-2,2] уничтожает 70% градиента

**Файлы**: `deep_cfr.py:340,499`, `train.py:191`

log1p уже ограничивает: log1p(10^18) ≈ 41.5. Дополнительный clip[-2,2] обрезал:
- 71.9% fold-regrets
- 62.7% call-regrets
- 69.2% raise-regrets

Корреляция raw regret и clipped values: 0.27–0.45 (почти случайная).

**Фикс**: Убран `np.clip(scaled_regret, -2.0, 2.0)`, оставлен `sign(x)*log1p(|x|)`.

---

### P2: HuberLoss delta=1.0 не соответствует масштабу log1p

**Файлы**: `deep_cfr.py:604,651`

Таргеты в log1p-пространстве ±5, старый delta=1.0 для ±2. При delta=1 и таргетах ±5 — loss почти всегда линейный (L1), не квадратичный.

**Фикс**: `F.smooth_l1_loss(..., beta=5.0)`.

---

### P3: Regret Matching + argmax fallback = fold-lock (КЛЮЧЕВОЙ)

**Файлы**: `deep_cfr.py:286-300`, `deep_cfr.py:447-462`, `train.py:122-137`

Каскад fold-collapse:
1. Все advantages ≤ 0 → argmax fallback → 100% fold
2. strategy_memory заполняется fold-стратегиями
3. strategy_net учит fold → оппоненты fold → маленькие action_values
4. Маленькие regrets → ещё больше clipping → LOOP

**Фикс**: Softmax-policy с temperature decay:
```python
temperature = max(0.5, 5.0 - 4.5 * iteration / 3000.0)
adv_logits[a] = advantages[a] / temperature
strategy = softmax(adv_logits)  # всегда валидное распределение
```

**Удалены**: argmax fallback, ε-greedy (5 костылей → 1 механизм).

---

### P4: Сырой regret ±200 в PG memory

**Файлы**: `deep_cfr.py:365,513`, `train.py:207`

regret_raise = action_values[2] - ev может быть ±200 (200bb стеки). Running baseline не может отследить такой масштаб.

**Фикс**: `sign(x)*log1p(|x|)` при записи в pg_memory. ±200 → ±5.3.

---

### P5: log_prob → +13.8 → PG explosion

**Файлы**: `model.py:56,93`

При tanh(z)→1: `_log_prob_correction` даёт `-log(1 - 1 + 1e-6)` → +13.8. Умножение на advantage → взрыв PG loss.

**Фикс**: `torch.clamp(correction, -5.0, 5.0)` и `torch.clamp(log_prob, -5.0, 5.0)`.

---

### P6: log_std_min=-5.0 → энтропия → 0

**Файл**: `model.py:46`

std_min = e^(-5) ≈ 0.0067 — стратегия становится почти детерминированной после 3000 итераций (entropy bonus decay).

**Фикс**: `torch.clamp(log_std, -2.0, 0.0)`. std_min = e^(-2) ≈ 0.135.

---

### P9: Running baseline — константа, не зависит от состояния

**Файлы**: `model.py`, `deep_cfr.py`

`pg_baseline_mean` — одно число для всех состояний. AA и 72o получают одинаковый baseline.

**Фикс**: `value_head = nn.Linear(hidden_size, 1)` в PokerNetwork. `advantage = regret_raise - V(state).detach()`. Обучается MSE. Advantage clamp [-3.0, 3.0].

---

## Что удалено (костыли → архитектура)

| Костыль | Где был | Заменён на |
|---------|---------|------------|
| argmax fallback | deep_cfr.py:292-294, 454-456; train.py:130-131 | Softmax-policy |
| ε-greedy exploration | deep_cfr.py:296-300, 458-462; train.py:133-137 | τ-decay в softmax |
| running_mean baseline | deep_cfr.py:152-158, 682-685 | Value Head (P9) |
| _update_pg_baseline | deep_cfr.py:152-158 | Больше не нужен |
| advantage clamp [-1,1] | deep_cfr.py:685 | Value Head + clamp [-3,3] |

---

## Изменённые файлы

| Файл | Изменения |
|------|-----------|
| `src/core/model.py` | value_head, forward→4 выхода, log_std_min=-2, log_prob clamp[-5,5] |
| `src/core/deep_cfr.py` | softmax-policy, log1p PG, HuberLoss beta=5, Value Head PG, DCFR+ bootstrapping |
| `src/training/train.py` | softmax-policy, --clean-buffers, LR warmup, TensorBoard τ |

---

## Запуск с нуля

```bash
python -m src.training.train --self-play-multi --iterations 6000 --traversals 200 --clean-buffers --save-dir models/multi_v2 --log-dir logs/deepcfr_v2
```
