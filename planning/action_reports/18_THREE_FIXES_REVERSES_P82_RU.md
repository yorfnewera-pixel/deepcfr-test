# Баг-репорт #18: Два исправления Deep CFR + подтверждение P8-2 (epsilon-greedy, нормализация таргетов)

**Проект**: deepcfr-test  
**Дата**: 2026-05-07  
**Серьёзность**: Medium (качество обучения)  
**Статус**: Fixed  

---

## Источник

5 дискуссий с Gemini 2.5 Flash Thinking через CDP-браузер (порт 9222):

| # | Тема | Итог |
|---|------|------|
| 1 | Bootstrap + сходимость DCFR | Не гарантирует сходимость, но и не нарушает |
| 2 | PG для sizing_head | PG нужен (мультимодальность + sampling bias + advantage-weighted) |
| 3 | PER vs Reservoir | PER оправдан с IS-weights. PER bias ≠ catastrophic interference |
| 4 | strategy_net | Нужна! Без неё — осцилляции и эксплуатируемость |
| 5 | clamp ДО vs ПОСЛЕ суммы | **ПОДТВЕРЖДЕНО: clamp ДО суммы (P8-2 верен)** — статья DCFR+ явно пишет `max(R^{t-1},0)*d+r^t` |

---

## Исправление #1: ОТМЕНЕНО — P8-2 (clamp ДО суммы) подтверждён повторно

**Приоритет**: ~~HIGH~~ — ОТМЕНЕНО  
**Статус**: P8-2 из баг-репорта #17 остаётся верным  

### История конфликта

1. **Баг-репорт #17 (P8-2)**: 3 AI (Claude, Haiku, Gemini) → clamp ДО суммы
2. **Данная сессия, дискуссия #1**: Gemini рекомендовал clamp ПОСЛЕ суммы → реверс P8-2
3. **Данная сессия, дискуссия #5**: Gemini **отозвал рекомендацию**, подтвердил статью

### Почему реверс был ошибкой

Gemini признал что рекомендация clamp ПОСЛЕ суммы была "инженерной безопасностью" для нейросетей, но **нарушает теоретические гарантии DCFR+**:

1. **Статья VR-DeepPDCFR+ Algorithm 1 ЯВНО**: `max(R^{t-1}, 0) * d^t + r^t` — Вариант 2 (clamp ДО суммы)
2. **Вариант 3 (clamp ПОСЛЕ суммы) = CFR+**, не DCFR+ — это **другой алгоритм** с другими гарантиями
3. **Bias при target ≥ 0**: Если сеть предсказывает только ≥ 0, шум аппроксимации не может уйти в минус → среднее предсказание для плохих действий всегда > 0 → Regret Matching даёт им слишком большой вес
4. **Потеря информации о глубине отрицательного регрета**: Вариант 3 "схлопывает" всю отрицательную область в 0

### Итоговый код (P8-2 подтверждён)

```python
# DCFR+ (Вариант 2 — статья) — ПРАВИЛЬНО
bootstrap_target = torch.clamp(prev_pred * discount, min=0) + regret_tensors
```

**Три рекурсии для справки:**

| Алгоритм | Рекурсия | R может быть < 0? |
|----------|----------|-------------------|
| DCFR | R^t = R^{t-1}*d + r^t | Да |
| **DCFR+ (наш)** | R^t = max(R^{t-1}, 0)*d + r^t | Да, от r^t |
| CFR+ | Q^t = max(Q^{t-1}*d + r^t, 0) | Нет |

---

## Исправление #2: Epsilon-greedy в cfr_traverse

**Приоритет**: MEDIUM  
**Файлы**: `src/core/deep_cfr.py` — `cfr_traverse` (~строка 308), `cfr_traverse_multi` (~строка 463)  
**Источник**: Дискуссия с Gemini (тема #3: PER vs Reservoir — рекомендация улучшить исследование)  

### Проблема

Стратегия в traverse — softmax по advantages с температурой. Без epsilon-greedy агент застревает в подоптимальных ветках дерева, особенно на ранних итерациях когда advantages шумные.

### Исправление

```python
# После вычисления strategy через softmax:
epsilon = max(0.05, 0.3 * (1 - iteration / 1000))
if np.random.random() < epsilon:
    strategy = np.zeros(self.num_actions)
    for a in legal_action_types:
        strategy[a] = 1.0 / len(legal_action_types)
```

### Параметры

| Параметр | Значение | Обоснование |
|----------|----------|-------------|
| epsilon_start | 0.3 | Агрессивное исследование на ранних итерациях |
| epsilon_min | 0.05 | Минимальное исследование — предотвращает локальные оптимумы |
| decay_iter | 1000 | Линейное затухание; к итерации 1000 стратегия определяется преимуществами |

### Взаимодействие с τ-decay (P8-3)

| Итерация | τ (температура) | epsilon | Эффективная стратегия |
|----------|-----------------|---------|----------------------|
| 1 | 5.0 | 0.30 | Почти uniform (τ + ε) |
| 100 | 4.1 | 0.27 | Ещё очень случайная |
| 500 | 0.5 | 0.15 | Преимущества начинают доминировать |
| 1000 | 0.5 | 0.05 | Почти чистые преимущества, ε ≈ 0 |

### Риск

- Увеличивает дисперсию traversals на ранних итерациях
- Может замедлить сходимость если epsilon слишком высок
- Линейное затухание — консервативный выбор, можно переключить на cosine

---

## Исправление #3: Нормализация таргетов advantage (running mean/std)

**Приоритет**: MEDIUM  
**Файлы**: `src/core/deep_cfr.py` — `__init__` (~строка 148), `train_advantage_network` (~строка 630), `train_advantage_network_multi` (~строка 695)  
**Источник**: Дискуссия с Gemini (тема #1: Bootstrap сходимость — рекомендация стабилизировать target)  

### Проблема

`bootstrap_target` имеет большой разброс, особенно на ранних итерациях когда discount близок к 0 и prev_pred ещё не обучен. Нестабильный градиент, скачущий loss.

### Исправление

**В `__init__`:**

```python
self.target_mean = 0.0
self.target_var = 1.0
self.target_momentum = 0.99
```

**В `train_advantage_network` / `train_advantage_network_multi`, после вычисления bootstrap_target:**

```python
with torch.no_grad():
    batch_mean = bootstrap_target.mean().item()
    batch_var = bootstrap_target.var().item()
    self.target_mean = self.target_momentum * self.target_mean + (1 - self.target_momentum) * batch_mean
    self.target_var = self.target_momentum * self.target_var + (1 - self.target_momentum) * max(batch_var, 1e-8)

normalized_target = (bootstrap_target - self.target_mean) / (self.target_var ** 0.5 + 1e-8)

action_loss = (F.smooth_l1_loss(predicted_regrets, normalized_target, reduction='none', beta=5.0) * weight_tensors).mean()
```

### Механизм

1. **Running statistics**: EMA с momentum=0.99 обновляет mean/var каждый батч
2. **Нормализация**: target центрируется и масштабируется → стабильный градиент
3. **max(batch_var, 1e-8)**: Защита от деления на 0 на первых батчах
4. **Вместе с IS-weights**: Нормализация не заменяет importance sampling — они ортогональны

### Взаимодействие с clamp ДО суммы (P8-2 подтверждён)

При clamp ДО суммы: `target ∈ (-∞, +∞)` — mean ≈ 0, var больше.
Нормализация помогает стабилизировать target, который может быть отрицательным.

---

## Итоговая матрица изменений

| Файл | Изменение | Статус |
|------|-----------|--------|
| `src/core/deep_cfr.py:609` | `clamp(prev*discount, min=0) + regret` | P8-2 подтверждён, без изменений |
| `src/core/deep_cfr.py:666` | `clamp(prev*discount, min=0) + regret` | P8-2 подтверждён, без изменений |
| `src/core/deep_cfr.py:308` | epsilon-greedy в cfr_traverse | Новое |
| `src/core/deep_cfr.py:463` | epsilon-greedy в cfr_traverse_multi | Новое |
| `src/core/deep_cfr.py:148` | target_mean, target_var, target_momentum | Новое |
| `src/core/deep_cfr.py:630` | Нормализация target в train_advantage_network | Новое |
| `src/core/deep_cfr.py:695` | Нормализация target в train_advantage_network_multi | Новое |

---

## Дискуссии с Gemini — ключевые инсайты (не вошли в код)

### PER bias ≠ Catastrophic Interference

PER bias (решается IS-weights, уже реализовано) — статистическая проблема смещённого семплирования.  
Catastrophic Interference (решается one-hot position) — архитектурная проблема shared weights.  
PER **усиливает** interference через feature representation shift, но это ДВЕ РАЗНЫЕ проблемы.

Если префлоп "плывёт" → проблема в interference, не в PER bias. Лечение: widening сети или stage-specific heads.

### Strategy_net НЕ overengineering

Удаление strategy_net превратит агента из "ищущего Нэша" в "эксплуатирующего и сам легко эксплуатируемого". Средняя стратегия = единственное что сходится к равновесию. Текущая стратегия осциллирует (Камень-Ножницы-Бумага: 100%К→100%Б→100%Н vs. равновесие 1/3,1/3,1/3).

---

## Валидация

- `python -c "from src.core.deep_cfr import DeepCFRAgent; print('OK')"` — Import OK
- `pytest tests/test_training_regressions.py -v` — 3/3 passed
- AST parse — OK

### Что мониторить при запуске 6000 итераций

1. **advantage_loss**: Не должен скакать > 10x между эпохами (нормализация должна помочь)
2. **Zero-Collapse**: Доля батчей где `bootstrap_target.sum() == 0` — если > 20% → реверс P8-2
3. **Epsilon schedule**: После итерации 1000 epsilon = 0.05, стратегия должна стабилизироваться
4. **target_mean/target_var**: Должны стабилизироваться после ~50 итераций
5. **Префлоп стратегия**: UTG open-raise % не должен "плыть" — признак catastrophic interference
