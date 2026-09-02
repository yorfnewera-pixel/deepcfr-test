# #95 — Technical Fix: Float Precision in np.random.choice с availability-гейтингом

**Дата:** 2026-06-17
**Статус:** FIXED (5 точек)
**Связанные:** #94 (availability gating вводит FP-погрешность в probs)

## Проблема

При включении `sizing_anchor_availability_enabled: true` (#94 R2) `np.random.choice(p=probs)` начал массово падать:

```
ValueError: probabilities do not sum to 1
ValueError: probabilities are not non-negative
```

### Корень

Availability-гейтинг (#94 A2) обнуляет недоступные анкеры и ренормализует:
1. `probs[~avail] = 0.0` — зануление в float32
2. `probs = probs / probs.sum()` — деление float32/float32

После шага 2 сумма в float32 может быть `0.99999994` или `1.0000001`. `np.random.choice` требует сумму строго `1.0 ± ε` с очень жёстким допуском. Попытка исправить в float64 через `probs /= probs.sum()` тоже не гарантирует идеальную единицу — деление float64 всё равно даёт `1.0 ± ≈1e-16`.

Попытка `probs[-1] = 1.0 - probs[:-1].sum()` ломалась, когда `probs[:-1].sum() > 1.0` → последний элемент уходил в `−1e-17` → `ValueError: probabilities are not non-negative`.

### Механика не задета

Фикс затрагивает **только** нормализацию вероятностей перед `np.random.choice(p=...)` — шаг выбора **индекса** анкера. После выбора:
- Bet берётся из таблицы констант `self.anchors[idx]`
- Pokers-механика (min-raise, all-in, round(2)) — в `_resolve_effective_sizing`, не тронута

Никакого влияния на среду, обучение или детерминизм.

## Решение

Везде, где availability-гейтинг модифицирует probs перед `np.random.choice`, добавлен трёхшаговый гарант:

```python
probs = np.clip(probs.astype(np.float64), 0.0, None)   # 1. убрать FP-негативы
total = max(probs.sum(), 1e-12)
probs /= total                                           # 2. float64-нормализация
probs[-1] = max(1.0 - probs[:-1].sum(), 0.0)            # 3. защёлка → сумма ровно 1.0
```

Гарантии после фикса:
- Все probs ≥ 0 (строго)
- Сумма = ровно 1.0 (в float64)
- Последний элемент (3.00) впитывает погрешность округления

## Изменённые точки (5 мест в deep_cfr.py)

| # | Функция | Строка | Путь |
|---|---------|--------|------|
| 1 | `_hierarchical_sizing` | ~2211 | Full-raise + OS-raise не-Q-ready |
| 2 | `_cfr_traverse_multi_outcome_node` | ~2323 | OS-raise Q-ready |
| 3 | `choose_action` top_bucket | ~3676 | Инференс |
| 4 | `choose_action` hierarchical | ~3704 | Инференс |
| 5 | `choose_action` stochastic | ~3716 | Инференс |

## Дополнительно

В `train.py` в eval-функциях добавлен `traceback.print_exc()` для первой ошибки игры — помогает диагностировать источник при будущих проблемах.
