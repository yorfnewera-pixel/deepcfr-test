# #87b — Pot+Stack Normalization (одна переменная vs #87)

> **Статус:** PRE-REGISTERED — код исправлен (баг на iter 50: opponent-путь не нормализовался), ожидает перезапуска.
> **⚠ Кавеат:** прогон на 400 traversals (исправлено с 1000). Возврат к baseline #84.
> **Баг-фикс:** `replaceAll` пропустил opponent-путь из-за разного indentation. Добавлен `elif pot_stack_relative` в строку 2530.
> **База:** #87 (q_reward_propagation=mc, q_target_norm=pot_relative). Одна переменная: `q_target_norm: pot_relative → pot_stack_relative`.
> **Связанные:** #87 (FAIL-M0, хвосты из префлоп-записей), #86 (MC propagation PARTIAL), #84b (relapse mechanism)

---

## 0. Pre-registered критерии (зафиксированы ДО запуска)

### Мотивировка

#87 FAIL-M0: pot-relative нормализация (`reward /= pot`) улучшила масштаб Q (target_term 0.7-1.4, q_loss 28), но НЕ убрала хвосты. Причина: на префлопе pot=2-3, а v_sampled может быть ±hundreds chips → reward сотни «потов» (raise std=28.5, max=+787). MSE доминируют хвосты → fit_ratio застревает на 0.04-0.07.

**Решение:** знаменатель = `pot + stake_hero`. По построению |v_sampled| ≤ pot + stake — reward гарантированно ∈ [-1, +1] для всех записей.

### Дизайн

| Параметр | #87 (reference) | #87b (experiment) |
|----------|----------------|-------------------|
| `q_reward_propagation` | `mc` | `mc` |
| `q_target_norm` | `pot_relative` | **`pot_stack_relative`** |
| `q_terminal_balanced_loss_enabled` | false | false |
| `q_grad_clip_max_norm` | 100 | 100 |
| Итерации | 300 (остановлен на ~140) | **300** |
| Сиды | seed1 | seed1 |

### Реализация

В `deep_cfr.py`, в обеих точках `q_buffer.add`, вместо `reward /= pot`:

```python
if self.q_target_norm == 'pot_stack_relative':
    denom = max(float(state.pot) + float(state.players_state[traversing_player].stake), 1.0)
    reward = reward / denom
```

`stake_hero` — текущий стек игрока на момент записи. `pot + stake` — верхняя граница |v_sampled| → reward гарантированно ∈ [-1, +1]. Знаменатель — константа состояния (один для всех действий) → сравнение Q не искажается.

### Диагноз хвостов #87 (подтверждение выбора)

Из full_report #87 iter 100:

| Действие | reward mean | reward std | reward min | reward max |
|----------|------------|-----------|------------|------------|
| fold | +0.01 | 5.96 | -192 | +482 |
| check | -0.08 | 4.12 | -169 | +346 |
| call | -0.09 | 12.03 | -491 | +1082 |
| raise | -0.29 | **28.48** | -392 | +787 |

Хвосты ±400-1000 «потов» — это префлоп-записи с pot=2-3 и v_sampled ±hundreds chips.

С `pot_stack_relative`: preflop pot=3, stack=200 → denom=203. v_sampled max=200 → reward max=0.99. Хвосты убраны.

### Критерии — ТРИ УРОВНЯ

**M0 (масштаб, iter 50 — СТРОГО, ранний стоп ОБЯЗАТЕЛЕН):**
- `target_term` < 2 (ожидаем ~0.5 после pot+stack)
- `q_loss` < 100 (в #87 было 30-350 к iter 140)
- `fit_ratio > 0.5` к iter 50-100
- **Не выполнен → СТОП сразу. Без исключений (урок #87).**

**M1 (механизм):** raise reward mean ≠ 0, знакопеременный, весь диапазон ~[-1.5, +1.5]

**M2 (механизм, relapse_diff 100↔200↔300):**
- flip отсутствует (adv_fold не переходит − → +)
- |δ adv_raise| < 1.0
- **НОВОЕ:** `adv_raise max > 0` хотя бы на части состояний (в #87 был < 0 на ВСЕХ)

**B (поведение, каждый чекпоинт 100/200/300):**
- `verdict = OK`, `raise_freq ≥ 40%`, `unique_anchors = 15/15`
- `top-2 sizing mass ≤ 55%`, `анкер 0.10 ≤ 15% массы`

### Исходы

| Исход | Условие |
|-------|---------|
| **PASS** | M0 + M1 + M2 + B на всех чекпоинтах → консолидация seed2 + iter 400 |
| **PARTIAL** | M0-M2 ok, B проседает (adv_raise>0 но политика не догнала) → обсудить усиление |
| **FAIL** | M0 не выполнен к iter 50 → СТОП. Или M2 flip/adv_raise<0 везде → механизм глубже |

---

## 1. Результаты

### seed1

| iter | verdict | raise_freq | unique_anchors | top-2 mass | 0.10 mass | win_rate | mean_reward | fit_ratio |
|------|---------|------------|----------------|-----------|-----------|----------|-------------|-----------|
| 100 | ? | ? | ? | ? | ? | ? | ? | ? |
| 200 | ? | ? | ? | ? | ? | ? | ? | ? |
| 300 | ? | ? | ? | ? | ? | ? | ? | ? |

### Сравнение с #87 (reference)

| iter | #87 raise_freq | #87b raise_freq | #87 fit_ratio | #87b fit_ratio | #87 q_loss | #87b q_loss |
|------|---------------|----------------|--------------|---------------|-----------|------------|
| 100 | 21.4% | ? | 0.04-0.07 | ? | 8-63 | ? |
| 200 | — | ? | — | ? | — | ? |
| 300 | — | ? | — | ? | — | ? |

### M0 (iter 50)

| Метрика | Цель | Факт |
|---------|------|------|
| target_term | < 2 | ? |
| fit_ratio | > 0.5 | ? |
| q_loss | < 100 | ? |

### Relapse diff 100→200→300

| Метрика | iter 100 | iter 200 | iter 300 | adv_raise max>0? | Flip? |
|---------|----------|----------|----------|-----------------|-------|
| adv_fold mean | ? | ? | ? | — | ? |
| adv_raise mean | ? | ? | ? | ? | — |
| raise_freq (fixed) | ? | ? | ? | — | — |

---

## 2. Вывод: PASS / PARTIAL / FAIL

<!-- Заполнить после прогона -->

---

## 3. Next Steps

- **PASS:** консолидация seed2 + iter 400. #87b = кандидат в baseline.
- **PARTIAL:** обсудить усиление (возможно advantage-регуляризация как страховка от H3).
- **FAIL:** M0 провален → вернуться к RFC с данными. Механизм глубже нормализации.
