# Bug #64: One-step lookahead — контрфактические sizing Q-target'ы для несыгранных размеров

**Severity:** HIGH

**Дата:** 2026-06-04

**Связан с:** Bug #61 (sizing collapse), Bug #62 (q_network_gap), Bug #63 (probe затухает)

---

## Симптомы

Диагностика `q_compare` показала, что probe (#63) частично помогает поисковому покрытию, но **не решает корневую проблему**: sizing Q-поверхность остаётся монотонно убывающей по размерам, потому что:

1. Q-сэмпл пишется **только для одного** сыгранного размера на raise-узел (`cfr_traverse_multi`:1828)
2. Несыгранные размеры не получают свежих target'ов
3. Их Q застывает заниженным — «холодный старт» не лечится только exploration'ом

Probe даёт real rollout samples, но их плотность недостаточна для 15-анкорной сетки.

---

## Root Cause

Текущий код в raise-legal узле пишет sizing Q-sample ТОЛЬКО для одного эффективного размера:

```python
# deep_cfr.py:1823-1829 (vanilla traverse)
eff_idx = int(np.argmin(np.abs(self.anchors_arr - eff_mult)))
self._add_sizing_q_sample(..., is_probe=is_probe, anchor_idx=eff_idx)
```

Остальные 14 анкоров не получают target → Q застывает → heat_weighting сдвигает всю массу на единственный «тёплый» анкор (обычно 0.10).

---

## Решение: One-step lookahead (#64)

**Идея:** для каждого несыгранного анкора сделать **один шаг вперёд** (`apply_action` + bootstrap через `q_net`) вместо полного rollout'а. Стоимость: 15 × (один шаг), а не 15 × (дерево до showdown).

### Что сделано

#### 1. Новые конфиг-параметры

```yaml
# config.yaml
sizing_lookahead_enabled: true   # вкл/выкл
sizing_lookahead_prob: 0.10      # вероятность на raise-узел
sizing_lookahead_max_anchors: 5  # макс. анкоров за узел (случайная подвыборка)
```

Соответствующие default'ы в `config.py:_DEFAULTS`.

#### 2. `_bootstrap_sizing_value(state, traversing_player)` — новый метод

Оценивает value состояния через `q_net` (action-value), а не `sizing_q_net` (размер-value):

```python
V(s_next) = sum(strategy[a] * q_net(s_next)[a])
```

- `strategy` — regret matching с перспективы `current_player` (через `advantage_net`)
- `q_net` — с перспективы `traversing_player`
- Соответствует логике TD-target в `train_q_network`

#### 3. `_perform_sizing_lookahead(...)` — новый метод

Алгоритм:
1. `max_anchors <= 0` → выход
2. `lookahead_enabled? q_enabled? q_net есть?` → иначе выход
3. `random >= prob` → выход
4. `Raise not in legal_actions` → выход
5. `candidate_indices = все анкоры без eff_idx`
6. `random.sample(candidate_indices, min(max_anchors, len))`
7. Для каждого анкора:
   - `action_type_to_pokers_action(3, state, size_k)`
   - `apply_action` → проверка `status == Ok`
   - terminal → `reward`, иначе → `_bootstrap_sizing_value`
   - `_add_sizing_q_sample(..., is_lookahead=True, anchor_idx=anchor_idx)` → **source=2**

#### 4. `_add_sizing_q_sample` — расширен source-кодированием

```python
source=2 if is_lookahead else (1 if is_probe else 0)
```

| source | Значение |
|--------|----------|
| 0 | Обычный real rollout |
| 1 | Probe (random exploration) |
| 2 | One-step lookahead (bootstrap) |

#### 5. Интеграция в `cfr_traverse_multi`

После существующего `_add_sizing_q_sample` для сыгранного размера (строка ~1829) добавлен вызов:

```python
if self.sizing_q_enabled and self.sizing_q_net is not None and 3 in legal_action_types:
    _lookahead_eff_idx = eff_idx if sampled_bet_size is not None else -1
    self._perform_sizing_lookahead(
        state, _state_arr, iteration, traversing_player, ev,
        eff_idx=_lookahead_eff_idx,
    )
```

---

## Почему это решает проблему

1. **Каждый raise-узел с вероятностью 10%** пишет Q-target'ы для 5 случайных анкоров
2. Свежие target'ы поступают для **всех** размеров, а не только для популярных
3. `sizing_q_net` получает плотное покрытие по размерной сетке без взрыва вычислительной стоимости
4. Поверхность Q перестаёт зависеть от того, какие размеры реально играются

---

## Файлы, затронутые правками

| Файл | Изменение |
|------|----------|
| `config.yaml` | +3 параметра: `sizing_lookahead_*` |
| `src/utils/config.py` | +3 default'а в `_DEFAULTS` |
| `src/core/deep_cfr.py` | +2 метода: `_bootstrap_sizing_value`, `_perform_sizing_lookahead`; +3 строки в `__init__`; правка `_add_sizing_q_sample` (is_lookahead); вызов в `cfr_traverse_multi` |

---

## Не тронуто

- Probe — оставлен как дополнительный источник real rollout samples
- Outcome sampling-ветка (`_cfr_traverse_multi_outcome_node`) — lookahead пока не добавлен туда
- `train_sizing_q_network` — работает как раньше, получает больше разнообразных сэмплов

---

## Диагностика после внедрения

Проверить на checkpoint 100/300:
1. `full_report` — `sizing_q_stats`, `best_sizing_anchor_dist`, `raise_freq`
2. Долю source=2 в `sizing_q_buffer` (через `anchor_counts`)
3. `q_compare` — форма Q-поверхности по размерам (должна перестать быть строго убывающей)
4. `anchor_raw_probs` — распределение массы по анкорам (должно расшириться на крупные размеры)
