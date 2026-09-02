# Bug Report — Sizing Architecture (сводный)

**Дата**: 2026-06-01
**Статус**: Bug #50 (Fixed Grid) — P0/P1 исправлены 2026-06-01. P3 (AWR config) — deprecated, оставлен для совместимости старых чекпоинтов.

---

## 1. Что сделано правильно (Bug #50)

### Модели (model.py) — чисто

| Класс | Статус |
|-------|--------|
| `SizingAnchorNet` (стр. 79) | `anchor_head → Tensor[B, K]`, без position_head |
| `StrategySizingNet` (стр. 45) | `(slot_logits[B,K], scalar_bet[B,1])`, без position_head |
| `SizingQNetwork` (стр. 218) | Q(state, size) — не менялся |

Удалено: `position_head`, `floating_positions_to_sizes`, `sizing_normalize_size`, `unpack_sizing_output`.

### Основной CFR traversal (deep_cfr.py:1440-1550) — чисто

- Q-ready guard перед Q-heat весами
- Fallback = uniform по фиксированной сетке
- `_apply_sizing_bucket_floor` применяется
- Stochastic sample из `np.random.choice(fixed_grid, p=weights)`
- Запись в `sizing_strategy_buffer` только при `_sizing_q_is_ready()`

### OS traversal (deep_cfr.py:1280-1400) — чисто

- Q-ready → Q-heat, иначе uniform
- Bucket floor НЕ применяется (OS — lightweight)
- Запись в `sizing_strategy_buffer` только при `_sizing_q_is_ready()`

### Обучение

| Метод | Guard | Механика |
|-------|-------|----------|
| `train_sizing_anchor_network` | `Q-ready → skip` | MSE(slot_logits, Q_adv) |
| `train_strategy_sizing_anchor_network` | `Q-ready → skip` | KL-div от buffer target_weights |

### choose_action — чисто

- Использует `strategy_sizing_net`
- `deterministic=False` по умолчанию → stochastic sample

### Инференс (inference/core.py) — чисто

- `StrategySizingNet` без position_head
- Stochastic sample из фиксированной сетки

---

## 2. Проблемы

### 2.1. ~~Legacy traversal использует advantage_sizing_net напрямую~~ ✅ ИСПРАВЛЕНО

**Файл**: `src/training/train.py`, функция `_cfr_traverse_with_opponents`

**Было** (advantage_sizing_net → softmax, веса попадали в sizing_strategy_buffer):
```python
slot_logits = agent.advantage_sizing_net(state_tensor.unsqueeze(0))
probs = F.softmax(slot_logits, dim=1)[0].cpu().numpy()
sampled_bet_size = float(np.random.choice(agent.anchors, p=probs))
slot_weights_np = probs.copy()
```

**Стало** (Q-guided, как в основном CFR traversal):
```python
slot_sizes_np = np.array(agent.anchors, dtype=np.float32)
if agent._sizing_q_is_ready():
    q_vals = agent._evaluate_sizing_q_for_sizes(state_tensor, slot_sizes_np)
    slot_weights_np = compute_sizing_heat_weights(q_vals, ...)
else:
    slot_weights_np = np.ones(agent.num_anchors) / agent.num_anchors
slot_weights_np = agent._apply_sizing_bucket_floor(slot_weights_np, iteration)
sampled_bet_size = float(np.random.choice(slot_sizes_np, p=slot_weights_np))
```

### 2.2. ~~Мёртвый AWR/floating код~~ ✅ УДАЛЁН

Из `deep_cfr.py` удалены (~70 строк):
- `_build_sizing_candidate_matrix` — собирал anchors + policy_mean + samples через `_squash()`
- `_compute_awr_weights` — AWR веса из Q-advantages
- `_compute_awr_loss` — AWR loss, вызывал бы `AttributeError` на `SizingAnchorNet`

### 2.3. ~~`train_sizing_network` — dead code~~ ✅ УДАЛЁН

Удалён dead wrapper (5 строк), который вызывал `train_sizing_anchor_network` с несуществующим kwarg `epochs=1` и возвращал 14-элементный кортеж.

### 2.4. ~~Неиспользуемый импорт `SizingNetwork` в deep_cfr.py~~ ✅ УБРАН

Импорт `SizingNetwork` удалён из `src/core/deep_cfr.py`. Сам класс `SizingNetwork` в `model.py` **оставлен** — используется в `diagnose_sizing.py`.

### 2.5. AWR конфиг-параметры в `__init__` — НЕ ТРОНУТ (deprecated)

```python
self.sizing_candidate_sizes = cfg_get('sizing_candidate_sizes', ...)
self.sizing_anchor_sizes = cfg_get('sizing_anchor_sizes', ...)
self.sizing_policy_samples_per_state = ... 
self.sizing_awr_temperature = ...
self.sizing_awr_uniform_mix = ...
self.sizing_reinforce_weight = ...
```

Оставлены как deprecated: удаление может сломать загрузку старых чекпоинтов (поля сохраняются в `save_checkpoint`). Не используются в логике.

---

## 3. Сводная таблица: источники sizing-весов (актуальное)

| Путь | Источник sizing | Q-guided? | Пишет в sizing_strategy_buffer? |
|------|-----------------|-----------|--------------------------------|
| CFR traversal (deep_cfr:1440) | Q-heat или uniform | Да (Q-ready) | Только Q-ready |
| OS traversal (deep_cfr:1280) | Q-heat или uniform | Да (Q-ready) | Только Q-ready |
| **Legacy traversal (train.py:209)** | **~~advantage_sizing_net~~ → Q-heat или uniform** ✅ | **Да (Q-ready)** | **Только Q-ready** ✅ |
| choose_action (deep_cfr) | strategy_sizing_net → softmax | Нет (distilled) | Нет |
| inference (inference/core.py) | strategy_sizing_net → softmax | Нет (distilled) | Нет |

**Все три traversal-пути теперь консистентны**: Q-ready → Q-heat → bucket_floor → stochastic sample. Fallback = uniform.

---

## 4. Что было исправлено (2026-06-01)

| Приоритет | Правка | Файл | Строки |
|-----------|--------|------|--------|
| **P0** | Legacy traversal: `advantage_sizing_net` → Q-guided + bucket_floor | `train.py` | 209-229 |
| **P1** | Удалён мёртвый AWR код (3 метода) | `deep_cfr.py` | ~70 строк |
| **P1** | Удалён dead `train_sizing_network` | `deep_cfr.py` | 5 строк |
| **P2** | Убран неиспользуемый импорт `SizingNetwork` | `deep_cfr.py` | стр. 12 |

### Добавлены тесты

| Тест | Файл | Что проверяет |
|------|------|--------------|
| `test_sizing_q_not_ready_fallback_is_uniform_not_advantage_sizing_net` | `test_sizing_q_regret.py` | Q не ready → uniform fallback, не advantage_sizing_net |
| `test_advantage_sizing_net_is_not_floating_sizing_network` | `test_sizing_q_regret.py` | `SizingAnchorNet` не имеет floating-методов (`_squash`, etc.) |

---

## 5. Что остаётся валидным из примечаний

### Реализовано:
- [x] Cold-start: Q не ready → uniform/probe, sizing сети не обучаются
- [x] Fallback self-distillation убран
- [x] `_sizing_q_is_ready`: `buffer >= 5000 AND anchor_coverage`
- [x] `train_strategy_sizing_anchor_network` — только Q-ready
- [x] `pos_mse` удалён (floating полностью)
- [x] `deterministic=False` по умолчанию
- [x] Q-heat → target_weights → strategy_sizing_net → stochastic sample
- [x] **Legacy traversal**: Q-guided + bucket_floor, консистентно с основным CFR ✅
- [x] Мёртвый AWR код удалён ✅
- [x] Dead `train_sizing_network` удалён ✅
- [x] Неиспользуемый импорт `SizingNetwork` убран ✅

### Оставлено как deprecated:
- [ ] AWR конфиг в `__init__` — нужен для загрузки старых чекпоинтов

### Архитектурно корректно:
- Sizing = Q-guided stochastic sizing policy, НЕ full CFR over sizes
- CFR над Fold/Check/Call/Raise, sizing отдельно
- Q = главный источник сигнала
- advantage_sizing_net → candidate logits, обучается от Q
- strategy_sizing_net → финальная policy, обучается от Q-derived weights

---

## 6. Карта файлов (актуальное)

| Файл | Статус |
|------|--------|
| `src/core/model.py` | Чисто. `SizingNetwork` оставлен для `diagnose_sizing.py` |
| `src/core/deep_cfr.py` | Чисто. AWR dead code удалён, `SizingNetwork` импорт убран, `train_sizing_network` удалён |
| `src/training/train.py` | Чисто. Legacy traversal → Q-guided ✅ |
| `inference/core.py` | Чисто |
| `config.yaml` | Чисто |
| `tests/test_anchor_sizing.py` | 7 тестов ✅ |
| `tests/test_sizing_q_regret.py` | 8 тестов (было 6) ✅ |
| `diagnose_sizing.py` | OK, `_fixed_grid_forward`, использует `SizingNetwork` |
