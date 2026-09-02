# Bug #45 — Sizing policy: action-level regret → CSA + нормализация

## Статус: FIXED (3 итерации фикса)

## Суть бага

Во всех местах CFR traversal в `pg_memory.add()` хранился `cf_regrets[3]` — regret действия Raise относительно текущей стратегии:

```
cf_regrets[3] = action_values[Raise] - EV(стратегии)
```

Этот сигнал **одинаков для любого размера ставки** — он отвечает на "рейзить ли вообще?", а не "какой размер лучше?".

Следствия:
- `sizing_head` не получает градиента для дифференциации sizing по состояниям
- `ZMeanStd ≈ 0.0005` при диапазоне [-3, 3] — sizing_head выдаёт почти константное z_mean
- Policy обучается к одному универсальному sizing (min или mid), не адаптируясь к контексту

---

## Фикс A: хранить `action_values[3]` вместо `cf_regrets[3]`

Семантика `regret_tensors` в `pg_memory` меняется:

```
БЫЛО:  pg_memory = cf_regrets[3] = V(Raise) - EV(стратегии)
СТАЛО: pg_memory = V(Raise при конкретном sampled_bet_size)  [log1p-scaled]
```

`train_sizing_network()` **не меняется** — логика та же:

```python
advantages = regret_tensors - values.squeeze(1).detach()
```

Теперь `value_head` учится предсказывать `E[action_values[3]]` (среднее V(Raise) по всем размерам из политики), а `advantages` становится настоящим conditional size advantage:

```
advantages = V(Raise_this_size) - E[V(Raise_all_sizes)]
           = "насколько ЭТА ставка лучше типичной ставки"
```

### Изменения в коде (фикс A)

**`src/core/deep_cfr.py` — 3 места:**

**1. `cfr_traverse()` — vanilla CFR traversal:**

```python
# БЫЛО:
regret_raise = action_values[3] - ev
scaled_regret_raise = np.sign(regret_raise) * np.log1p(abs(regret_raise))
self.pg_memory.add(_state_arr, z_raw_for_pg, scaled_regret_raise, ...)

# СТАЛО:
scaled_av3 = np.sign(action_values[3]) * np.log1p(abs(action_values[3]))
self.pg_memory.add(_state_arr, z_raw_for_pg, scaled_av3, ...)
```

**2. `_cfr_traverse_multi_outcome_node()` — outcome sampling:**

В outcome sampling `action_values[3]` недоступен (не все действия traversаются). Используем `value_hat[3]` — Q-value оценку с importance correction.

```python
# БЫЛО:
sizing_advantage = float(cf_regrets[3])
sizing_advantage = max(-clip, min(clip, sizing_advantage))
self.os_sizing_advantage_sum += sizing_advantage
self.os_sizing_advantage_sq_sum += sizing_advantage ** 2
self.os_sizing_advantage_count += 1
self.pg_memory.add(_state_arr, z_raw_for_pg, sizing_advantage, ...)

# СТАЛО:
scaled_v3 = np.sign(value_hat[3]) * np.log1p(abs(value_hat[3]))
self.pg_memory.add(_state_arr, z_raw_for_pg, scaled_v3, ...)
```

**3. `cfr_traverse_multi()` — multi-player traversal:**

```python
# БЫЛО:
self.pg_memory.add(_state_arr, z_raw_for_pg, cf_regrets[3], ...)

# СТАЛО:
scaled_av3 = np.sign(action_values[3]) * np.log1p(abs(action_values[3]))
self.pg_memory.add(_state_arr, z_raw_for_pg, scaled_av3, ...)
```

**`src/training/train.py` — 1 место:**

**4. `_cfr_traverse_with_opponents()`:**

```python
# БЫЛО:
regret_raise = action_values[3] - ev
scaled_regret_raise = np.sign(regret_raise) * np.log1p(abs(regret_raise))
agent.pg_memory.add(..., scaled_regret_raise, ...)

# СТАЛО:
scaled_av3 = np.sign(action_values[3]) * np.log1p(abs(action_values[3]))
agent.pg_memory.add(..., scaled_av3, ...)
```

### Удалённый мёртвый код (фикс A)

| Файл | Что удалено |
|---|---|
| `deep_cfr.py` | `self.hybrid_os_sizing_advantage_clip` — поле объекта |
| `deep_cfr.py` | `os_sizing_advantage_sum/sq_sum/count` — поля в `reset_traversal_stats()` |
| `deep_cfr.py` | `os_sizing_advantage_*` в `get_traversal_stats()` |
| `train.py` | Print-блок `if os_stats['os_sizing_advantage_count'] > 0` |
| `config.py` | `hybrid_os_sizing_advantage_clip: 10.0` |

---

## Фикс B: Clamp ПОСЛЕ нормализации (а не ДО)

### Проблема обнаружена после fresh run

С CSA-сигналом `regret_tensors ∈ [-4, +4]` вместо ≈ 0, clamp ДО нормализации создавал вырожденную ситуацию:

1. `value_head` временно переоценивает средний EV → `advantages` систематически отрицательные
2. `torch.clamp(advantages, -3.0, 3.0)` → большинство samples = `-3.0` (заперты на клипе)
3. `advantages.std() ≈ 0` (все в одной точке) → нормализация через `/ (std + 1e-6)` даёт `policy_advantages ≈ ±1000`
4. Экстремальный сигнал → `z_mean` улетает в границу tanh → collapse

**Симптом:** `ZMeanStd: 1.33 → 0.013` за одну итерацию около iter ~300.

### Фикс

```python
# БЫЛО:
advantages = regret_tensors - values.squeeze(1).detach()
advantages = torch.clamp(advantages, -3.0, 3.0)        # clamp ДО нормализации

policy_advantages = advantages - advantages.mean()
policy_advantages = policy_advantages / (policy_advantages.std(unbiased=False) + 1e-6)
# ← нет защиты от выбросов

# СТАЛО:
advantages = regret_tensors - values.squeeze(1).detach()
# без clamp — std остаётся осмысленным

policy_advantages = advantages - advantages.mean()
policy_advantages = policy_advantages / (policy_advantages.std(unbiased=False) + 1e-6)
policy_advantages = torch.clamp(policy_advantages, -3.0, 3.0)  # clamp ПОСЛЕ нормализации
```

**Почему правильно:**
- Advantages сохраняют полный диапазон → `std` всегда осмысленный → нормализация не вырождается
- Clamp ±3.0 после нормализации отсекает только настоящие выбросы (>3σ), а не систематически запертые значения
- `value_loss` по-прежнему использует `regret_tensors.detach()` — без изменений

### Дополнительно: Temperature 1.0 → 2.0

С CSA `regret_tensors ∈ [-4, +4]` при `temperature=1.0`: `sigmoid(±4) = 0.018/0.982` — почти бинарные веса, effective_batch маленький.

При `temperature=2.0`: `sigmoid(±4/2) = 0.12/0.88` — мягче, больше обучающих samples.

```yaml
# config.yaml
pg_sizing_weight_temperature: 2.0  # было 1.0
```

---

## Полная сводка изменений

| Файл | Изменение | Фикс |
|---|---|---|
| `deep_cfr.py` | `cfr_traverse()`: `cf_regrets[3]` → `scaled_av3` | A |
| `deep_cfr.py` | `_cfr_traverse_multi_outcome_node()`: `cf_regrets[3]` → `scaled_v3` | A |
| `deep_cfr.py` | `cfr_traverse_multi()`: `cf_regrets[3]` → `scaled_av3` | A |
| `deep_cfr.py` | Удалён `hybrid_os_sizing_advantage_clip` + `os_sizing_advantage_*` | A |
| `train.py` | `_cfr_traverse_with_opponents()`: `cf_regrets[3]` → `scaled_av3` | A |
| `train.py` | Удалён print-блок `os_sizing_advantage_count` | A |
| `config.py` | Удалён `hybrid_os_sizing_advantage_clip: 10.0` | A |
| `deep_cfr.py` | Clamp перенесён: ДО → ПОСЛЕ нормализации `policy_advantages` | B |
| `config.yaml` | `pg_sizing_weight_temperature`: 1.0 → 2.0 | B |
| `inference/core.py` | `forward()`: z_mean bounded через tanh, log_std через sigmoid | C |

### Что НЕ меняется

- `model.py` — `detach()` оставлен
- `train_sizing_network()` — основная логика без изменений
- `PolicyGradientMemory` — API без изменений
- Return tuple (14 значений) — без изменений
- `advantage_buffer`, CFR-регреты — не тронуты
- Soft weighted PG (`raise_weight`) — остаётся

---

## Фикс C: Inference `forward()` не соответствовал training

### Проблема

`inference/core.py` использовал **старую** параметризацию z_mean и log_std:

```python
# inference/core.py — БЫЛО:
z_mean = sizing_params[:, 0:1]                      # не bounded
log_std = torch.clamp(sizing_params[:, 1:2], -2.0, 0.0)  # старый clamp

# src/core/model.py — ПРАВИЛЬНО (с фикса #43):
raw_z_mean = sizing_params[:, 0:1]
z_mean = 3.0 * torch.tanh(raw_z_mean / 3.0)          # bounded

raw_log_std = sizing_params[:, 1:2]
log_std = -2.0 + 2.0 * torch.sigmoid(raw_log_std)    # sigmoid-map
```

При загрузке чекпоинта веса sizing_head обучались с bounded z_mean и sigmoid log_std, а инференс применял raw значения → sizing был неверным.

### Фикс

`inference/core.py`, метод `forward()`:

```python
# СТАЛО:
raw_z_mean = sizing_params[:, 0:1]
z_mean = 3.0 * torch.tanh(raw_z_mean / 3.0)

raw_log_std = sizing_params[:, 1:2]
log_std = -2.0 + 2.0 * torch.sigmoid(raw_log_std)
```

Теперь training и inference используют идентичную параметризацию.

---

## Переходный период

**Первые 50–150 итераций** с чекпоинтом — `value_head` перестраивается с предсказания `cf_regrets ≈ 0` на предсказание `action_values[3]` (другой масштаб). В этот период `advantages` шумные → sizing дёргается. Нормально, `value_lr = 1e-3` адаптируется быстро.

Фикс B предотвращает collapse на iter ~300, но не устраняет переходный шум — это ожидаемо.

## Критерии что CSA заработал (200+ итераций)

| Метрика | Было | Ожидание |
|---|---|---|
| `Sizing/ZMeanStd` | ~0.0005 | **>0.05** — главный индикатор: sizing дифференцирует состояния |
| `PG/RawRewardMean` | ~0.0 | Ненулевое, ±1–4 после log1p |
| `PG/FracPositiveAdvantage` | хаотично 0.28–0.95 | Стабилизируется ~0.50 |
| `Sizing/Mean_Predicted_Size` | осцилляция/коллапс | Монотонный рост или стабильный диапазон |
| `PG/SizingEffectiveBatchSize` | ~35-50 | >25 с temperature=2.0 |
| Win rate vs random | положительный | Не падает |

## Связанные баги

- **Bug #43** (FIXED): log_std deadlock + z_mean runaway
- **Bug #44** (SUPERSEDED): soft weighted PG — остаётся как стабилизатор, но CSA делает главный вклад
- **Bug #45** (этот): корневой фикс reward-design + нормализация

## Консенсус

- Opus предложил CSA как элегантную альтернативу multi-sample — нулевой overhead, меняется только семантика pg_memory
- GPT: диагностировал collapse от clamp ДО нормализации, предложил перенос clamp + temperature=2.0
- GLM: (1) `detach()` оставить, (2) место 2 использует `value_hat[3]` вместо `action_values[3]`, (3) согласен с переносом clamp, (4) обнаружил рассинхрон inference/core.py
- Все согласны: `ZMeanStd > 0.05` после 200 итераций = sizing_head дифференцирует состояния
