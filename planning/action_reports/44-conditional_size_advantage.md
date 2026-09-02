# Bug #44 — Sizing policy обучается на action-level regret, а не на conditional size advantage

## Статус: PARTIAL FIX (soft weighted PG)

## Суть бага

Sizing head получает PG-сигнал, который отвечает на вопрос:

> "Рейзить ли вообще?"

А должен получать ответ на:

> "Если рейзим, какой размер лучше?"

### Код, где формируется reward

Во всех трёх местах CFR traversal в `pg_memory.add()` кладётся **action-level regret** — advantage действия Raise относительно текущей стратегии:

| Файл | Строка | Что кладётся |
|---|---|---|
| `src/core/deep_cfr.py` | 810 (vanilla CFR) | `action_values[3] - ev` (scaled log1p) |
| `src/core/deep_cfr.py` | 971 (hybrid OS) | `cf_regrets[3]` (clipped) |
| `src/core/deep_cfr.py` | 1093 (multi-traverse) | `cf_regrets[3]` |

`cf_regrets[3] = action_values[Raise] - EV(стратегии)` — это regret действия Raise, а не advantage конкретного размера ставки.

### Почему это ломает sizing

1. Когда Raise в среднем хуже Call/Fold → `cf_regrets[3]` отрицательный → PG получает сигнал "уменьшай sizing"
2. Нет способа выразить "этот размер рейза лучше другого размера" — reward одинаковый для всех размеров
3. Sizing policy учится минимизировать ущерб от Raise → коллапс к min-sizing (0.1)

### Подтверждение из метрик (fresh training после фикса #43)

- `Mean_Predicted_Size` стартует ~1.55, затем падает к 0.15–0.35 — устойчивый выбор нижнего края диапазона
- `PGRawRewardMean` осциллирует вокруг 0 (шумный)
- `PGFracPositiveAdvantage` ~0.35–0.55 — сигнал не мёртвый, но шумный
- `z_mean` bounded, `std` живой, `entropy` не объясняет collapse

**Вывод**: после фикса #43 численная стабильность восстановлена, но sizing всё равно коллапсирует из-за неправильного reward.

---

## Патч 1: Soft Weighted PG (текущий фикс)

### Идея

Отрицательный action-level regret не должен агрессивно толкать sizing policy к краю. Samples с положительным raise-regret получают больший вес.

### Реализация

**`src/core/deep_cfr.py`, `train_sizing_network()`:**

```python
temperature = cfg_get('pg_sizing_weight_temperature', 1.0)
raw_raise_signal = torch.clamp(regret_tensors.detach(), -3.0, 3.0)
raise_weight = torch.sigmoid(raw_raise_signal / temperature)

weighted_terms = raise_weight.unsqueeze(1) * log_prob * policy_advantages.unsqueeze(1)
policy_loss = -weighted_terms.sum() / (raise_weight.sum() + 1e-6)
pg_loss = policy_loss - effective_bonus * entropy
```

### Весы при `temperature = 1.0` и `regret ∈ [-3, 3]`

```
regret = -3.0 → weight ≈ 0.047
regret = -2.0 → weight ≈ 0.119
regret = -1.0 → weight ≈ 0.269
regret =  0.0 → weight = 0.500
regret =  1.0 → weight ≈ 0.731
regret =  2.0 → weight ≈ 0.881
regret =  3.0 → weight ≈ 0.953
```

### Что это НЕ решает

Если конкретный sampled size плох внутри raise-positive состояния, текущий reward всё равно этого не знает. Soft mask — стабилизатор, не финальное решение.

### Новые метрики

| Метрика | TensorBoard tag | Описание |
|---|---|---|
| `weight_mean` | `PG/SizingWeightMean` | Средний вес sample |
| `weight_std` | `PG/SizingWeightStd` | Разброс весов |
| `effective_batch` | `PG/SizingEffectiveBatchSize` | `(sum(w)^2 / sum(w^2))` — эффективный размер batch |

Если `effective_batch` часто < 16 при batch=64 → temperature слишком маленький или reward слишком отрицательный.

### Конфиг

```yaml
# config.yaml
pg_sizing_weight_temperature: 1.0  # Попробовать 0.75/1.5 если нужно
```

---

## Патч 2: Conditional Size Advantage (NEXT — не реализован)

### Концепция

При CFR traversal на узле с Raise сэмплировать несколько размеров, оценивать utility каждого, сохранять в PG memory не `cf_regrets[3]`, а:

```python
size_advantage = utility_raise_this_size - mean(utility_over_sampled_raise_sizes)
```

Это настоящий conditional objective для sizing: "этот размер лучше/хуже других размеров рейза".

### Сложность

- Требует изменения CFR traversal: multiple forward passes для каждого raise-узла
- Увеличивает стоимость traversal пропорционально количеству sampled sizes
- Нужен новый параметр `num_sizing_samples` в config

### Когда делать

Если soft weighted PG (патч 1) не устранит collapse к min-sizing после fresh run 300–500 итераций.

---

## Патч 3: Снятие `detach()` с sizing features (FUTURE)

### Текущее состояние

```python
# model.py: PokerNetwork.forward()
detached_features = features.detach()
sizing_params = self.sizing_head(detached_features)
```

Sizing loss не обучает shared base network. Градиенты от sizing_head не проходят в `self.base`.

### Почему НЕ снимать сейчас

1. `pg_optimizer` оптимизирует только `sizing_head.parameters()` — base не обновится даже без `detach()`
2. Текущий sizing reward шумный/неправильный — разрешить ему менять shared features опасно для `action_head`
3. Нужно: добавить base в `pg_optimizer` с маленьким LR + сначала исправить reward (патч 2)

### План после исправления reward

```python
self.pg_optimizer = optim.Adam([
    {"params": self.advantage_net.sizing_head.parameters(), "lr": _pg_lr},
    {"params": self.advantage_net.base.parameters(), "lr": _pg_lr * 0.1},
])
```

+ опциональный `detach_sizing_features: true/false` в config.

---

## Порядок действий

1. ✅ Soft weighted PG — уменьшает вред от отрицательного action-regret
2. `detach()` оставить пока reward не исправлен
3. Fresh run — проверить, уменьшает ли weighted PG collapse
4. Если collapse сохраняется → реализовать conditional size advantage (multi-sample)
5. Только после стабильного conditional size-reward → убрать `detach()`, добавить base в optimizer

---

## Связанные баги

- **Bug #43** (FIXED): `log_std` deadlock + `z_mean` runaway — численная нестабильность sizing
- **Bug #44** (этот): reward-design проблема — action-level regret вместо conditional size advantage

## Консенсус

- GPT и GLM согласны на диагноз и порядок патчей
- GPT: soft mask через weighted PG, `detach()` оставить
- GLM: первоначально предложил убрать `detach()`, принял аргумент что pg_optimizer не содержит base-параметров и что шумный reward опасен для shared features
- Оба согласны: сначала стабилизировать reward, потом открывать доступ к base
