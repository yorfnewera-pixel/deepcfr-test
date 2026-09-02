# Bug #43 — sizing policy collapse: от `log_std` deadlock к `z_mean` runaway

## Статус: FIXED (3 итерации фикса)

## История

### Фикс A — `log_std` sigmoid параметризация (первый патч)

В `PokerNetwork.forward()` параметризация `log_std` через `torch.clamp(x, -2.0, 0.0)`:
- Нулевая инициализация `sizing_head[-1]` → `raw=0` → `clamp(0, -2, 0) = 0` → `std = 1.0`
- На границе `0.0` градиент обрезается → `std` не может двигаться вниз

**Фикс**: `log_std = -2.0 + 2.0 * sigmoid(raw_log_std)` → стартовый `std ≈ 0.368`

**Результат fresh training (390 итераций)**: `std` действительно стартовал ~0.368 и начал двигаться, но sizing policy всё равно коллапсировал — теперь к max bet вместо min bet. `z_mean` улетел до 20-50, `Predicted Size` → 3.0.

### Фикс B — smooth bound для `z_mean` (второй патч)

`z_mean = 20+` численно бессмысленен после `tanh`. Это не решает #44 полностью, но убирает численный runaway.

**Было**: `z_mean = sizing_params[:, 0:1]` — неограниченный
**Стало**: `z_mean = 3.0 * tanh(raw_z_mean / 3.0)` — `z_mean ∈ [-3, 3]`

Почему `3.0`: `tanh(3) ≈ 0.995` — практически max bet. Значения больше 3 не несут информации после `_squash`.

### Фикс C — batch-centering для `policy_advantages` (второй патч)

Корневая проблема (#44): PG получает сигнал "рейзить в среднем плохо", а не "этот размер лучше других". Глобальный bias доминирует над относительным sizing-сигналом.

**Было**:
```python
advantages = regret_tensors - values.squeeze(1).detach()
advantages = torch.clamp(advantages, -3.0, 3.0)
pg_loss = -(log_prob * advantages.unsqueeze(1)).mean() - ...
```

**Стало**:
```python
advantages = regret_tensors - values.squeeze(1).detach()
advantages = torch.clamp(advantages, -3.0, 3.0)

policy_advantages = advantages - advantages.mean()
policy_advantages = policy_advantages / (policy_advantages.std(unbiased=False) + 1e-6)

pg_loss = -(log_prob * policy_advantages.unsqueeze(1)).mean() - ...
```

Value loss по-прежнему использует оригинальные `advantages`. `policy_advantages` — только для PG sizing.

Эффект: глобальный сигнал "Raise плох" вычитается. Остаётся относительный сигнал внутри batch. Дешёвая аппроксимация conditional size advantage без multi-sample CFR.

## Что НЕ делаем

### Не clamp-им entropy снизу

При `log_std < -1.42` энтропия Normal становится отрицательной. Это математически корректно. `clamp(min=0)` убил бы градиент, который **сопротивляется** collapse `std`:

```
loss_entropy = -bonus * H
H = const + log_std
d(loss_entropy)/d(log_std) = -bonus  → увеличивает std
```

Отрицательная энтропия — индикатор, не причина death spiral. (Оспорено GLM, принято после проверки математики.)

## Дополнительные изменения: расширенное логирование

**`src/core/deep_cfr.py`, `train_sizing_network()`:**
- Добавлены метрики: `z_mean_mean`, `z_mean_std`, `log_std_mean`
- Return tuple: 8 → 11 значений

**`src/training/train.py`:**
- Unpacking обновлён под 11-кортеж
- Добавлен print: `z_mean: ...±... log_std: ...`
- Добавлены TensorBoard scalars: `Sizing/ZMean`, `Sizing/ZMeanStd`, `Sizing/LogStd`

## Итоговый код `PokerNetwork.forward()`

```python
raw_z_mean = sizing_params[:, 0:1]
z_mean = 3.0 * torch.tanh(raw_z_mean / 3.0)

raw_log_std = sizing_params[:, 1:2]
log_std = -2.0 + 2.0 * torch.sigmoid(raw_log_std)
```

## Совместимость checkpoint

Структура `sizing_head` **не изменена** — веса совместимы. Но старые checkpoint'ы с деградировавшим `z_mean` могут не восстановиться. Рекомендуется **fresh training**.

## Критерии валидации (fresh training, 300–500 итераций)

| Метрика | Tag | Ожидание |
|---------|-----|----------|
| ZMean | `Sizing/ZMean` | Остается в [-3, 3], не улетает |
| Std | `Sizing/Mean_Standard_Deviation` | Старт ~0.368, стабильный |
| Mean Size | `Sizing/Mean_Predicted_Size` | Не прилипает к краю (0.1 или 3.0) |
| FracPositive | `PG/FracPositiveAdvantage` | ~0.5 (после centering) |

## Связанные баги

- **Bug #44** (conditional size advantage): Batch-centering — дешёвая аппроксимация. Если sizing всё равно нестабилен — переходить к multi-sample conditional size advantage.

## Консенсус

- GPT и GLM согласны на фиксах A/B/C.
- GLM ошибся про "death spiral от отрицательной энтропии" — градиент сопротивляется collapse, не усиливает. Принято после проверки.
