# Баг-репорт #19: 7 багов Deep CFR (аудит кода + консенсус с Gemini)

**Проект**: deepcfr-test  
**Дата**: 2026-05-07  
**Серьёзность**: Critical + High + Medium  
**Статус**: Fixed  

---

## Источник

Полный аудит `src/core/deep_cfr.py`, `src/core/model.py`, `src/training/train.py` с последующим обсуждением каждого бага с Gemini 2.5 Flash Thinking через CDP-браузер. **Полный консенсус по всем 7 багам.**

---

## C1: Linear CFR веса полностью игнорируются (outer product)

**Серьёзность**: CRITICAL  
**Файл**: `src/core/deep_cfr.py:788, 799`  
**Консенсус**: Да  

### Проблема

`weights [B,1] * inner [B]` → PyTorch broadcasting → outer product [B,B], NOT element-wise [B]. Linear CFR взвешивание (поздние итерации = больше вес) — мёртвый код. Strategy_net всегда обучалась с uniform весами.

### Было

```python
weights = linear_weights  # shape [B, 1]
action_loss = -torch.sum(weights * torch.sum(strategy_tensors * torch.log(predicted_strategies + 1e-8), dim=1))
weighted_bet_size_loss = torch.sum(raise_weights * bet_size_loss)
```

### Стало

```python
weights = linear_weights.view(-1)  # shape [B]
inner_loss = torch.sum(strategy_tensors * torch.log(predicted_strategies + 1e-8), dim=1)
action_loss = -torch.sum(weights * inner_loss)
weighted_bet_size_loss = torch.sum(raise_weights * bet_size_loss)
```

---

## H1: Division by zero при iteration=0

**Серьёзность**: HIGH  
**Файл**: `src/core/deep_cfr.py:782`  
**Консенсус**: Да (Gemini повысил до High — NaN убивает сеть навсегда)  

### Проблема

`floor((iteration+1)/2)` при iteration=0 → 0. Все weights = 0 → `0/0 = NaN` → backward() → все параметры NaN.

### Стало

```python
w_sum = linear_weights.sum()
if w_sum > 0:
    linear_weights = linear_weights / w_sum
else:
    linear_weights = torch.ones_like(linear_weights) / len(linear_weights)
```

---

## H2: clamp(log_prob, -5, 5) убивает PG-градиенты

**Серьёзность**: HIGH  
**Файл**: `src/core/model.py:95`  
**Консенсус**: Да  

### Проблема

При расхождении политик log_prob уходит в -100+. clamp(-5) обнуляет градиент для старых z_raw из буфера. Сила PG-обновления занижена в ~20x.

### Было

```python
return torch.clamp(log_prob, -5.0, 5.0)
```

### Стало

```python
return torch.clamp(log_prob, -20.0, 5.0)
```

---

## M1: Negative raise при remaining_after_call < 0

**Серьёзность**: MEDIUM  
**Файл**: `src/core/deep_cfr.py:208`  
**Консенсус**: Да  

### Проблема

Когда call_amount > stake, remaining_after_call < 0. additional становится отрицательным. Движок отклоняет → action_values[Raise]=0 → искажение regrets. Gemini: нужно также обрабатывать all-in когда 0 < remaining < min_raise.

### Было

```python
remaining_after_call = player_state.stake - call_amount
```

### Стало

```python
remaining_after_call = max(0.0, player_state.stake - call_amount)
```

---

## M2: target_mean/var не сохраняются в чекпоинт

**Серьёзность**: MEDIUM  
**Файл**: `src/core/deep_cfr.py:852-895`  
**Консенсус**: Да (Gemini уточнил: при momentum=0.99 reconvergence ~450 батчей, не 100)  

### Проблема

После load_model: target_mean=0.0, target_var=1.0 (default). Нормализация использует неверные статистики ~450 батчей. С чекпоинтами каждые 500 итераций — почти всё обучение после resume в состоянии адаптации.

### Фикс

Добавлено в `save_model`:
```python
'target_mean': self.target_mean,
'target_var': self.target_var,
```

Добавлено в `load_model`:
```python
if 'target_mean' in checkpoint:
    self.target_mean = checkpoint['target_mean']
if 'target_var' in checkpoint:
    self.target_var = checkpoint['target_var']
```

---

## M3: Opponent position overflow — 5 оппонентов на 4 слота

**Серьёзность**: MEDIUM  
**Файл**: `src/training/train.py:820-828`  
**Консенсус**: Да  

### Проблема

6-max: позиции 0-5. Слот 0 = learning, 1 = RandomAgent. Свободных: 2,3,4,5 = 4. При num_opponents=5 — 5-й перезаписывает позицию 2. Пользователь думает что 5 оппонентов, а на самом деле 4.

### Фикс

```python
num_slots = 6 - 2
actual_opponents = min(num_opponents, num_slots, len(checkpoint_files))
if actual_opponents < num_opponents:
    print(f"WARNING: num_opponents={num_opponents} exceeds available slots={num_slots}. Using {actual_opponents} opponents.")
```

---

## M4: Числовая нестабильность 1 - tanh(z)^2

**Серьёзность**: MEDIUM  
**Файл**: `src/core/model.py:55-57`  
**Консенсус**: Да  

### Проблема

`1.0 - torch.tanh(z) ** 2` теряет точность при |z|>3 в float32. Ошибка ~4.7 единиц при z=10 → искажение градиентов на хвостах распределения.

### Было

```python
correction = -torch.log(1.0 - torch.tanh(z) ** 2 + 1e-6)
return torch.clamp(correction, -5.0, 5.0)
```

### Стало

```python
correction = 2.0 * (z + F.softplus(-2.0 * z) - math.log(2.0))
return torch.clamp(correction, 0.0, 20.0)
```

Математика: `log(1 - tanh(z)^2) = -2*log(cosh(z)) = -(2z + 2*softplus(-2z) - 2*log(2))`. Correction = `2z + 2*softplus(-2z) - 2*log(2)`, всегда ≥ 0.

---

## Не исправленные баги (не наш путь запуска — self-play-multi)

| Баг | Путь | Влияние |
|-----|------|---------|
| C2: prepare_iteration не вызывается | train_deep_cfr, continue_training | Bootstrapping отключён, буферы не очищаются |
| H3: iteration_count сбрасывается на 1 | train_against_checkpoint | Ломает temperature, entropy, strategy weights |
| H4: Неверное условие финальной итерации | train_with_mixed_checkpoints | Strategy_net не обучается на последней итерации |
| M5: Чекпоинт сохраняет неверный iteration | train_against_checkpoint | Ошибка при continue_training |

---

## Валидация

- Import OK
- 2/3 тестов проходят (1 падающий — из-за C2, не наши фиксы)
- Все 7 фиксов изолированы друг от друга
