# Баг-репорт: Fold-Collapse — агент фолдит премиум-руки (AA/KK/QQ) на префлопе

**Проект**: deepcfr-test  
**Дата**: 2026-05-06  
**Серьёзность**: Critical  
**Статус**: Open  

---

## Симптомы

После 3500 итераций × 400 traversals обучения Deep CFR (6-max NL Hold'em):

- **76.7% действий — Fold** (184/240 в тесте из 200 рук)
- **6 из 7 премиум-рук (AA/KK/QQ) фолдятся на префлопе**
- strategy_net выдаёт Fold probability 91-96% на **всех** руках, включая 77 (95% fold)
- advantage_net при этом корректно оценивает: Fold-regret для KK = -0.301, Raise-regret = +0.098

```
Пример из теста:
Ad As | preflop=['RAISE '] | reward=662.4     ← единственная сыгранная AA
As Ad | preflop=['Fold']    | reward=-0.0      ← фолд AA
Qc Qs | preflop=['Fold']    | reward=-1.0      ← фолд QQ
Ks Kd | preflop=['Fold']    | reward=-0.0      ← фолд KK
```

---

## Корневые причины

### Баг A: ε-exploration отсутствует в CFR traversal (Critical)

**Файл**: `src/core/deep_cfr.py:289-294`, `src/core/deep_cfr.py:433-438`, `src/training/train.py:130-136`

Когда advantage_net ещё не обучен и все advantages ≤ 0, стратегия формируется так:

```python
else:
    strategy = np.zeros(self.num_actions)
    best_a = max(legal_action_types, key=lambda a: advantages[a])
    strategy[best_a] = 1.0  # ← 100% одно действие, 0 exploration
```

Это **детерминистическая** стратегия. Если Fold случайно получает наивысший advantage → стратегия = [1.0, 0, 0]. Эта стратегия записывается в strategy_memory.

**Death spiral**:
```
Fold strategy → Fold-записи доминируют в strategy_memory
→ strategy_net учится "Fold по умолчанию"
→ на inference фолдит всё, включая AA
→ advantage_net не получает достаточные контрпримеры
→ Fold advantage остаётся "лучшим" → цикл замыкается
```

Оригинальный Deep CFR (Brown et al. 2018) использует ε-on-policy sampling: с вероятностью ε агент играет по текущей стратегии, с (1-ε) — равномерно по легальным действиям. Здесь ε = 0.

### Баг B: PrioritizedMemory.sample() игнорируется (Critical)

**Файл**: `src/core/deep_cfr.py:581`

```python
# PrioritizedMemory ИМЕЕТ метод sample() с приоритетным сэмплированием:
def sample(self, batch_size, beta=0.4):
    ...
    indices = np.random.choice(len(self.buffer), batch_size, p=probabilities, replace=False)
    ...

# Но train_advantage_network ИГНОРИРУЕТ его и делает uniform:
idxs = random.sample(range(len(memory.buffer)), min(batch_size, len(memory.buffer)))
```

**Последствие**: AA — ~0.45% рук. При буфере 300K и батче 256 → ~1.15 записи AA на батч. Сеть практически не видит сильные руки. При приоритетном сэмплировании (priority = |regret| + 0.01) AA с Fold-regret ~0.3 получила бы priority ~0.31 vs 0.02 для слабых рук → ~15x чаще в батчах.

Аналогично `train_strategy_network` (строка 711) — uniform sampling.

### Баг C: Regret normalization подавляет сигнал для сильных рук (High)

**Файл**: `src/core/deep_cfr.py:327-336`

```python
max_abs_val = max(abs(max(action_values)), abs(min(action_values)), 1.0)
normalized_regret = regret / max_abs_val
clipped_regret = np.clip(normalized_regret, -2.0, 2.0)
```

Проблема: `max_abs_val` берётся по action_values, не по regret. Для AA на префлопе action_values могут быть ~30+ (большой профит), но regret = action_value - ev ≈ несколько единиц. Нормализация через max_abs_val ≈ 30 делает normalized_regret ≈ 0.1 → clip не нужен, но сигнал подавлен.

**Обратная зависимость**: чем сильнее рука → тем больше action_values → тем больше max_abs_val → тем слабее normalized_regret → тем слабее gradient signal для advantage_net.

Для слабых рук: action_values маленькие → max_abs_val маленький → normalized_regret относительно большой → сеть лучше учится на слабых руках, чем на сильных.

---

## Количественная оценка

При 3500 итераций × 400 traversals:
- Всего обходов: ~1.4M
- Обходов для learning player: ~233K (1/6)
- Обходов с AA: ~1,050 (0.45%)
- Записей AA на батч 256 при uniform: **~0.9** → почти ни одной
- Записей AA на батч 256 при PER (priority ×15): **~13.5** → значимо

---

## Предлагаемые исправления

### P0-B: ε-exploration в CFR traversal (Critical)

Добавить смешивание с равномерной стратегией:

```python
epsilon = max(0.05, 0.3 * (1.0 - iteration / 3000.0))  # decay 0.3 → 0.05
uniform = np.zeros(self.num_actions)
for a in legal_action_types:
    uniform[a] = 1.0 / len(legal_action_types)
strategy = (1 - epsilon) * strategy + epsilon * uniform
```

3 места: `cfr_traverse`, `cfr_traverse_multi`, `_cfr_traverse_with_opponents`.

### P0-A: Использовать PrioritizedMemory.sample() (Critical)

Заменить uniform на приоритетное сэмплирование с IS-весами:

```python
samples, indices, weights = memory.sample(batch_size, beta=0.4)
weight_tensors = torch.FloatTensor(weights).to(self.device)
action_loss = (F.smooth_l1_loss(predicted_regrets, regret_tensors, reduction='none') * weight_tensors).mean()
```

2 места: `train_advantage_network`, `train_advantage_network_multi`.

### P1-C: Regret scaling через log1p (High)

Заменить нормализацию через max_abs_val:

```python
scaled_regret = np.sign(regret) * np.log1p(abs(regret))
clipped_regret = np.clip(scaled_regret, -2.0, 2.0)
```

3 места: `cfr_traverse`, `cfr_traverse_multi`, `_cfr_traverse_with_opponents`.

---

## Затронутые файлы

| Файл | Изменения |
|---|---|
| `src/core/deep_cfr.py` | cfr_traverse/multi: +ε-exploration, +log1p regret; train_advantage_network: PER + IS weights |
| `src/training/train.py` | _cfr_traverse_with_opponents: +ε-exploration, +log1p regret |

---

## Окружение

- Python 3.x, PyTorch
- Покерный движок: `pokers` (Rust-based)
- Обучение: 6-max NL Hold'em, sb=1, bb=2, stake=200
- Чекпоинт: `models/multi/multi_checkpoint_iter_3500.pt` (3500 итераций, 400 traversals/iter)
