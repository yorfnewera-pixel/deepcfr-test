# Баг-репорт: Деградация PG Sizing Head при обучении Deep CFR

**Проект**: deepcfr-test  
**Дата**: 2026-05-06  
**Серьёзность**: Critical  
**Статус**: Open  

---

## Симптомы

При обучении Deep CFR агента наблюдаются следующие аномалии в метриках TensorBoard (~2781 шаг):

| Метрика | Наблюдение | Ожидание |
|---|---|---|
| `PGMeanAdvantage2` | ~1e-8 (практически ноль) | Ненулевое значение, отражающее силу преимущества |
| `PGMeanEntropy` | Растёт с 1.44 → 2.0+ | Снижение по мере сходимости политики |
| `LossPG` | Осциллирует, уходит в отрицательную сторону, выброс 4.4 на ~шаге 923 | Стабильное снижение |
| `SizingMean_Standard_Deviation` | Растёт с 1.02 → 2.0+ | Стабилизация или снижение |
| `SizingMean_Predicted_Size` | Относительно стабильно ~1.57-1.65 | Адаптация к оптимальным сайзингам |

**Итог**: Sizing head не получает значимый градиент → политика расходится к равномерной случайной.

---

## Корневые причины

### Баг 1: PG Reward = сырой Q(s, Raise) вместо Regret Raise (Critical)

**Файл**: `src/core/deep_cfr.py:342-348`, `src/core/deep_cfr.py:489-495`, `src/training/train.py:206-212`

В `pg_memory.add()` сохраняется `action_values[2]` — значение Q для Raise. Но для PG нужен **advantage** = `Q(s,a) - V(s)`, где V(s) = EV текущей стратегии. В контексте CFR это **regret Raise** = `action_values[2] - ev`.

Без вычитания baseline V(s) сигнал смещён: даже «плохой» Raise получает положительный Q, если игра в целом прибыльна. Нет дифференциации между хорошими и плохими сайзингами.

**Воспроизведение**: Любой прогон обучения. `action_values[2]` всегда положительно в прибыльных спотах → PG получает неверный сигнал.

---

### Баг 2: Z-score нормализация зануляет advantage (Critical)

**Файл**: `src/core/deep_cfr.py:650-654`

```python
reward_std = reward_tensors.std()
if reward_std > 1e-8:
    advantages = (reward_tensors - reward_tensors.mean()) / reward_std
else:
    advantages = reward_tensors - reward_tensors.mean()
```

После Z-score нормализации `advantages.mean() ≈ 0` **по определению математики**. Метрика `PGMeanAdvantage2` логирует `advantages.mean().item()` — она **всегда** ≈ 0, независимо от реальной силы сигнала. Это делает метрику полностью бесполезной для диагностики.

Более того, Z-score на малых батчах (batch_size=64) создаёт высоковариативный advantage, что дестабилизирует обучение.

---

### Баг 3: Entropy bonus доминирует над PG-сигналом (Critical)

**Файл**: `src/core/deep_cfr.py:666`

```python
pg_loss = -(log_prob * advantages).mean() - self.entropy_bonus * entropy
```

- `log_prob * advantages` → слабый сигнал (advantages малы после Z-score, log_prob невысок)
- `entropy_bonus * entropy` → всегда значимый положительный член (entropy ≥ 1.0 для Normal с std ≥ 0.5)
- **Минимизация loss → максимизация entropy → std растёт → политика расходится**

Нет механизма снижения entropy bonus со временем. На ранних итерациях exploration полезен, но позже нужна эксплуатация.

---

### Баг 4: Некорректный log_prob из-за clamp (High)

**Файл**: `src/core/deep_cfr.py:256-259`, `src/core/deep_cfr.py:422-425`, `src/training/train.py:118-121`

```python
_sampled = dist.sample()                          # z = 3.5 (например)
_sampled = torch.clamp(_sampled, 0.1, 3.0)       # z_clamped = 3.0
raise_log_prob = dist.log_prob(_sampled).item()   # log_prob(3.0) — НЕ log_prob(3.5)!
```

**Проблема**: log_prob вычисляется на **clamped** значении, а не на оригинальном сэмпле. Это создаёт систематический bias:

- Если сэмпл близок к краю (3.5 → clamp → 3.0), log_prob(3.0) **выше**, чем реальный log_prob(3.5)
- Градиент получает ложный сигнал «3.0 — высоковероятное значение, продолжай двигаться к краю»
- Это поощряет сеть к краевым значениям, что усиливает энтропию

**Двойной clamp**: `action_type_to_pokers_action` (строка 182) делает clamp ещё раз → clamp на clamp.

---

### Баг 5: Чрезмерно широкий диапазон std (Medium)

**Файл**: `src/core/model.py:43`

```python
log_std = torch.clamp(log_std_raw, -5.0, 1.0)
```

`e^1.0 = 2.72` — для множителя рейза в диапазоне [0.1, 3.0] это чрезмерная дисперсия. Сеть может сэмплировать значения далеко за пределами осмысленного диапазона, что:
- Увеличивает частоту срабатывания clamp → усугубляет баг 4
- Создаёт высокую энтропию → усиливает баг 3
- Дестабилизирует обучение через высоковариативные сэмплы

---

## Взаимосвязь багов (каскадная деградация)

```
Баг 5 (широкий std)
  → больше крайних сэмплов
    → Баг 4 (clamp log_prob bias) сильнее
      → ложный градиент к краям
        → std растёт дальше (петля обратной связи)

Баг 1 (сырой Q вместо regret)
  → смещённый reward
    → Баг 2 (Z-score) маскирует проблему
      → advantage ≈ 0
        → Баг 3 (entropy доминирует)
          → политика расходится
```

---

## Предлагаемое исправление

### Решение: SAC-style Tanh Squashing (Variant В)

Полная замена механизма сэмплирования sizing на Tanh-squashed Gaussian policy (как в Soft Actor-Critic). Это устраняет баги 4 и 5 **архитектурно**:

1. **Архитектура**: sizing_head выдаёт `z_mean` (raw) и `log_std` (clamped [-5, 0])
2. **Сэмплирование**: `z ~ Normal(z_mean, std)` → `bet_size = min + (max-min) * (tanh(z)+1)/2`
3. **log_prob**: `log Normal(z) - log(1 - tanh(z)²)` — коррекция смены переменных
4. **Сеть физически не может** сэмплировать вне [min, max] — clamp не нужен

Дополнительно:
- Reward = regret Raise (`action_values[2] - ev`) вместо сырого Q
- Running baseline (экспоненциальное скользящее среднее) вместо Z-score
- Entropy decay: `effective_bonus = bonus * max(0.1, 1.0 - iteration/3000)`
- Clip advantages [-1, 1] от выбросов
- Информативные метрики: raw reward mean/std, доля положительных advantages

### Затронутые файлы

| Файл | Изменения |
|---|---|
| `src/core/model.py` | PokerNetwork: forward(), +sample_sizing(), +mean_sizing(), min/max bet params |
| `src/core/deep_cfr.py` | __init__: +running baseline; cfr_traverse/multi: regret reward, sample_sizing(); train_sizing_network: полная переработка; train_strategy_network: mean_sizing(); choose_action: mean_sizing() |
| `src/training/train.py` | _cfr_traverse_with_opponents: regret reward, sample_sizing(); _train_pg_and_log: новые метрики |

---

## Окружение

- Python 3.x, PyTorch
- Покерный движок: `pokers` (Rust-based)
- Обучение: 6-max NL Hold'em, sb=1, bb=2, stake=200
- Гиперпараметры по умолчанию: pg_lr=1e-4, entropy_bonus=0.01, pg_memory_size=100000
