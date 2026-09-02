# Баг-репорт: Vanilla CFR накапливает ошибки ранних итераций — внедрение DCFR+ Discounting

**Проект**: deepcfr-test  
**Дата**: 2026-05-07  
**Серьёзность**: High (конвергенция)  
**Статус**: Fixed (обновлено: P8-2, P8-3)  

---

## Описание

Vanilla CFR одинаково взвешивает данные итераций 1 и 5000. Ранние ошибки (когда бот ещё глупый) навсегда остаются в advantage_memory и загрязняют обучение. Это критично для 6-max NLH, где первые ~1000 итераций стратегия практически случайная.

---

## Источник

Статья VR-DeepPDCFR+ (Xu et al., AAAI 2026, `C:\Users\Cassmall\Desktop\DeepPredictive\deeppdcfr.tex`):

- Section 4.1: Fitting Cumulative Advantages by Bootstrapping
- Section 4.2: Approximating Advanced CFR Variants
- Algorithm 1: Training procedures for VR-DeepDCFR+

Ключевая формула bootstrapping loss:
```
L(θ_t) = E[(max(R(I,a|θ^{t-1}), 0) * (t-1)^α/((t-1)^α+1) + r(I,a) - R(I,a|θ^t))²]
```

---

## Реализация (P8)

### 1. Frozen prev_advantage_net

**Файл**: `src/core/deep_cfr.py`

```python
def prepare_iteration(self, iteration, traversing_player):
    # Hard copy раз в 6 итераций (полный цикл ротации)
    if iteration % self.num_players == 0:
        self.prev_advantage_net = copy.deepcopy(self.advantage_net)
        self.prev_advantage_net.eval()
        for p in self.prev_advantage_net.parameters():
            p.requires_grad = False
    # Очищаем ТОЛЬКО буфер текущего traversing_player
    self.advantage_memories[traversing_player].buffer.clear()
    self.advantage_memories[traversing_player].priorities.clear()
    self.advantage_memories[traversing_player].position = 0
```

### 2. DCFR+ Bootstrapping target

**Файл**: `src/core/deep_cfr.py` — `train_advantage_network`, `train_advantage_network_multi`

```python
t = self.iteration_count
if t > 1 and self.prev_advantage_net is not None:
    with torch.no_grad():
        prev_advantages, _, _, _ = self.prev_advantage_net(state_tensors)
        prev_pred = prev_advantages.gather(1, action_type_tensors.unsqueeze(1)).squeeze(1)
    discount = (t - 1) ** self.discount_alpha / ((t - 1) ** self.discount_alpha + 1)
    bootstrap_target = torch.clamp(prev_pred * discount + regret_tensors, min=0)
else:
    bootstrap_target = regret_tensors
```

Параметры: `discount_alpha = 2.0` (как в статье для DeepDCFR+).

---

## Критический дизайн-выбор: Hard Copy vs Polyak Averaging

### Проблема ротации

Архитектура использует `traversing_player = iteration % 6` — 1 позиция за итерацию. За полный цикл = 6 итераций. Статья предполагает все N игроков проходят за 1 итерацию.

### Рассмотренные варианты

| Вариант | Плюсы | Минусы |
|---------|-------|--------|
| **Hard copy каждую итерацию** | Просто | Очищает все 6 буферов → 5 всегда пустые |
| **Hard copy раз в 6 итераций** | Семантически корректно | Discount запаздывает на 6 шагов |
| **Polyak τ=0.2 каждую итерацию** | Плавный переход | prev_net = размытая смесь, не R_{t-1} |
| **Polyak + warm-up 30 iter** | Сглаживает ранний шум | Дополнительный гиперпараметр |

### Консультация с Claude (Sonnet 4.6 Adaptive)

> «Рекомендую hard copy раз в 6 итераций — это семантически правильнее. В оригинальном DCFR+ «итерация» = один полный проход по всем игрокам. У тебя это 6 последовательных шагов. prev_advantage_net должен быть снимком в начале полного цикла. Polyak создаёт семантическую проблему: R_prev перестаёт быть «предыдущей итерацией» — становится размытой смесью. Очистка только одного буфера — самое важное из двух изменений. Выбор между Polyak и hard copy — вторичен.»

**Решение**: Hard copy раз в 6 итераций + очистка только traversing_player.

---

## Эффективное поведение discount

При α=2.0:

| Итерация t | discount = (t-1)²/((t-1)²+1) | Доля старых весов |
|------------|-------------------------------|-------------------|
| 2 | 1/2 = 0.50 | 50% |
| 7 | 36/37 = 0.973 | 97.3% |
| 13 | 144/145 = 0.993 | 99.3% |
| 31 | 900/901 = 0.999 | 99.9% |

Интерпретация: после 7-й итерации старые regret-накопления почти полностью сохраняются, но при this + new regret формируют target. Это заставляет сеть «забывать» ранние ошибки медленнее, чем vanilla CFR (который взвешивает одинаково), но быстрее чем без discounting.

---

## Дополнительно: почему НЕ внедрён Predictive Update (PDCFR+)

Статья описывает VR-DeepPDCFR+ с дополнительной instantaneous advantage net для предсказания r^{t+1}. Это удваивает количество сетей (6 игроков × 2 nets = 12 advantage сетей).

Отложено по причинам:
1. RTX 4060 — memory constraint
2. DCFR+ даёт 80% выгоды PDCFR+ без extra сети
3. В статье DeepPDCFR+ сходится схоже с DeepDCFR+ в большинстве игр (разница значима только в Kuhn Poker — самой маленькой игре)

---

## Изменённые файлы

| Файл | Изменения |
|------|-----------|
| `src/core/deep_cfr.py` | `prev_advantage_net`, `discount_alpha=2.0`, `prepare_iteration(iteration, traversing_player)`, bootstrapping target в train_advantage_network/multi |
| `src/training/train.py` | `agent.prepare_iteration(iteration, traversing_player)` перед traversals |

---

## Валидация

Smoke test (8 итераций, 5 traversals):
- Advantage memory total растёт: 95→185→259→359→498→579 (буферы не теряются)
- После итерации 7 (P1 снова) total упал с 579 до 567 — только буфер P1 очищен
- Bootstrapping активируется с итерации 6 (iteration % 6 == 0)

---

## P8-2: Клиппинг bootstrapping target ДО суммы (подтверждено)

**Дата**: 2026-05-07  
**Источники**: Консультация с 3 AI (Claude Sonnet 4.6, Haiku 4.5, Gemini 2.5 Flash Thinking)  
**Статус**: Подтверждено после 2 промтов обсуждения

### Вопрос

Какой порядок операций для DCFR+ bootstrapping target при softmax-policy?

- **Вариант А (ДО суммы)**: `max(R_prev * discount, 0) + r_current` — клиппим только историческую часть
- **Вариант Б (ПОСЛЕ суммы)**: `max(R_prev * discount + r_current, 0)` — клиппим финальный результат

### Аргументация за клиппинг ДО суммы (Вариант А)

1. **Формула статьи**: VR-DeepPDCFR+ (Algorithm 1) — `max(R^{t-1}, 0) * discount + r^t` — клиппит ДО суммы
2. **Zero-Collapse при клиппинге ПОСЛЕ**: Если R_prev*d = -2.0, fold r_curr = -0.35, raise r_curr = -1.47:
   - ПОСЛЕ суммы: max(-2.35, 0) = 0, max(-3.47, 0) = 0 → fold и raise неразличимы
   - ДО суммы: max(-2.0, 0) + (-0.35) = -0.35, max(-2.0, 0) + (-1.47) = -1.47 → softmax различает
3. **log1p защищает от "мёртвых действий"**: Диапазон ±5.3, R не может уйти в -50
4. **Клиппинг ПОСЛЕ убивает преимущество softmax**: Превращает softmax в подобие RM, теряя работу с отрицательным пространством

### Ход обсуждения (2 промта)

| Промт | Gemini позиция | Результат |
|---|---|---|
| #1 | Клиппить ПОСЛЕ суммы | Расхождение с Claude/Haiku |
| #2 | Предоставлены контраргументы (формула статьи + Zero-Collapse + log1p) | **Gemini признал ошибку, перешёл на ДО суммы** |

### Консенсус 3 AI (после обсуждения)

| AI | Позиция | Ключевой аргумент |
|---|---|---|
| Claude Sonnet 4.6 | Клиппить ДО суммы | Softmax обрабатывает отрицательные advantages |
| Haiku 4.5 | Клиппить ДО суммы | Клиппинг после суммы вводит bias |
| **Gemini 2.5 Thinking** | Клиппить ДО суммы | Zero-Collapse убивает различимость; log1p защищает от глубокого минуса |

**Все 3 AI едины**: `max(R_prev * discount, 0) + r_current` — клиппить только историческую часть, `r_current` без клиппинга.

### Реализация (подтверждена)

```python
bootstrap_target = torch.clamp(prev_pred * discount, min=0) + regret_tensors
```

**Файлы**: `src/core/deep_cfr.py` — строки 610 и 667 (2 места: `train_advantage_network`, `train_advantage_network_multi`)

---

## P8-3: Ускорение τ-decay (5.0→0.5 за 500 итераций)

**Дата**: 2026-05-07  
**Источники**: Консультация с 3 AI (Claude Sonnet 4.6, Haiku 4.5, Gemini 2.5 Flash Thinking)

### Проблема: τ-decay vs discount timing conflict

При τ-decay за 3000 итераций:

| Итерация | discount | τ | Проблема |
|---|---|---|---|
| 10 | 0.81 | 4.95 | 81% шумных regrets сохраняются при uniform стратегии |
| 100 | 0.99 | 4.67 | **Крайне критично**: discount≈1, стратегия ещё 90% uniform |
| 500 | ≈1.00 | 3.33 | Старые regrets ЗАМОРОЖЕНЫ, стратегия наконец дифференцируется |

DCFR+ discount→1 к итерации ~10 (α=2.0), но τ ещё 4.8. На итерациях 10-500 discount≈1 (старые regrets полностью сохраняются), а стратегия почти равномерная (τ=5.0). DCFR+ "запоминает" шумные regrets от почти случайной игры.

### Консенсус 3 AI

| AI | Рекомендация | Вариант |
|---|---|---|
| Claude Sonnet 4.6 | Конфликт реален | Ускорить τ |
| Haiku 4.5 | КРИТИЧНО, вероятность 70% | Предложил 3 варианта (A/B/C) |
| **Gemini 2.5 Thinking** | Фикс нужен, Вариант А | Ускорить τ, НЕ менять discount |

**Все 3 AI согласны**: τ-decay нужно ускорить. Вариант C (tanh вместо DCFR+ формулы) отвергнут — ломает теоретическое обоснование.

### Исправление

```python
# ДО (P8):
temperature = max(0.5, 5.0 - 4.5 * iteration / 3000.0)

# ПОСЛЕ (P8-3):
temperature = max(0.5, 5.0 - 4.5 * iteration / 500.0)
```

**Файлы**: `src/training/train.py` (строки 122, 279), `src/core/deep_cfr.py` (строки 299, 454)

### Логика

- Итерации 1-50: τ ≈ 5.0→0.5 → "горячий сбор данных", высокая энтропия — допустимо, discount ещё мал
- К итерации 100: τ ≈ 0.5 → стратегия уже дифференцирована, DCFR+ "замораживает" уже осмысленные regrets
- Шум итераций 1-50 не критичен: их веса в итоговой средней стратегии ничтожны

### Отклонённые альтернативы

| Вариант | Почему отклонён |
|---|---|
| B: Regret reset на t=200 | Потеря данных, ad-hoc |
| C: tanh(t/50) вместо DCFR+ discount | Ломает теоретическое обоснование α=2.0 |

---

## Итоговая матрица решений (3 AI консенсус)

| Вопрос | Решение | Claude | Haiku | Gemini |
|---|---|---|---|---|
| Клиппинг target | ДО суммы (все 3 согласны) | До суммы | До суммы | До суммы (после обсуждения) |
| τ-decay timing | За 500 итер. | Ускорить | Ускорить | Ускорить |
| V(s) vs V(s,a) | Один V(s) | Достаточен | Достаточен | Достаточен |
| Shared weights | Оставить | — | Interference 60% | Преувеличено |
| Итерации 6000 | Запустить, смотреть | — | -6bb/100h | Мало, нужно 50K+ |
