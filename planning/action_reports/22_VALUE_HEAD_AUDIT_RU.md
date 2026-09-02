# Баг-репорт #22: Аудит Value Head — не мёртвый код, а активный стабилизатор sizing-пути

## Статус: НЕ БАГ — Архитектурный аудит (информационный)

## Описание

При обсуждении с Gemini было достигнуто ошибочное согласие, что `value_head` в `PokerNetwork` — «мёртвый код», который рассчитывается, но нигде не используется в loss. Углублённый аудит кода показал, что это **неверно**.

## Факты

### Value Head ИСПОЛЬЗУЕТСЯ в sizing-пути

**Файл**: `src/core/deep_cfr.py`

1. **Отдельный оптимизатор** (строка ~202):
```python
self.value_optimizer = optim.Adam(self.advantage_net.value_head.parameters(),
                                   lr=cfg_get('value_lr', 1e-3))
```

2. **Активное использование в `train_sizing_network()`** (строки ~749-756):
```python
_, z_mean, log_std, values = self.advantage_net(state_tensors)
std = torch.exp(log_std)

advantages = regret_tensors - values.squeeze(1).detach()   # BASELINE!
advantages = torch.clamp(advantages, -3.0, 3.0)

value_loss = F.mse_loss(values.squeeze(1), regret_tensors.detach())
self.value_optimizer.zero_grad()
value_loss.backward()
self.value_optimizer.step()
```

### Value Head НЕ используется в advantage-пути

В `train_advantage_network()` и `train_advantage_network_multi()` четвёртый выход `forward()` игнорируется (`_value`). Это **корректно**, потому что CFR считает regrets аналитически через обход дерева — baseline не нужен для детерминированных значений.

## Архитектурный анализ: двухуровневый Advantage

Система реализует элегантную «матрёшку» преимуществ:

| Уровень | Компонент | Что считает |
|---------|-----------|-------------|
| 1 (CFR) | `regret_raise = action_values[2] - ev` | Преимущество Raise над текущей стратегией |
| 2 (Value Head) | `V(s)` обучается на `scaled_regret_raise` | Ожидаемое преимущество рейза в состоянии |
| 3 (PG Sizing) | `advantages = regret_tensors - values.detach()` | Насколько конкретный сайзинг лучше среднего рейза |

**Ключевой инсайт**: PG для sizing-а не видит «сырых» выигрышей. Он видит только отклонение от ожидаемого преимущества рейза. Это изолирует выбор размера ставки от общей силы руки.

## Решение

**НЕ УДАЛЯТЬ Value Head.** Он:

1. Активно работает как baseline (variance reduction) для PG sizing-обучения
2. Обучается с `value_lr=1e-3` (10x быстрее advantage_lr=1e-4) — корректно для Actor-Critic (критик должен опережать актора)
3. Не мешает advantage-пути (CFR-регреты детерминированы, baseline там не нужен)
4. Не является высоконагруженным решением — это стандартный логичный ход для PG sizing

## Консенсус с Gemini

- Stage-Specific Heads → План Б, не текущая задача (stage уже one-hot)
- Value Head → Активный стабилизатор sizing-пути, НЕ мёртвый код
- value_lr=1e-3 → Оставить как есть (критик должен учиться быстрее актора)
- Архитектура одобрена к запуску без изменений
