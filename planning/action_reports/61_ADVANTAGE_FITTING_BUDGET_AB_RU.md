# Баг-репорт #61: избыточный fitting budget advantage-сети

## Статус: ИЗМЕНЕНО, ГОТОВО К A/B-ПРОГОНУ

## Проблема

Текущий full-buffer режим обучал action advantage-сеть слишком большим количеством
optimizer updates на каждой CFR-итерации:

```text
advantage_batch_size: 1024
advantage_epochs: 3
buffer: до 1_000_000 samples
```

При полном буфере это примерно:

```text
3 * ceil(1_000_000 / 1024) = 2931 updates
```

Для bootstrapped cumulative advantage target это может быть чрезмерным fitting
budget: сеть слишком хорошо подгоняется под snapshot текущего буфера, хотя regret
landscape меняется от итерации к итерации.

## Решение

В `config.yaml` уменьшен fitting budget только для advantage-сети:

```yaml
advantage_batch_size: 2048
advantage_epochs: 1
```

Strategy-сеть оставлена без изменений:

```yaml
strategy_batch_size: 1024
strategy_epochs: 3
```

Причина: strategy-сеть аппроксимирует среднюю политику по истории, поэтому её
недообучение является отдельным риском. Этот A/B должен изолированно проверить
гипотезу переобучения advantage-сети.

## Итоговая конфигурация

```yaml
hidden_size: 256

advantage_lr: 1.0e-4
strategy_lr: 5.0e-5

advantage_batch_size: 2048
strategy_batch_size: 1024

advantage_epochs: 1
strategy_epochs: 3
```

## Ожидаемый эффект

При полном advantage-buffer на 1M samples количество updates станет примерно:

```text
ceil(1_000_000 / 2048) = 489 updates
```

Это должно снизить риск overfit-а к текущему advantage snapshot, не меняя LR,
weight decay, архитектуру сети и training budget strategy-сети.

## Метрики для чтения результата

- `last_advantage_profile.actual_steps == expected_steps`
- `LossAdvantage`
- `Train/AdvantageLearningRate`
- `target_abs_max` и `error_abs_mean`
- `PerformanceRaiseFreq`
- `next_strategy_raise_mass`
- поздний checkpoint против раннего checkpoint

## Откат

Вернуть прежние значения:

```yaml
advantage_batch_size: 1024
advantage_epochs: 3
```
