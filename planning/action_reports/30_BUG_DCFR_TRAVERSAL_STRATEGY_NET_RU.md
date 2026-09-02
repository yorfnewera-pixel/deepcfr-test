# BUG 30 — DCFR+ traversal использует `strategy_net` вместо текущей regret-matching стратегии

## Статус
Критичный алгоритмический баг / архитектурное отклонение от Deep DCFR+.

## Где
Основной файл:

- `src/core/deep_cfr.py`
- метод `DeepCFRAgent.cfr_traverse_multi()`

Связанные места:

- `train_advantage_network_multi()`
- `train_strategy_network()`
- `AdvantageBuffer`
- `StrategyBuffer`

## Суть проблемы
В текущей реализации traversal устроен так:

- для `traversing_player` используется `advantage_net` и regret matching+;
- для остальных игроков используется `strategy_net` и softmax по logits.

То есть non-traversing players выбирают действия через average/stale policy approximation, а не через текущую CFR-стратегию из cumulative regrets.

В canonical Deep CFR/DCFR+ traversal должен использовать current strategy profile:

```text
advantage_net -> cumulative regrets -> regret matching+ -> current_strategy
```

Для traversing player нужно перебирать все legal actions. Для non-traversing players нужно sample one action из current_strategy. Но источник current_strategy должен быть тот же: `advantage_net`, а не `strategy_net`.

## Почему это плохо

### 1. `strategy_net` вмешивается в traversal
`strategy_net` должна аппроксимировать average strategy и использоваться для финальной игры/evaluation. Сейчас она влияет на то, какие ветки дерева посещаются в CFR traversal.

### 2. Обучение идёт против stale/average opponents
`advantage_net` обновляется каждую итерацию, а `strategy_net` обучается реже. Поэтому traversing player играет не против самой свежей current CFR strategy, а против сглаженной/устаревшей average strategy.

### 3. `strategy_net` частично учится на самой себе
В opponent branch probabilities из `strategy_net` записываются обратно в `strategy_buffer`. Получается петля:

```text
strategy_net -> strategy_buffer -> train_strategy_network -> strategy_net
```

Это self-distillation, а не обучение average strategy из regret matching policies.

### 4. Advantage replay слишком большой для raw instantaneous regrets
Сейчас есть per-player `AdvantageBuffer`, но очищается только buffer текущего traversing player. Затем `train_advantage_network_multi()` учится на всех player buffers.

Итог:

```text
fresh regrets текущего игрока + old raw regrets остальных игроков
```

Для всех samples строится target:

```text
target = clamp(target_net(state), 0) * discount + regret_tensor
```

Если `regret_tensor` старый, он повторно добавляется как новый instantaneous regret. Это не чистая DCFR+ динамика.

## Решение v1 — минимальный безопасный фикс

### Шаг 1
Убрать запись opponent predictions из `strategy_net` в `strategy_buffer`.

В `cfr_traverse_multi()` в ветке `current_player != traversing_player` удалить/закомментировать блок, который делает:

```text
opp_strategy = probs from strategy_net
strategy_buffer.add(opp_state_arr, opp_strategy, opp_mask, iteration)
```

`StrategyBuffer` должен получать targets только из policies, полученных через:

```text
advantage_net -> regret matching+
```

### Шаг 2
Перевести non-traversing players на `advantage_net + regret matching+`.

Было:

```text
strategy_net -> logits -> masked softmax -> sample action
```

Должно стать:

```text
advantage_net -> regrets -> regret matching+ -> sample action
```

Cost не растёт x6, потому что non-traversing players всё ещё sample only one action.

### Шаг 3
Оставить `strategy_net` только для:

- `choose_action()`;
- evaluation;
- light checkpoint;
- final average policy.

Не использовать `strategy_net` внутри CFR traversal.

## Решение v2 — buffer stabilization
Не делать сразу `clear all advantage buffers` без защиты, потому что одна общая `advantage_net` на 6 игроков может забывать других игроков.

Быстрый pragmatic fix:

```text
memory_size = 4096 на игрока
итого ≈ 24K raw regret samples total
```

Допустимый диапазон:

```text
2048-8192 samples per player
12K-49K total
```

Это не идеальный DCFR+, но лучше, чем 300K на игрока.

Правильный долгосрочный вариант:

```text
fresh regret buffer + rehearsal/distillation buffer
```

Где fresh buffer хранит fresh instantaneous regrets, а rehearsal buffer хранит frozen cumulative targets / old predictions, но не old raw regrets как additive term.

## Решение v3 — архитектура
Для защиты от forgetting лучше заменить one-hot-only conditioning на:

```text
shared trunk + 6 per-player heads
```

Или хотя бы использовать разные learning rates:

```text
shared trunk lr ≈ 3e-5
player head lr ≈ 1e-4
```

## Ожидаемый результат
После фикса:

- traversal станет ближе к Deep DCFR+;
- opponents будут играть current regret-matching strategy из свежей `advantage_net`;
- `strategy_net` перестанет вмешиваться в regret updates;
- `strategy_net` перестанет частично обучаться на самой себе;
- replay старых raw regrets будет ограничен или заменён корректным rehearsal.

## Приоритет внедрения

1. Убрать запись opponent `strategy_net` probs в `strategy_buffer`.
2. Перевести opponents в traversal на `advantage_net + regret matching+`.
3. Уменьшить `memory_size` до 4096 на игрока как быстрый фикс.
4. Позже внедрить fresh/rehearsal split и per-player heads.

## Проверка
Добавить fixed probe set по каждому player_id и логировать:

- KL стратегии до/после train;
- regret sign flip rate;
- argmax action change rate;
- winrate vs random;
- winrate vs previous checkpoints;
- fold/call/raise frequencies by street and position.
