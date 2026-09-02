# Баг #12: Одноигроковый CFR не сходится к Нэшу в мультиплеере

**Серьёзность:** Критическая (архитектурная)
**Статус:** Исправлено
**Дата:** 2026-05-06

## Суть проблемы

DeepCFRAgent обучался **только для player_id=0**. Остальные 5 позиций всегда были фиксированными агентами (RandomAgent или checkpoint). Это не Nash Equilibrium — это **best-response** против заданного набора оппонентов.

### Почему это баг

Оригинальный Deep CFR (Brown et al. 2018) для 2--player zero-sum:
- Traversal для **обоих** игроков
- Раздельные advantage-сети на каждого игрока
- Сходится к Nash (теорема Zinkevich et al. 2008)

Наш код для 6-player:
- Traversal **только** для player_id=0
- Одна advantage-сеть, один буфер регретов
- **Не сходится к Nash** — сходится к best-response

Pluribus (Brown & Sandholm 2019) для 6-player:
- Traversal **по очереди** для каждого из 6 игроков
- Общая advantage-сеть (с player_id на входе)
- Сходится к ε-Nash

## Как проявлялось

1. **Phase 1 (vs random):** Агент учится бить рандом → легко, но стратегия бессмысленная
2. **Phase 2 (vs checkpoint):** Агент стартует с нуля против сильного оппонента → «спираль смерти»
3. **Phase 3 (mixed pool):** Лучше за счёт разнообразия, но всё равно best-response, не Nash
4. **Итог:** Стратегия переобучается под конкретных оппонентов и не является равновесной

## Исправление

### 1. 6 раздельных буферов регретов

```python
# БЫЛО:
self.advantage_memory = PrioritizedMemory(memory_size)  # один буфер

# СТАЛО:
self.advantage_memories = [PrioritizedMemory(memory_size) for _ in range(num_players)]
self.advantage_memory = self.advantage_memories[player_id]  # алиас для обратной совместимости
```

### 2. Мультиплеерный обход cfr_traverse_multi()

```python
# БЫЛО (cfr_traverse):
if current_player == self.player_id:  # всегда 0
    ...собираем регреты в advantage_memory...
else:
    action = random_agents[current_player].choose_action(state)  # фиксированные оппоненты

# СТАЛО (cfr_traverse_multi):
if current_player == traversing_player:  # вращается: iteration % 6
    ...собираем регреты в advantage_memories[traversing_player]...
else:
    # Оппоненты сэмплируют из strategy_net (лучшая версия себя)
    logits = strategy_net(encode_state(state, current_player))
    action = sample(softmax(logits))
```

Ключевое отличие: **оппоненты = strategy_net**, не фиксированные агенты. Все 6 игроков используют одну сеть с разными позициями на входе.

### 3. Ротация traversing_player

```python
# Былый цикл:
for iteration in range(N):
    traversing_player = 0  # ВСЕГДА 0
    cfr_traverse(state, iteration, random_agents)

# Новый цикл:
for iteration in range(N):
    traversing_player = iteration % 6  # ВРАЩАЕМ
    cfr_traverse_multi(state, iteration, traversing_player)
    train_advantage_network(player_id=traversing_player)  # учим на данных текущего игрока
```

### 4. train_advantage_network(player_id=...)

```python
# БЫЛО:
def train_advantage_network(self, ...):
    if len(self.advantage_memory) < batch_size:  # всегда буфер игрока 0
        return 0
    batch = self.advantage_memory.sample(...)

# СТАЛО:
def train_advantage_network(self, player_id=None, ...):
    if player_id is None:
        player_id = self.player_id
    memory = self.advantage_memories[player_id]
    if len(memory) < batch_size:
        return 0
    batch = memory.sample(...)
```

### 5. train_advantage_network_multi() — обучение на всех 6 игроках

При обучении advantage-net **только** на данных текущего traversing_player возникает catastrophic forgetting — сеть забывает преимущества других игроков. Метод `train_advantage_network_multi()` обучает на смешанных данных всех 6 буферов одновременно.

### 6. Фикс encode_state (баг #13-A)

`initial_stake = state.players_state[0].stake` мог быть < 1 после all-in → деление на ≈0 → секстиллионы в нормализованных фичах → взрыв advantage loss. Фикс: `if initial_stake < 1.0: initial_stake = 1.0`

## Почему общие веса работают

1. `encode_state(state, player_id)` содержит **one-hot позицию** → сеть «знает» для какого игрока считает
2. Покер — **симметричная игра** (позиции вращаются через баттон) → существует симметричное равновесие
3. Одна сеть с разными входами = одна стратегия, адаптированная к позиции
4. Подтверждено в статье Steinberger "Single Deep CFR" (2020)

## Compute: НУЛЕВАЯ разница

| Метрика | Было (vs random) | Стало (multi) |
|---------|-----------------|---------------|
| Traversals/iter | 200 | 200 |
| Advantage train/iter | 1 раз | 1 раз |
| Strategy train | Каждые 10 iter | Каждые 10 iter |
| RAM | ~300K записей | ~1.8M записей (+360 МБ) |

## Запуск

```bash
py -m src.training.train --self-play-multi --iterations 6000 --traversals 200 --save-dir models/multi --log-dir logs/multi
```

С warm start:
```bash
py -m src.training.train --self-play-multi --initial-checkpoint models/phase1/checkpoint_iter_1000.pt --iterations 6000
```

## Обратная совместимость

- Существующие режимы (`--self-play`, `--mixed`, `--checkpoint`) **без изменений**
- `cfr_traverse()` и `_cfr_traverse_with_opponents()` **без изменений**
- `advantage_memory` — алиас на `advantage_memories[0]` для старого кода
- `train_advantage_network()` без `player_id` — учит на `self.player_id` (былое поведение)

## Затронутые файлы

- `src/core/deep_cfr.py`: +110 строк (6 буферов, cfr_traverse_multi, train_advantage_network с player_id, total_advantage_memory_size)
- `src/training/train.py`: +105 строк (train_self_play_multi, --self-play-multi флаг, обработка в CLI)
- `src/core/model.py`: без изменений
