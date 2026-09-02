# #99 — Random Agent + Per-Iteration Traversing Player + Optional Q (v1)

**Дата:** 2026-06-20
**Статус:** IMPLEMENTED + SUPERSEDED by v2 (прогон N5_3)
**База:** N5 (R2 + OS off + advantage_reward_scale 200 + sizing_cfr_mode + legality boundary)

> **⚠ v2 исправления (#99v2):** per-player стратеджи-буферы откачены к синглу, opp sizing через advantage_sizing (не strategy), per-player advantage sizing, adaptive batch. Strategy — чистый distillation. См. `sizing_reports/99-v2-cfr-purity-architecture-fixes.md`.

---

## Диагноз

Мульти-агентный режим с ротацией traversing_player на каждом ТРАВЕРСЕ (а не итерации) создаёт нестабильность: advantage-сети всех игроков получают перемешанные данные внутри одной итерации. Q-сеть создаёт self-suppressing loop через bootstrap (raise структурно обездолен). Нужно:

1. Изолировать traversing_player на целую итерацию
2. Добавить random-агента на одно место (как в базовом `train_deep_cfr`)
3. Стратеджи учить только на данных текущего наблюдаемого игрока
4. Сделать Q опциональным (отключаемым через конфиг)

---

## Архитектура (5+1 игроков)

```
Игроки за столом (6):      0    1    2    3    4    5
                          сеть сеть сеть сеть сеть  RandomAgent
                          
traversing_player:        iteration % 5  →  циклит по 0-4
button:                   random.randint(0, 5) каждый травес
```

- **5 advantage-сетей** (по одной на игрока 0-4), **5 advantage-буферов**
- **RandomAgent всегда на player_id=5** — не обучается, не пишет в буферы  
- **1 общая strategy-сеть** учится на **5 per-player strategy-буферах**
- **Button случайный** каждый травес → все позиции эффективно случайны относительно баттона

### Обход `cfr_traverse_multi`

```python
if current_player == 5 (random):
    → random_agent.choose_action(state), без записи в буферы
    
elif current_player == traversing_player:
    → hero-нода: regret matching + запись в advantage_buffer[traversing_player]
    
else:  # остальные 4 игрока (0-4, не traversing)
    → opponent-нода: advantage_net[current_player] → стратегия → сэмплинг
```

### Передача `random_agent` через рекурсию

Используется instance-переменная `self._traversal_random_agent`, устанавливаемая на depth=0. Все рекурсивные вызовы не требуют изменения сигнатуры — глубокая интеграция без правки 3+ внутренних вызовов.

**Критический багфикс:** `random_agent` добавлен ПОСЛЕ `depth` в сигнатуре метода (`depth=0, random_agent=None`), чтобы не сломать все внутренние позиционные рекурсивные вызовы `cfr_traverse_multi(new_state, iter, trav, depth+1)`.

---

## Per-player стратеджи-буферы

### Было
- 1 общий `strategy_buffer`, копит всех игроков
- Тренировка раз в 10 итераций на всём миксе

### Стало
- **5 per-player** `strategy_buffers[0..4]` + `sizing_strategy_buffers[0..4]`
- `prepare_iteration(traversing_player)` чистит ТОЛЬКО буфер текущего traversing
- Данные пишутся в `strategy_buffers[traversing_player]`
- `train_strategy_network(player_id=traversing_player)` — **каждую итерацию** (не раз в 10)
- `train_strategy_sizing_anchor_network(player_id=traversing_player)` — каждую итерацию

### Обобщение
Одна общая strategy-сеть учится по очереди на перспективе каждого из 5 игроков. Сеть видит данные каждого наблюдаемого → обобщение через саму архитектуру.

---

## Опциональный Q (action-level + sizing)

### Новый guard в `__init__`

```python
if self.use_q_baseline:
    self.q_net = QValueNetwork(...)
    self.q_target_net = ...
    self.q_buffer = ...
else:
    self.q_net = None
    self.q_target_net = None
    self.q_buffer = None
```

### Что отключается при `use_q_baseline: false`

| Компонент | Эффект |
|---|---|
| `q_net` / `q_target_net` | Не создаются (0 памяти) |
| Q variance reduction на opp-node | Пропускается |
| Outcome sampling | Авто-отключён (`_should_use_outcome_sampling` проверяет `use_q_baseline`) |
| `train_q_network` | Возвращает `None` |
| Checkpoint save/load | Q-поля = `None`, обрабатываются корректно |

### Guard'ы (deep_cfr.py + train.py + checkpoint_tools.py)
- `cfr_traverse_multi`: opp-node проверяет `if self.use_q_baseline`
- `_sizing_q_is_ready`: ранний return при `sizing_q_buffer is None`
- `_add_sizing_q_sample`: guard `sizing_q_buffer is not None`
- `_sync_q_target_net`: guard `q_net is not None`
- `_train_q_network_refit`: guard `q_net/q_buffer is not None`
- Checkpoint save: тернарный `... if self.q_net is not None else None`
- Checkpoint load: guard `checkpoint['q_net'] is not None and self.q_net is not None`
- `checkpoint_tools.py`: `run_weight_sanity`, `load_full_checkpoint` — guard for None

### Конфиг (`config.yaml`)
```yaml
use_q_baseline: false      # action-level Q выкл
sizing_q_enabled: true     # sizing Q вкл (source0/1, source2 через sizing_lookahead_enabled)
```

---

## Точки изменений

### `src/core/deep_cfr.py`

| # | Локация | Что |
|---|---------|-----|
| 1 | `__init__:661` | `self.num_trainable_players = 5` |
| 2 | `__init__:678-712` | advantage-сети/буферы: `range(self.num_trainable_players)` (5 шт) |
| 3 | `__init__:724-816` | 5 `strategy_buffers` + 5 `sizing_strategy_buffers` |
| 4 | `__init__:872-913` | Q-net guard: создаётся только при `use_q_baseline=True` |
| 5 | `__init__:1117` | `self._traversal_random_agent = None` |
| 6 | `prepare_iteration` | Чистка `strategy_buffers[tp]` + `sizing_strategy_buffers[tp]` |
| 7 | `cfr_traverse_multi:2536` | Сигнатура `(..., depth=0, random_agent=None)` |
| 8 | `cfr_traverse_multi:2555-2568` | Random агент ветка: player 5 → `random_agent.choose_action` |
| 9 | `cfr_traverse_multi` (hero/opp nodes) | Все `strategy_buffer.add` → `strategy_buffers[traversing_player].add` |
| 10 | `_cfr_traverse_multi_outcome_node` | Аналогичная замена strategy buffer записей |
| 11 | `train_strategy_network` | Параметр `player_id=0`, использует `strategy_buffers[player_id]` |
| 12 | `train_strategy_sizing_anchor_network` | Параметр `player_id=0` |
| 13 | `train_sizing_anchor_network` | Параметр `player_id=0` |
| 14 | `_sizing_q_is_ready` | Ранний return при `sizing_q_buffer is None` |
| 15 | `_add_sizing_q_sample` | Guard `sizing_q_buffer is not None` |
| 16 | `_sync_q_target_net` / `_train_q_network_refit` | Guard `q_net/q_buffer is not None` |
| 17 | Checkpoint save/load | Тернарные Q-поля + guard при загрузке |
| 18 | Q-bootstrap `range(self.num_players)` | → `range(self.num_trainable_players)` |

### `src/training/train.py`

| # | Локация | Что |
|---|---------|-----|
| 19 | `train_deep_cfr` | `traversing_player = (iter-1) % num_trainable` (вне цикла травесов) |
| 20 | `train_deep_cfr` | `random_opponent = RandomAgent(num_players-1)` |
| 21 | `train_deep_cfr` | `cfr_traverse_multi(..., random_agent=random_opponent)` |
| 22 | `train_deep_cfr` | `for p in range(num_trainable)` для advantage обучения |
| 23 | `train_deep_cfr` | Стратеджи обучение каждую итерацию с `player_id=traversing_player` |
| 24 | `continue_training` | Аналогичные изменения (п. 19-23) |
| 25 | `train_self_play_multi` | Аналогичные изменения (п. 19-23) |
| 26 | `train_against_checkpoint` | `for p in range(learning_agent.num_trainable_players)` + стратеджи `player_id=0` |
| 27 | `train_with_mixed_checkpoints` | Аналогично п. 26 |
| 28 | `_cfr_traverse_with_opponents` | `agent.strategy_buffers[0].add(...)` |
| 29 | `_train_pg_and_log` | Параметр `player_id=0`, передаётся в `train_sizing_anchor_network` |
| 30 | Все `len(strategy_buffer)` в логах | → `sum(len(b) for b in strategy_buffers)` |
| 31 | `clean_buffers` | `for buf in strategy_buffers: buf.clear()` |

### `tools/checkpoint_tools.py`

| # | Локация | Что |
|---|---------|-----|
| 32 | `run_weight_sanity` | Guard for `None` + `advantage_nets` list support |
| 33 | `load_full_checkpoint` | `if 'q_net' in ckpt and ckpt['q_net'] is not None:` (×3: q_net/q_target/sizing_q) |

### `config.yaml`

| # | Ключ | Значение |
|---|------|----------|
| 34 | `use_q_baseline` | `false` |
| 35 | `sizing_q_enabled` | `true` |
| 36 | `sizing_target_ema_enabled` | `false` — сырые per-state sizing-таргеты без EMA-бакетизации |

### Отключение EMA в sizing-таргетах

`sizing_target_ema_enabled: false` — ключевая переменная прогона N5_3:

- При `false` → `_smooth_sizing_target_by_bucket` сразу возвращает сырые weights (строка 1745), бакетизация не используется
- Каждое состояние получает свой per-state sizing-таргет — как action-сеть
- При `true` (или без ключа — default) → включается EMA-бакетизация обратно
- Это делает sizing-обучение симметричным action-обучению: никакого усреднения между состояниями

---

## Результаты (N5_3 — прогон 98seed1N5_3)

**Лучший по стабильности прогон из всех.**

| iter | raise% | win% | reward |
|------|--------|------|--------|
| 100 | 19.1 | 11.3 | 6.4 |
| 300 | 19.3 | 9.0 | 6.8 |
| 500 | 21.6 | 12.6 | 12.0 |
| 800 | 19.1 | 9.6 | 9.2 |
| 1100 | 18.8 | 11.3 | 14.7 |
| 1400 | 22.2 | 12.1 | 10.2 |
| 1600 | 16.5 | 9.0 | 9.6 |

**Ключевые наблюдения:**
- **EMA в sizing отключена** (`sizing_target_ema_enabled: false`) — сырые per-state sizing-таргеты без бакетизации. Симметрично с action-сетью.
- **raise держится ~16-22% на протяжении ВСЕХ 1600 итераций — дрейфа НЕТ**
- Это ключевое отличие: у N5/N5_2 raise сползал (42→10, 37→6), здесь — плоско ~19% до самого конца
- **Остаточный дрейф, который беспокоил, устранён**
- H2H 1600_vs_100: reward +1.3 → поздний бьёт ранний (как и раньше, late > early) ✓

### Почему win_rate низкий при положительном mean_reward

Win rate 9-12%, но mean_reward +6-15 — это TAG-профиль (Tight-Aggressive): модель играет мало рук, но когда входит — забирает крупные банки. В 6-max случайный win rate = 16.7%. Модель выигрывает меньше раздач, но больше фишек — оптимизация профита, а не частоты.

### Вердикт

Стабильность достигнута. Дрейф raise_freq устранён. Следующие шаги: повышение win_rate через усиление оппонентов (self-play против чекпоинтов, не random), масштабирование до 10k+ итераций.

---

## См. также

- `sizing_reports/98-n5-global-fixed-reward-scale.md` — N5 baseline
- `sizing_reports/SIZING.md` — полная хронология sizing bugs
