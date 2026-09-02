# Bug #62: Разрыв между q_net[Raise] и sizing_q_net — корень call-heavy стратегии

**Severity:** CRITICAL

**Дата:** 2026-06-04

**Связан с:** Bug #61 (коллапс sizing в 0.10)

---

## Симптомы

После фиксов #61 (tp_b=1→2) и смягчения heat (temperature 0.35→0.7, min_advantage 0.03→0.005) модель всё равно коллапсирует к 400 итерации:

| Метрика | iter_100 | iter_200 | iter_400 |
|---------|----------|----------|----------|
| raise_freq | 46.9% | 22.5% | **8.4%** |
| 0.10+0.25 доля | 8.7% | 52.8% | **83.1%** |
| call% | 46.7% | 42.5% | 64.0% |
| mean_reward | 23.20 | -1.99 | 0.81 |

Модель становится call-heavy, рейзит редко и только мелкими сайзингами. Фиксы #61 решили **симптом** (sizing-коллапс), но не **причину** (недооценка Raise как действия).

---

## Root Cause: q_net[Raise] систематически занижен

Новая диагностика `checkpoint_tools.py diagnose` (Q-сравнение) на iter_200 показала:

```
Action-level Q (q_net):
  Fold:  mean=-3.64
  Call:  mean=-5.55
  Raise: mean=-5.82   ← ХУЖЕ Call!

q_net[Raise] < q_net[Call]  in 64.0% of states
q_net[Raise] < q_net[Fold]  in 96.0% of states
```

При этом sizing_q_net оценивает лучший размер ставки положительно:

```
max_sizing_q: mean=+0.57
q_net[Raise] - max_sizing_q: -6.39   ← разрыв >6 единиц
```

### Интерпретация

Две Q-сети живут в разных мирах:

1. **sizing_q_net** учится на Q-таргетах конкретных размеров ставок. Видит: «ставка 0.10 = хорошо (+0.57)»
2. **q_net** учится на action-таргетах. Видит: «Raise → бет 0.10 → мало EV → Raise = плохо (-5.82)»

Q-таргет для Raise вычисляется через sizing — если модель рейзит мелко, Q(Raise) будет низким. Это создаёт feedback loop:

```
модель рейзит 0.10 (мелко) 
       ↓
Q(Raise) учится, что рейз = мало EV 
       ↓
q_net[Raise] < q_net[Call] 
       ↓
модель реже рейзит, а когда рейзит — мелко 
       ↓
(повтор)
```

**Корень:** q_net[Raise] не знает, что можно рейзить крупнее и получать больше EV. Он оценивает Raise по тому сайзингу, который модель реально использует. А модель использует мелкий, потому что q_net говорит что Raise плох.

---

## Диагностика

### Инструмент

`tools/checkpoint_tools.py` — унифицированная утилита. Три режима:

#### `diagnose` — NaN-check + collapse + Q-сравнение (JSON авто)
```powershell
py -c "import sys; sys.argv=['t','diagnose','models/multi/multi_checkpoint_iter_400.pt','--num-states','3000']; import tools.checkpoint_tools; tools.checkpoint_tools.main()"
```
→ `*_diagnose_report.json`

#### `eval` — NaN-check + игры + вердикт (JSON авто)
```powershell
# strategy_net, sampling (обычный)
py -c "import sys; sys.argv=['t','eval','models/multi/multi_checkpoint_iter_400_light.pt','--games','3000']; import tools.checkpoint_tools; tools.checkpoint_tools.main()"

# strategy_net, argmax (детерминистик)
py -c "import sys; sys.argv=['t','eval','models/multi/multi_checkpoint_iter_400_light.pt','--games','3000','--deterministic']; import tools.checkpoint_tools; tools.checkpoint_tools.main()"

# advantage_net + regret matching, sampling
py -c "import sys; sys.argv=['t','eval','models/multi/multi_checkpoint_iter_600.pt','--games','3000','--regret-matching']; import tools.checkpoint_tools; tools.checkpoint_tools.main()"

# advantage_net + regret matching, argmax
py -c "import sys; sys.argv=['t','eval','models/multi/multi_checkpoint_iter_600.pt','--games','3000','--regret-matching','--deterministic']; import tools.checkpoint_tools; tools.checkpoint_tools.main()"

# vs другой чекпоинт (self-play / cross-play)
py -c "import sys; sys.argv=['t','eval','models/multi/multi_checkpoint_iter_400_light.pt','--opponent','models/multi/multi_checkpoint_iter_100_light.pt','--games','10000']; import tools.checkpoint_tools; tools.checkpoint_tools.main()"
```
→ `*_eval_report.json`

#### `full` — diagnose + eval + JSON
```powershell
py -c "import sys; sys.argv=['t','full','models/multi/multi_checkpoint_iter_600.pt','--games','3000','--num-states','3000']; import tools.checkpoint_tools; tools.checkpoint_tools.main()"
```
→ `*_full_report.json`

#### Флаги

| Флаг | Режимы | Описание |
|------|--------|----------|
| `--games N` | eval, full | Количество игр |
| `--num-states N` | diagnose, full | Диагностических состояний |
| `--deterministic` | eval, full | argmax вместо sampling |
| `--regret-matching` | eval | advantage_net + regret matching (только full чекпоинты) |
| `--opponent PATH` | eval, full | Чекпоинт противника (все 5 слотов) |
| `--top-buckets N` | diagnose, full | Бакетов для sparsify |
| `--top-per-bucket N` | diagnose, full | Анкеров на бакет |

JSON сохраняется **автоматически** во всех режимах, флаг `--output` удалён.

### Ключевая метрика

Блок `Q COMPARISON` в режиме `diagnose`:
- `q_net[Raise] < q_net[Call] in XX%` — если >50%, Raise недооценён
- `q_net[Raise] - max(sizing_q)` — разрыв между action Q и sizing Q
- Если разрыв >3 единиц — две сети расходятся, нужна синхронизация

---

## Возможные фиксы

| # | Подход | Механизм | Риск |
|---|--------|----------|------|
| A | Q-таргет для Raise = max(sizing_q), а не Q от фактического сайзинга | Разрывает feedback loop: q_net[Raise] видит лучший возможный сайзинг | Меняет семантику Q-target |
| B | Q-таргет для Raise = среднее по топ-K sizing_q | Мягче варианта A, учитывает uncertainty | Нужен тюнинг K |
| C | Добавить exploration bonus к q_net[Raise] в loss | Искусственно поднимает Q(Raise) при недостатке сэмплов | Может перекосить в другую сторону |
| D | Разделить q_target_normalization для Raise | Сейчас `sizing_q_target_normalization: 'pot'` — нормировка на пот делает крупные ставки менее привлекательными | Нужно тестирование |
| E | Увеличить probe_prob для sizing — больше крупных проб | Больше данных о крупных сайзингах → Q(Raise) точнее | Шум в обучении |

---

## Статус

**Открыт.** Фиксы #61 решили симптом, но не причину. Разрыв между q_net и sizing_q_net требует архитектурного решения.

### Критерии исправления
- `q_net[Raise] < q_net[Call]` падает ниже 30%
- `q_net[Raise] - max(sizing_q)` становится < 2.0
- raise_freq стабилизируется выше 15% на iter_400+
