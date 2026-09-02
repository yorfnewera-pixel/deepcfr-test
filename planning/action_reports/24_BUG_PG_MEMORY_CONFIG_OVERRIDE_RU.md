# Баг-репорт: pg_memory_size=100000 перебивает config.yaml (10000)

**Проект**: deepcfr-test  
**Дата**: 2026-05-09  
**Серьёзность**: Medium  
**Статус**: Fixed  

---

## Симптомы

В консоли при тренировке отображалось `PG memory: 100000`, хотя `config.yaml` задавал `pg_memory_size: 10000`. Буфер PG хранил данные за 1000+ итераций, затягивая обучение sizing_head устаревшими (stale) данными.

## Корневая причина

В `src/training/train.py` — **5 функций** + **1 argparse** имели хардкод `pg_memory_size=100000`:

| Строка | Функция | Дефолт |
|--------|---------|--------|
| 288 | `train_deep_cfr()` | `pg_memory_size=100000` |
| 394 | `train_deep_cfr_continued()` | `pg_memory_size=100000` |
| 513 | `train_deep_cfr_selfplay()` | `pg_memory_size=100000` |
| 734 | `train_deep_cfr_mixed()` | `pg_memory_size=100000` |
| 941 | `train_deep_cfr_multi()` | `pg_memory_size=100000` |
| 1133 | argparse `--pg-memory-size` | `default=100000` |

Конструктор `DeepCFRAgent` использует:
```python
_pg_memory_size = pg_memory_size or cfg_get('pg_memory_size', 10000)
```

Поскольку `pg_memory_size=100000` передавался явно (не None), `or` не срабатывал → config.yaml игнорировался.

## Почему 10000 правильно, а 100000 — нет

1. **PG учит только сайзинг** (sizing_head), не всю стратегию. Батч=64, буфер=10000 → 156 уникальных батчей — достаточно
2. **Stale data**: при ~30-60 PG-записей/итерацию, 100000 ≈ 1700-3300 итераций в буфере. Данные с итерации 1000 неактуальны на итерации 2100
3. **10000 даёт ~170-330 итераций** — «золотая середина» (<100 — мало, >500 — слишком устаревшие данные)

Консенсус с Gemini: 10000 — правильный размер.

## Фикс

Все дефолты изменены с `100000` → `None`:

```python
# До:
def train_deep_cfr(..., pg_lr=1e-4, entropy_bonus=0.01, pg_memory_size=100000):
parser.add_argument('--pg-memory-size', type=int, default=100000, ...)

# После:
def train_deep_cfr(..., pg_lr=None, entropy_bonus=None, pg_memory_size=None):
parser.add_argument('--pg-memory-size', type=int, default=None, help='... (default: from config.yaml)')
```

Теперь config.yaml — единый источник истины для всех PG-параметров.

## Верификация

```
agent = DeepCFRAgent(pg_memory_size=None)
→ agent.pg_memory_size = 10000  ✅
→ agent.pg_memory.capacity = 10000  ✅
→ agent.pg_lr = 0.0001  ✅
→ agent.entropy_bonus = 0.01  ✅
```

## Затронутые файлы

- `src/training/train.py` — 6 изменений (5 функций + 1 argparse)

## Примечание

При следующей тренировке PG-буфер начнётся с чистого листа (размер 10000 вместо 100000). Возможен временный всплеск PG-loss пока sizing_head переучивается на свежих данных, но затем сходимость улучшится.
