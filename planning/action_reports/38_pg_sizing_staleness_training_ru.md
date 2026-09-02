================================================================================
БАГ-РЕПОРТ #38: PG sizing head — недообучение, staleness, дисбаланс с advantage
================================================================================

ТИП: Performance / Data quality
ДАТА: 2026-05-15
СТАТУС: Исправлено
СВЯЗАННЫЕ: #36 (Hybrid OS), #37 (AdvantageBuffer training)

================================================================================
ОПИСАНИЕ
================================================================================

Три взаимосвязанных проблемы с PG sizing head:

1. НЕДООБУЧЕНИЕ: train_sizing_network вызывается 1 раз за итерацию,
   batch_size=64, 1 optimizer step. Advantage net делает ~1170 steps.
   Дисбаланс: 1 vs 1170. Sizing head почти не учится.

2. STALENESS: pg_memory — circular buffer без очистки. z_raw сэмплирован
   старой политикой, log_prob считается текущей → off-policy bias без IS
   correction. При capacity=10000 и ~8000 записей/итерацию, старые записи
   живут ~1-2 итерации. Но с Hybrid OS (~300 записей/итерацию) старые записи
   живут ~33 итерации — sizing head учится на данных из 33 итераций назад.

3. МАЛЫЙ БУФЕР: pg_memory_size=10000. С OS только sampled Raise пишет в
   pg_memory (~300/итерацию). Буфер заполняется медленно, но содержит смесь
   свежих и старых записей без способа различить их.

================================================================================
КОНСЕНСУС (GLM + GPT/Opus)
================================================================================

1. pg_memory_size: 10000 → 30000 — да.
2. pg_train_steps: 1 → 5 (config flag) — да.
3. pg_memory.clear() — НЕТ. При OS даёт ~300 записей → overfitting на
   крошечной выборке одной итерации. Circular buffer без clear() лучше.
4. Iteration tracking в PolicyGradientMemory — да. Добавить _iterations.
5. Freshness window — да. Обучаться только на записях из последних N итераций.
6. Без очистки, stale data вытесняется новыми естественным образом через
   circular buffer + freshness фильтр при sampling.

================================================================================
ИСПРАВЛЕНИЯ
================================================================================

1. config.yaml + _DEFAULTS
   - pg_memory_size: 10000 → 30000
   - pg_train_steps_per_iteration: 5 (новый)
   - pg_freshness_window: 5 (новый)

2. PolicyGradientMemory (src/core/deep_cfr.py)
   - Добавлен self._iterations = np.empty(capacity, dtype=np.float32)
   - add(): новый параметр iteration=-1
   - sample(): новый параметр min_iteration=None, фильтрация по свежести
     Если fresh записей < batch_size — вернуть все свежие (не None)

3. Все pg_memory.add() вызовы
   - Передают iteration=iteration (3 места: cfr_traverse, cfr_traverse_multi,
     _cfr_traverse_multi_outcome_node)

4. train_sizing_network (src/core/deep_cfr.py)
   - Новый параметр min_iteration=None
   - Если pg_freshness_window > 0, автоматически считать:
     min_iteration = max(0, iteration_count - pg_freshness_window + 1)
   - Передаёт min_iteration в pg_memory.sample()

5. _train_pg_and_log (src/training/train.py)
   - Вызывает train_sizing_network() N раз (pg_train_steps_per_iteration)
   - Логирует avg loss/advantage по всем шагам
   - Новый TensorBoard scalar: Train/PGSteps

================================================================================
ПОВЕДЕНИЕ С HYBRID OS
================================================================================

С OS (~300 Raise записей/итерацию):
  - Freshness window = 5 → 5 × 300 = 1500 свежих записей
  - 5 train steps × batch 64 = 320 записей используется
  - Достаточно для стабильного обучения без stale data

Без OS (~8000 Raise записей/итерацию):
  - Freshness window = 5 → 5 × 8000 = 40000 → ограничено buffer 30K
  - 5 train steps × batch 64 = 320 записей используется
  - То же качество, больше свежих данных

================================================================================
ИЗМЕНЁННЫЕ ФАЙЛЫ
================================================================================

1. config.yaml
   - pg_memory_size: 300000

2. src/utils/config.py
   - _DEFAULTS: pg_memory_size=30000, pg_train_steps_per_iteration=5,
     pg_freshness_window=5

3. src/core/deep_cfr.py
   - PolicyGradientMemory: _iterations, add(iteration), sample(min_iteration)
   - DeepCFRAgent.__init__: pg_train_steps_per_iteration, pg_freshness_window
   - train_sizing_network: min_iteration, freshness filter
   - Все pg_memory.add: iteration=iteration

4. src/training/train.py
   - _train_pg_and_log: N вызовов train_sizing_network, avg метрики,
     Train/PGSteps scalar

================================================================================
СОВМЕСТИМОСТЬ
================================================================================

- pg_memory.add(iteration=-1) — backward compatible, старый код работает
- pg_memory.sample(min_iteration=None) — backward compatible
- train_sizing_network(min_iteration=None) — backward compatible
- pg_freshness_window=0 — отключает freshness filter
- Старые чекпоинты: не затронуты (pg_memory не сохраняется)
