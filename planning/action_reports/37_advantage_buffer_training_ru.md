================================================================================
БАГ-РЕПОРТ #37: AdvantageBuffer — reservoir недоиспользуется, advantage-net недообучается
================================================================================

ТИП: Performance / Data loss
ДАТА: 2026-05-15
СТАТУС: Исправлено
СВЯЗАННЫЕ: #36 (Hybrid OS) — должен идти вместе

================================================================================
ОПИСАНИЕ
================================================================================

Проблема 1: AdvantageBuffer (reservoir sampling) имеет capacity=16384.
При 400 traversals на multiway postflop за итерацию генерируется ~40-100K+
advantage samples. Reservoir хранит uniform 16K подвыборку, остальные ~84K
теряются. Не biased потеря — равномерная, но 84% данных не используются.

Проблема 2: train_advantage_network_multi(...) обучает advantage-net только
на min(batch_size * 3, len) = 768 samples за epoch, даже если в buffer
лежит 16K. Это 4.7% данных. 95.3% собранной информации игнорируется.

Итого: двойная потеря данных
  - reservoir: 84% generated samples отброшены
  - training: 95% buffer samples не видит advantage-net
  - combined: advantage-net видит ~0.7% от всех generated samples

================================================================================
КОРНЕВАЯ ПРИЧИНА
================================================================================

1. advantage_memory_size: 16384 — слишком мало для 400 traversals на 6-max.
   Резервуар заполняется полностью, старые записи текущей итерации
   заменяются с uniform probability.

2. train_advantage_network_multi использует subset:
   n_select = min(batch_size * 3, len(all_states))
   При batch_size=256 → n_select=768.
   Это было сделано для скорости, но при дорогом traversal это расточительно.

================================================================================
КОНСЕНСУС (GLM + GPT)
================================================================================

1. AdvantageBuffer — reservoir sampling, НЕ circular. (Подтверждено кодом.)
2. capacity=16384 слишком мал для реального объёма данных.
3. batch_size * 3 subset заставляет advantage-net недополучать информацию.
4. Увеличение buffer без изменения training loop почти бесполезно.
5. Потери reservoir — "равномерные, но потери" (не "нет потерь").
6. epochs=1 — правильный дефолт. epochs > 1 не guaranteed overfitting,
   но повышать только экспериментально.
7. Полный проход по buffer + epochs=1 — не overfitting, потому что regrets
   обновляются каждую итерацию.

================================================================================
ИСПРАВЛЕНИЯ
================================================================================

1. advantage_memory_size: 16384 → 300000
   Файлы: config.yaml, src/utils/config.py (_DEFAULTS)

2. train_advantage_network_multi — полный проход mini-batches
   Заменено:
     for epoch in range(epochs):
         n_select = min(batch_size * 3, len(all_states))
         sel = np.random.choice(len(all_states), n_select, replace=False)
         # один optimizer step на subset
   На:
     for epoch in range(epochs):
         ep_perm = np.random.permutation(len(all_states))
         for start in range(0, len(all_states), batch_size):
             sel = ep_perm[start:start + batch_size]
             # optimizer step на каждый mini-batch

3. Default epochs: 3 → 1
   Файлы: config.yaml (advantage_epochs), train_advantage_network_multi signature
   Return: total_loss / max(steps, 1) вместо total_loss / epochs

4. Диагностика
   - self.last_advantage_train_steps — количество optimizer steps за вызов
   - Time/AdvantageTrain — время обучения advantage-net
   - Train/AdvantageSteps — steps в TensorBoard
   - Консоль: "Advantage loss: X.XXX (Y steps, Z.Zs)"

================================================================================
ИЗМЕНЁННЫЕ ФАЙЛЫ
================================================================================

1. config.yaml
   - advantage_memory_size: 16384 → 300000
   - advantage_epochs: 3 → 1

2. src/utils/config.py
   - _DEFAULTS['advantage_memory_size']: 300000

3. src/core/deep_cfr.py
   - DeepCFRAgent.__init__: self.last_advantage_train_steps = 0
   - train_advantage_network_multi: полный проход mini-batches, epochs=1,
     return total_loss/max(steps,1), self.last_advantage_train_steps = steps

4. src/training/train.py
   - Замер времени advantage обучения (time.time())
   - writer.add_scalar('Train/AdvantageSteps', ...)
   - writer.add_scalar('Time/AdvantageTrain', ...)
   - Консоль: steps + время в выводе

================================================================================
ОЖИДАЕМЫЙ ЭФФЕКТ
================================================================================

При 300K buffer + полный проход:
  - ~1170 optimizer steps за iteration (300K / 256 batch)
  - advantage-net видит 100% собранных данных (через reservoir)
  - Training time: ~30-60 сек на CPU (меньше чем traversal ~5 мин)
  - Качество advantage predictions должно улучшиться

С Hybrid OS (#36):
  - OS уменьшит generated samples → buffer заполняется медленнее
  - Training быстрее (меньше steps)
  - Механизм масштабируется автоматически

================================================================================
РИСКИ И МИТИГАЦИЯ
================================================================================

1. Overfitting на одну итерацию: НЕТ. Regrets обновляются каждую итерацию,
   epochs=1 — один честный проход. Target_net frozen, bootstrap работает.

2. Training time: МОЖЕТ БЫТЬ. 30-60 сек при 300K. Мониторить Time/AdvantageTrain.
   Если bottleneck — уменьшить traversals_per_iteration или включить Hybrid OS.

3. DCFR логика: НЕ НАРУШЕНА. Discounting через iter_tensors веса как раньше.
   Больше данных = лучше оценка.

================================================================================
СОВМЕСТИМОСТЬ
================================================================================

- Старые чекпоинты: не затронуты (buffer не сохраняется)
- advantage_net architecture: не меняется
- train_advantage_network (legacy): не затронут (NotImplementedError)
- Все существующие training loops: не ломаются
