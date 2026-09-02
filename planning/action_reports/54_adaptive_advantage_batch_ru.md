═══════════════════════════════════════════════════════════════════
  BUG #54 — Адаптивный batch для advantage-сети при OS
  Дата: 02.06.2026
  Статус: Исправлено
  Файлы: src/core/deep_cfr.py, src/training/train.py
═══════════════════════════════════════════════════════════════════

ОПИСАНИЕ
────────

После включения Outcome Sampling (OS) размер AdvantageBuffer падает
до ~970–1010 сэмплов (раньше был ~1200+). При advantage_batch_size = 1024
обучение жёстко пропускается:

  if n < batch_size:
      return 0

Результат: Loss/Advantage = 0, Train/AdvantageSteps = 0 — сеть не обучается.

Причина: buf.sample(n) уже берёт все данные, batch_size нужен только
для нарезки на mini-batch. Нет причин требовать n >= batch_size.

═══════════════════════════════════════════════════════════════════
  ПУНКТ 1 — Адаптивный early-return
  Файл: src/core/deep_cfr.py:1814-1817
═══════════════════════════════════════════════════════════════════

  Было:
    if n < batch_size:
        self.last_advantage_train_steps = 0
        return 0

  Стало:
    if n == 0:
        self.last_advantage_train_steps = 0
        self.last_advantage_effective_batch_size = 0
        return 0.0

  Edge-case: если буфер пуст — по-прежнему пропускаем.
  Если n > 0 — обучение запускается всегда.

═══════════════════════════════════════════════════════════════════
  ПУНКТ 2 — Адаптивный mini-batch
  Файл: src/core/deep_cfr.py:1819-1820, 1843-1844
═══════════════════════════════════════════════════════════════════

  Добавлено:
    effective_batch_size = min(batch_size, n)
    self.last_advantage_effective_batch_size = effective_batch_size

  Цикл:
    for start in range(0, len(all_states), effective_batch_size):
        sel = ep_perm[start:start + effective_batch_size]

  Сценарии:
    n = 1003, batch_size = 1024 → effective = 1003 → 1 train step
    n = 3000, batch_size = 1024 → effective = 1024 → 3 train steps
    n = 0                    → early-return 0.0

═══════════════════════════════════════════════════════════════════
  ПУНКТ 3 — Сброс при samples is None
  Файл: src/core/deep_cfr.py:1823-1826
═══════════════════════════════════════════════════════════════════

  Было:
    if samples is None:
        self.last_advantage_train_steps = 0
        return 0

  Стало:
    if samples is None:
        self.last_advantage_train_steps = 0
        self.last_advantage_effective_batch_size = 0
        return 0.0

  При ошибке семплирования effective_batch тоже сбрасывается.

═══════════════════════════════════════════════════════════════════
  ПУНКТ 4 — Инициализация атрибута
  Файл: src/core/deep_cfr.py:756
═══════════════════════════════════════════════════════════════════

  Добавлено в __init__:
    self.last_advantage_effective_batch_size = 0

  Рядом с:
    self.last_advantage_train_steps = 0

═══════════════════════════════════════════════════════════════════
  ПУНКТ 5 — Логирование
  Файл: src/training/train.py:1306-1314
═══════════════════════════════════════════════════════════════════

  Print:
    "Advantage loss: 0.0423 (2 steps, batch=1003, 1.2s)"

  TensorBoard:
    Train/AdvantageSteps              — как было
    Train/AdvantageEffectiveBatchSize — новый скаляр

═══════════════════════════════════════════════════════════════════
  ИТОГ
═══════════════════════════════════════════════════════════════════

  После фикса:
    • Train/AdvantageSteps > 0 при MemoryAdvantage > 0
    • при MemoryAdvantage ≈ 1000 и batch_size=1024 → 1-2 шага
    • Loss/Advantage больше не обрывается в 0.0
    • Train/AdvantageEffectiveBatchSize показывает фактический размер

═══════════════════════════════════════════════════════════════════
  СВЯЗАННЫЕ БАГИ
═══════════════════════════════════════════════════════════════════

  Bug #53 — DCFR+ дистилляция стратегии: per-sample MSE
  Bug #52 — утилита проверки чекпоинтов
  Bug #51 — оптимизация потоков данных, сетей и буферов
═══════════════════════════════════════════════════════════════════
