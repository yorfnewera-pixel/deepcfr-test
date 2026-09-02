═══════════════════════════════════════════════════════════════════
  BUG #53-v2 — Гиперпараметры: буферы, batch_size, epochs из конфига
  Дата: 02.06.2026 (обновлено)
  Статус: Исправлено
  Файлы: src/core/deep_cfr.py, config.yaml
  Связан: Bug #53 (per-sample MSE + объём обучения)
═══════════════════════════════════════════════════════════════════

ОПИСАНИЕ
────────

Гиперпараметры обучения (batch_size, epochs) были захардкожены
в сигнатурах функций. После #53 значения изменились, но не читались
из config.yaml. Buffer sizes были 300K против 1M в статье.

Компромиссный подход: не копировать все параметры статьи (10М эпизодов,
10K traversals), а приблизить ключевые — буферы, batch_size, epochs.


═══════════════════════════════════════════════════════════════════
  ПУНКТ 1 — config.yaml: новые ключи и значения
═══════════════════════════════════════════════════════════════════

  Файл: config.yaml

  ┌──────────────────────────────┬───────────┬─────────────────────┐
  │ Ключ                         │ Было      │ Стало               │
  ├──────────────────────────────┼───────────┼─────────────────────┤
  │ advantage_memory_size        │ 300000    │ 1000000             │
  │ strategy_memory_size         │ 300000    │ 1000000             │
  │ memory_size                  │ 300000    │ 1000000             │
  │ q_buffer_size                │ 300000    │ 1000000             │
  │ advantage_epochs             │ 1         │ 2                   │
  │ strategy_epochs              │ 20        │ 50                  │
  │ advantage_batch_size         │ (новый)   │ 1024                │
  │ strategy_batch_size          │ (новый)   │ 1024                │
  │ hybrid_os_epsilon_start      │ 0.25(def) │ 0.6                 │
  │ hybrid_os_epsilon_end        │ 0.05(def) │ 0.6                 │
  │ sizing_min_prob_start        │ 0.05(def) │ 0.015               │
  │ sizing_min_prob_end          │ 0.0125(d) │ 0.003               │
  └──────────────────────────────┴───────────┴─────────────────────┘


═══════════════════════════════════════════════════════════════════
  ПУНКТ 2 — deep_cfr.py __init__: чтение из конфига
═══════════════════════════════════════════════════════════════════

  Файл: src/core/deep_cfr.py:556-559

  Добавлено:
    self.advantage_batch_size = cfg_get('advantage_batch_size', 256)
    self.strategy_batch_size = cfg_get('strategy_batch_size', 128)
    self.advantage_epochs = cfg_get('advantage_epochs', 1)
    self.strategy_epochs = cfg_get('strategy_epochs', 3)


═══════════════════════════════════════════════════════════════════
  ПУНКТ 3 — deep_cfr.py: функции используют self.*
═══════════════════════════════════════════════════════════════════

  train_strategy_network (строка 1906):
    def train_strategy_network(self, batch_size=None, epochs=None):
        if batch_size is None:
            batch_size = self.strategy_batch_size
        if epochs is None:
            epochs = self.strategy_epochs

  train_advantage_network_multi (строка 1767):
    def train_advantage_network_multi(self, batch_size=None, epochs=None):
        if batch_size is None:
            batch_size = self.advantage_batch_size
        if epochs is None:
            epochs = self.advantage_epochs

  Все 5 точек вызова в train.py используют дефолтные аргументы
  → автоматически берут значения из конфига.


═══════════════════════════════════════════════════════════════════
  ПУНКТ 4 — hybrid_os_epsilon: мёртвый ключ → живые ключи
═══════════════════════════════════════════════════════════════════

  Файл: config.yaml:25-28

  Проблема: `hybrid_os_epsilon: 0.6` в конфиге — мёртвый ключ.
  Код читает `hybrid_os_epsilon_start` и `hybrid_os_epsilon_end`
  (deep_cfr.py:739-741), дефолты: 0.25 → 0.05.

  Фактический epsilon: 0.25 → 0.05 (аннилинг), вместо 0.6.

  Было:
    hybrid_os_epsilon: 0.6           # не читается кодом

  Стало:
    hybrid_os_epsilon_start: 0.6
    hybrid_os_epsilon_end: 0.6
    hybrid_os_epsilon_decay_iterations: 1

  Эффект: epsilon = 0.6 на всех итерациях.
  OS sampling policy: 60% uniform + 40% strategy.
  Exploration на постфлопе увеличен с 25% → 60%.


═══════════════════════════════════════════════════════════════════
  ПУНКТ 5 — СНИЖЕН sizing_min_prob: 5%→1.5% на анкер
═══════════════════════════════════════════════════════════════════

  Файл: config.yaml:53-55

  Проблема: sizing_min_prob_start = 5% на каждый из 15 анкеров
  → 15 × 5% = 75% массы — uniform floor, только 25% от Q-сигнала.
  Sizing на ранних итерациях на 75% случаен.

  Было (дефолты в коде, не в конфиге):
    sizing_min_prob_start: 0.05
    sizing_min_prob_end: 0.0125

  Стало:
    sizing_min_prob_start: 0.015    → 15 × 1.5% = 22.5% floor
    sizing_min_prob_end: 0.003      → 15 × 0.3% = 4.5% floor
    sizing_min_prob_decay_iterations: 1000

  Bucket floor (0.15→0.03) оставлен без изменений — он про
  покрытие Small/Medium/Large групп, не про равномерность анкеров.


═══════════════════════════════════════════════════════════════════
  СРАВНЕНИЕ СО СТАТЬЁЙ VR-DeepDCFR+
═══════════════════════════════════════════════════════════════════

  ┌───────────────────────────────┬─────────────┬────────────┬─────────┐
  │ Параметр                      │ Статья      │ У нас      │ Разрыв  │
  ├───────────────────────────────┼─────────────┼────────────┼─────────┤
  │ advantage_buffer_size         │ 1,000,000   │ 1,000,000  │ 1× ✅   │
  │ ave_policy_buffer_size        │ 1,000,000   │ 1,000,000  │ 1× ✅   │
  │ ave_policy_batch_size         │ 2,048       │ 1,024      │ 2×      │
  │ ave_policy_train_steps        │ 5,000       │ 50         │ 100×    │
  │ advantage_batch_size          │ 2,048       │ 1,024      │ 2×      │
  │ advantage_train_steps         │ 750         │ 2          │ 375×    │
  │ num_traversals                │ 10,000      │ ~200       │ 50×     │
  │ learning_rate                 │ 0.001       │ 1e-4       │ 10×     │
  │ hidden_size                   │ 64          │ 256        │ 0.25×   │
  │ epsilon                       │ 0.6         │ 0.6        │ 1× ✅   │
  │ alpha                         │ 2           │ 2          │ 1× ✅   │
  │ gamma                         │ 2           │ 2          │ 1× ✅   │
  └───────────────────────────────┴─────────────┴────────────┴─────────┘

  Компромисс: буферы доведены до статьи. Batch_size ≈ статье.
  Train steps и traversals меньше из-за ограничений времени обучения.
  LR и hidden_size — инженерный выбор для poker (больше состояний).


═══════════════════════════════════════════════════════════════════
  СВЯЗАННЫЕ БАГИ
═══════════════════════════════════════════════════════════════════

  Bug #53  — per-sample MSE + вес (t/T)^γ
  Bug #52  — утилита проверки чекпоинтов
  Bug #51  — оптимизация потоков данных, сетей и буферов
═══════════════════════════════════════════════════════════════════
