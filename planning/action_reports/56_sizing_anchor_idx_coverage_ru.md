═══════════════════════════════════════════════════════════════════
  BUG #56 — sizing_anchor_idx всегда -1: _sizing_q_is_ready() никогда не True
  Дата: 03.06.2026
  Статус: Исправлено
  Файлы: src/core/deep_cfr.py, src/training/train.py
═══════════════════════════════════════════════════════════════════

ОПИСАНИЕ
────────

Во всех путях добавления Q-семплов (cfr_traverse_multi,
_cfr_traverse_multi_outcome_node, _cfr_traverse_with_opponents)
переменная sizing_anchor_idx инициализируется в -1 и никогда не
перезаписывается, кроме probe-случая (где она намеренно -1).

Проблема:
  - _hierarchical_sizing() возвращает float sampled_bet_size, но не индекс
  - np.random.choice(slot_sizes_np, p=...) тоже возвращает float, не индекс
  - SizingQBuffer.anchor_counts() считает только anchor_idx >= 0
  - _sizing_q_has_anchor_coverage() всегда возвращает False
  - _sizing_q_is_ready() всегда возвращает False

Последствия:
  - sizing_strategy_buffer никогда не пополняется
  - train_strategy_sizing_anchor_network() всегда возвращает 0
  - Loss/SizingStrategy{A,B}ucket = 0 в TensorBoard
  - Memory/SizingStrategy отсутствует в логах
  - _hierarchical_sizing() работает через regret-matching fallback вместо Q
  - train_sizing_anchor_network() (advantage sizing PG) не обучается

Единственное что работало: train_sizing_q_network() — не проверяет
_sizing_q_is_ready(). Поэтому Q-буфер наполнялся (Memory/SizingQBuffer = 50000),
но Q считалась неготовой из-за отсутствия anchor coverage.

═══════════════════════════════════════════════════════════════════
  ПУНКТ 1 — _hierarchical_sizing: возвращать anchor_idx
  Файл: src/core/deep_cfr.py, строка 1378-1380
═══════════════════════════════════════════════════════════════════

Было:
  sampled_bet_size = float(np.random.choice(self.anchors, p=full_probs))
  return sampled_bet_size, full_probs.astype(np.float32)

Стало:
  sampled_anchor_idx = int(np.random.choice(self.num_anchors, p=full_probs))
  sampled_bet_size = float(self.anchors[sampled_anchor_idx])
  return sampled_bet_size, full_probs.astype(np.float32), sampled_anchor_idx

Принцип "sample index first" гарантирует согласованность индекса и размера
одним вызовом RNG.

═══════════════════════════════════════════════════════════════════
  ПУНКТ 2 — cfr_traverse_multi: распаковка 3 значений
  Файл: src/core/deep_cfr.py, строка 1631
═══════════════════════════════════════════════════════════════════

Было:
  sampled_bet_size, slot_weights_np = self._hierarchical_sizing(state_tensor, iteration)

Стало:
  sampled_bet_size, slot_weights_np, sizing_anchor_idx = self._hierarchical_sizing(state_tensor, iteration)

Теперь sizing_anchor_idx (строка 1618) перезаписывается валидным индексом.
Вызов _add_sizing_q_sample на строке 1692 уже передаёт anchor_idx=sizing_anchor_idx.

═══════════════════════════════════════════════════════════════════
  ПУНКТ 3 — _cfr_traverse_multi_outcome_node: инициализация sizing_anchor_idx
  Файл: src/core/deep_cfr.py, строка 1433
═══════════════════════════════════════════════════════════════════

Добавлена строка:
  sizing_anchor_idx = -1

Рядом с существующими:
  sampled_bet_size = None
  slot_weights_np = None

═══════════════════════════════════════════════════════════════════
  ПУНКТ 4 — _cfr_traverse_multi_outcome_node: fallback _hierarchical_sizing
  Файл: src/core/deep_cfr.py, строка 1462
═══════════════════════════════════════════════════════════════════

Было:
  _, slot_weights_np = self._hierarchical_sizing(state_tensor, iteration)

Стало:
  _, slot_weights_np, _ = self._hierarchical_sizing(state_tensor, iteration)

(Индекс не нужен — Q не ready, используется regret matching fallback)

═══════════════════════════════════════════════════════════════════
  ПУНКТ 5 — _cfr_traverse_multi_outcome_node: Q-ready путь index-first
  Файл: src/core/deep_cfr.py, строка 1464-1466
═══════════════════════════════════════════════════════════════════

Было:
  slot_sizes_np = np.array(self.anchors, dtype=np.float32)
  sampled_bet_size = float(np.random.choice(slot_sizes_np, p=slot_weights_np))

Стало:
  sizing_anchor_idx = int(np.random.choice(self.num_anchors, p=slot_weights_np))
  sampled_bet_size = float(self.anchors[sizing_anchor_idx])

Удалена неиспользуемая переменная slot_sizes_np.

═══════════════════════════════════════════════════════════════════
  ПУНКТ 6 — _cfr_traverse_multi_outcome_node: передача anchor_idx в Q-sample
  Файл: src/core/deep_cfr.py, строка 1551-1552
═══════════════════════════════════════════════════════════════════

Было:
  self._add_sizing_q_sample(_state_arr, sampled_bet_size, sizing_target, iteration, is_probe=False)

Стало:
  self._add_sizing_q_sample(_state_arr, sampled_bet_size, sizing_target, iteration, is_probe=False,
                             anchor_idx=sizing_anchor_idx)

═══════════════════════════════════════════════════════════════════
  ПУНКТ 7 — train.py: инициализация sizing_anchor_idx
  Файл: src/training/train.py, строка 214
═══════════════════════════════════════════════════════════════════

Добавлена строка:
  sizing_anchor_idx = -1

═══════════════════════════════════════════════════════════════════
  ПУНКТ 8 — train.py: index-first sampling в _cfr_traverse_with_opponents
  Файл: src/training/train.py, строка 236-237
═══════════════════════════════════════════════════════════════════

Было:
  sampled_bet_size = float(np.random.choice(slot_sizes_np, p=slot_weights_np))

Стало:
  sizing_anchor_idx = int(np.random.choice(len(agent.anchors), p=slot_weights_np))
  sampled_bet_size = float(agent.anchors[sizing_anchor_idx])

═══════════════════════════════════════════════════════════════════
  ПУНКТ 9 — train.py: передача anchor_idx в Q-sample
  Файл: src/training/train.py, строка 316-317
═══════════════════════════════════════════════════════════════════

Было:
  agent._add_sizing_q_sample(_state_arr, sampled_bet_size, sizing_target, iteration)

Стало:
  agent._add_sizing_q_sample(_state_arr, sampled_bet_size, sizing_target, iteration,
                              anchor_idx=sizing_anchor_idx)

═══════════════════════════════════════════════════════════════════
  ОЖИДАЕМЫЙ ЭФФЕКТ
═══════════════════════════════════════════════════════════════════

1. anchor_counts() начнёт возвращать ненулевые значения для анкеров,
   через которые проходит sizing-сэмплинг.
2. После ~1920 рейз-семплов (15 анкеров × 128) _sizing_q_has_anchor_coverage()
   вернёт True.
3. _sizing_q_is_ready() станет True.
4. sizing_strategy_buffer начнёт наполняться (строка 1682 в deep_cfr.py,
   строка 306 в train.py).
5. Через ~10 итераций буфер накопит ≥ 128 записей.
6. Loss/SizingStrategyAnchor, Loss/SizingStrategyBucket,
   Loss/StrategySizingAnchor появятся в TensorBoard.
7. Memory/SizingStrategy появится в TensorBoard (логируется из
   _train_pg_and_log при loss != 0).
8. _hierarchical_sizing() переключится с regret-matching на Q-guided sizing.

Probe-семплы остаются с anchor_idx=-1 — корректно, поскольку probe может
выдавать размеры вне фиксированной сетки анкеров.
