═══════════════════════════════════════════════════════════════════
  BUG #57 — _sizing_q_is_ready() O(50000) на каждый sizing-узел
  Дата: 03.06.2026
  Статус: Исправлено
  Файлы: src/core/deep_cfr.py, src/training/train.py
═══════════════════════════════════════════════════════════════════

ОПИСАНИЕ
────────

После фикса #56 _sizing_q_is_ready() начал реально возвращать True.
Но на каждом sizing-узле героя (~400 раз/итерацию) он вызывал
anchor_counts() — O(size_of_buffer) = O(50k+) полный скан. Это
объясняет рост времени итерации с 12-15с до 45-50с.

Помимо readiness-check, три других источника лишней работы на узел:

1. np.array(self.anchors, dtype=np.float32) — 3 аллокации на узел
   при наличии self.anchors_arr (строка 582)

2. advantage_sizing_net forward в _hierarchical_sizing() — всегда
   вызывается (строка 1350), но при Q-ready результат выбрасывается

3. CPU/GPU roundtrip — .cpu().numpy() на каждом узле (в Q-ready
   режиме больше не нужен)

═══ РЕШЕНИЕ: 4 оптимизации без изменения качества ═══

═══════════════════════════════════════════════════════════════════
  ПУНКТ 1 — SizingQBuffer: инкрементальные счётчики + версия
  Файл: src/core/deep_cfr.py, класс SizingQBuffer
═══════════════════════════════════════════════════════════════════

__init__ (строка 484-495):

  Добавлены поля:
    self._anchor_counts = np.zeros(num_anchors, dtype=np.int64)
    self._version = 0

  Параметр num_anchors добавлен в сигнатуру с default=15.

add() (строка 497-513):

  При перезаписи позиции в заполненном ring-buffer:
    - вытесняемый anchor_idx ≥ 0 → декремент счётчика
  При добавлении:
    - anchor_idx ≥ 0 → инкремент счётчика
  self._version += 1 при каждом add()

anchor_counts() (строка ~526):

  Было:  O(size) полный скан _anchor_indices
  Стало: O(num_anchors) копия _anchor_counts[:num_anchors]

═══════════════════════════════════════════════════════════════════
  ПУНКТ 2 — _sizing_q_is_ready(): version-based кэш
  Файл: src/core/deep_cfr.py, строка 1300-1318
═══════════════════════════════════════════════════════════════════

Добавлен O(1) быстрый путь:

  buf_ver = self.sizing_q_buffer._version
  if hasattr(self, '_sqr_version') and self._sqr_version == buf_ver:
      return self._sqr_cached

Пересчёт только при изменении версии буфера (т.е. при add()).
Версия не зависит от _size — корректно работает и после заполнения
ring-buffer (когда _size перестаёт расти).

═══════════════════════════════════════════════════════════════════
  ПУНКТ 3 — self.anchors_arr вместо np.array(self.anchors, ...)
  Файлы: src/core/deep_cfr.py, src/training/train.py
═══════════════════════════════════════════════════════════════════

Замены (4 места):

  deep_cfr.py:1349  _hierarchical_sizing          → self.anchors_arr
  deep_cfr.py:1447  _cfr_traverse_multi_outcome   → self.anchors_arr
  deep_cfr.py:1635  cfr_traverse_multi            → self.anchors_arr
  train.py:217      _cfr_traverse_with_opponents  → agent.anchors_arr

self.anchors_arr инициализирован на строка 582.

═══════════════════════════════════════════════════════════════════
  ПУНКТ 4 — _hierarchical_sizing: пропуск advantage_sizing_net при Q-ready
  Файл: src/core/deep_cfr.py, строка 1340-1391
═══════════════════════════════════════════════════════════════════

Реструктурирован порядок ветвления:

  Было:
    1. advantage_sizing_net forward (всегда)
    2. .cpu().numpy() (всегда)
    3. if Q-ready:  Q-guided (forward выброшен)
       else:        regret matching

  Стало:
    1. if Q-ready:  Q-guided (advantage_sizing_net не вызывается)
       else:        advantage_sizing_net forward → regret matching

При Q-ready экономия: 1 forward pass + 2 .cpu().numpy() на узел.

═══════════════════════════════════════════════════════════════════
  ПУНКТ 5 — Восстановление _anchor_counts/_version из чекпоинта
  Файл: src/core/deep_cfr.py, строка 2428-2433
═══════════════════════════════════════════════════════════════════

После загрузки sizing_q_buffer из чекпоинта добавлена перестройка:

  buf._anchor_counts[:] = 0
  for i in range(n):
      aidx = int(buf._anchor_indices[i])
      if aidx >= 0:
          buf._anchor_counts[aidx] += 1
  buf._version = n

Без этого загруженный чекпоинт имел бы пустые _anchor_counts
и _version=0 — кэш readiness был бы инвалидирован и пересчитан
один раз, но _anchor_counts были бы нулевыми → coverage failed бы.

═══════════════════════════════════════════════════════════════════
  ОЖИДАЕМЫЙ ЭФФЕКТ
═══════════════════════════════════════════════════════════════════

На sizing-узел героя:

  До:
    _sizing_q_is_ready():      O(50000) скан anchor_counts
    advantage_sizing_net:      1 forward + 2 .cpu().numpy()
    np.array(self.anchors):    1-3 аллокации

  После (Q-ready):
    _sizing_q_is_ready():      O(1) version check + кэш
    advantage_sizing_net:      0 forward (пропущен)
    np.array(self.anchors):    0 аллокаций (anchors_arr)

  После (Q-not-ready):
    _sizing_q_is_ready():      O(1) version check + кэш
    advantage_sizing_net:      1 forward (нужен для regret matching)
    np.array(self.anchors):    0 аллокаций (anchors_arr)

~400 sizing-узлов × O(50000) → ~400 × O(1): ожидаемое снижение
времени итерации до исходных 12-15с.
