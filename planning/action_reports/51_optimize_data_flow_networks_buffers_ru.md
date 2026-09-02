═══════════════════════════════════════════════════════════════════
  BUG #51 — Оптимизация потоков данных: Q-сети, sizing-сети,
  стратеджи-буферы и маршрутизация нейросетей
  Дата: 01.06.2026
  Статус: Готов к реализации
  Согласовано: анализ + план GPT проверены
═══════════════════════════════════════════════════════════════════

ОПИСАНИЕ
────────
Полная ревизия всех Q-сетей, sizing-сетей, strategy-буферов
и потоков данных между ними выявила 10 проблем:
- мёртвые буферы (не наполняются, не читаются)
- несогласованные записи (сырые vs сглаженные таргеты)
- сломанные режимы тренировки (NotImplementedError)
- скрытый AttributeError (метод не существует на классе)
- блокировка Q-ready через некорректный подсчёт coverage
- сеть, которая обучается, но не используется для решений
- target-сеть, замороженная навечно
- утечка памяти при отключённом sizing_q


═══════════════════════════════════════════════════════════════════
  ПУНКТ 1 — КРИТИЧЕСКИЙ
  SizingQBuffer.anchor_counts считает только probe-сэмплы
  → блокирует Q-ready
═══════════════════════════════════════════════════════════════════

  Файл: src/core/deep_cfr.py:550-559

  def anchor_counts(self, num_anchors):
      counts = np.zeros(num_anchors, dtype=np.int64)
      if self._size == 0:
          return counts
      probe_mask = self._sources[:self._size] == 1          # ← только source==1
      anchor_ids = self._anchor_indices[:self._size][probe_mask]
      for idx in anchor_ids:
          if 0 <= idx < num_anchors:
              counts[idx] += 1
      return counts

  Проблема: coverage считается ТОЛЬКО по probe-сэмплам (source=1).
  Регулярные сэмплы (source=0) игнорируются. При probe_prob,
  аннилящейся до 0.05, _sizing_q_has_anchor_coverage() может
  никогда не вернуть True → sizing-подсистема навсегда
  остаётся в фазе 0 (uniform sizing по анкорам).

  Исправление: считать по всем сэмплам с валидным anchor_idx:

  def anchor_counts(self, num_anchors):
      counts = np.zeros(num_anchors, dtype=np.int64)
      if self._size == 0:
          return counts
      anchor_ids = self._anchor_indices[:self._size]
      valid_mask = anchor_ids >= 0
      for idx in anchor_ids[valid_mask]:
          if 0 <= idx < num_anchors:
              counts[idx] += 1
      return counts


═══════════════════════════════════════════════════════════════════
  ПУНКТ 2 — КРИТИЧЕСКИЙ
  4 из 5 режимов тренировки сломаны — NotImplementedError
═══════════════════════════════════════════════════════════════════

  Файл: src/training/train.py

  Все режимы КРОМЕ train_self_play_multi вызывают LEGACY-метод
  train_advantage_network() (src/core/deep_cfr.py:1833), который
  бросает NotImplementedError.

  Затронутые режимы и строки:

  ┌──────────────────────────────┬────────────────────────────────┬─────────┐
  │ Режим                        │ train.py строка                │ Статус  │
  ├──────────────────────────────┼────────────────────────────────┼─────────┤
  │ train_deep_cfr               │ 463: train_advantage_network() │ СЛОМАН  │
  │ continue_training            │ 591: train_advantage_network() │ СЛОМАН  │
  │ train_against_checkpoint     │ 732: train_advantage_network() │ СЛОМАН  │
  │ train_with_mixed_checkpoints │ 1025: train_advantage_network()│ СЛОМАН  │
  │ train_self_play_multi        │ 1225: train_advantage_network_ │ OK      │
  │                              │       multi()                  │         │
  └──────────────────────────────┴────────────────────────────────┴─────────┘

  Исправление: заменить все 4 вызова на train_advantage_network_multi().

  Пример (train_deep_cfr, строка 463):
    БЫЛО:  adv_loss = agent.train_advantage_network()
    СТАЛО: adv_loss = agent.train_advantage_network_multi()
    Аналогично в остальных трёх режимах.


═══════════════════════════════════════════════════════════════════
  ПУНКТ 3 — ВЫСОКИЙ
  Несогласованная запись в sizing_strategy_buffer
═══════════════════════════════════════════════════════════════════

  Файлы: src/training/train.py, src/core/deep_cfr.py

  В cfr_traverse_multi (ES и OS пути) запись в sizing_strategy_buffer
  проходит через цепочку:
    _smooth_sizing_target_by_bucket → _sparsify_sizing_target_by_buckets → add

  В _cfr_traverse_with_opponents (train.py:300-301) пишутся СЫРЫЕ веса:
    agent.sizing_strategy_buffer.add(_state_arr, slot_weights_np, iteration)

  Это создаёт смесь сглаженных и сырых таргетов для обучения
  train_strategy_sizing_anchor_network.

  Исправление в _cfr_traverse_with_opponents (train.py ~строка 300):
    БЫЛО:
      agent.sizing_strategy_buffer.add(_state_arr, slot_weights_np, iteration)
    СТАЛО:
      target = agent._smooth_sizing_target_by_bucket(state, encoded_state, slot_weights_np, iteration)
      target = agent._sparsify_sizing_target_by_buckets(target)
      agent.sizing_strategy_buffer.add(_state_arr, target, iteration)


═══════════════════════════════════════════════════════════════════
  ПУНКТ 4 — ВЫСОКИЙ
  q_buffer не пополняется в _cfr_traverse_with_opponents
  → q_net не учится в checkpoint/mixed режимах
═══════════════════════════════════════════════════════════════════

  Файл: src/training/train.py, функция _cfr_traverse_with_opponents

  Уточнение: проблема ТОЛЬКО в режимах train_against_checkpoint
  и train_with_mixed_checkpoints. Режим train_self_play_multi
  использует cfr_traverse_multi, где q_buffer пишется нормально.

  В _cfr_traverse_with_opponents opponent-ветка (строка 316-349):
    opponent_agent.choose_action(state) → apply_action → recurse → return

  Нет записи в q_buffer, нет Q-baseline variance reduction.

  Исправление: после opponent-ветки, если use_q_baseline=True,
  добавить запись в q_buffer по аналогии с cfr_traverse_multi:
    - encode_state(new_state, traversing_player) для next_q_encoded
    - encode_state(new_state, new_state.current_player) для next_policy_encoded
    - get_legal_action_mask(new_state) для next_mask
    - add в q_buffer с правильными полями


═══════════════════════════════════════════════════════════════════
  ПУНКТ 5 — СРЕДНИЙ
  advantage_sizing_net обучается, но не используется для решений
═══════════════════════════════════════════════════════════════════

  Файл: src/core/deep_cfr.py

  train_sizing_anchor_network() обучает advantage_sizing_net
  предсказывать Q-derived advantages каждую итерацию (5 шагов).
  Но во время cfr_traverse_multi / choose_action её выходы НИГДЕ
  не используются — sizing всегда идёт через sizing_q_net напрямую.

  Решение: встроить advantage_sizing_net как fallback для sizing
  когда sizing_q_net ещё не ready (фаза 0). Это даст сети смысл
  и ускорит sizing-вычисления (1 forward вместо batch на 15 анкоров).

  Вариант реализации:
    if self._sizing_q_is_ready():
        q_vals = self._evaluate_sizing_q_for_sizes(...)   # Q-путь
        slot_weights_np = compute_sizing_heat_weights(...)
    else:
        slot_logits = self.advantage_sizing_net(state_tensor)  # fallback
        slot_weights_np = F.softmax(slot_logits, dim=1).cpu().numpy()
        # или regret_matching_anchors на logits


═══════════════════════════════════════════════════════════════════
  ПУНКТ 6 — СРЕДНИЙ
  sizing_target_net заморожена навечно
═══════════════════════════════════════════════════════════════════

  Файл: src/core/deep_cfr.py:714-721

  self.sizing_target_net = SizingAnchorNet(...)       # создание
  self.sizing_target_net.load_state_dict(...)         # копия при init
  self.sizing_target_net.eval()                       # заморозка

  В отличие от target_net (обновляется после каждого
  train_advantage_network_multi, строка 1906), sizing_target_net
  НИКОГДА не обновляется во время обучения.

  Решение: удалить sizing_target_net из __init__, _build_checkpoint,
  _load_checkpoint, save_model. Она нигде не используется для
  forward-пассов — только save/load создают иллюзию работы.


═══════════════════════════════════════════════════════════════════
  ПУНКТ 7 — СРЕДНИЙ
  SizingAdvantageBuffer — мёртвый буфер
═══════════════════════════════════════════════════════════════════

  Файл: src/core/deep_cfr.py:663, 2293-2296, 2427-2433
        src/training/train.py:1160

  Буфер создаётся, сохраняется в чекпоинты, загружается,
  очищается при clean_buffers — но НИ РАЗУ не наполняется
  (ни одного вызова .add() во всём коде).

  Решение: удалить sizing_advantage_buffer из:
    - __init__ (строка 663)
    - _build_checkpoint (строки 2293-2296)
    - _load_checkpoint (строки 2427-2433)
    - prepare_iteration / clean_buffers (train.py:1160)
    - логирования (train.py:403, 1332)


═══════════════════════════════════════════════════════════════════
  ПУНКТ 8 — НИЗКИЙ
  _sample_training_bet_size_eval — скрытый AttributeError
═══════════════════════════════════════════════════════════════════

  Файл: src/core/deep_cfr.py:1334-1346

  z_raw = self.advantage_sizing_net._squash_inv(...)

  SizingAnchorNet НЕ ИМЕЕТ метода _squash_inv (это метод старого
  SizingNetwork). Тест test_sizing_q_regret.py:123 подтверждает:
    assert not hasattr(net, '_squash_inv')

  Сейчас недостижим (вызывается только из deprecated cfr_traverse),
  но код висит в классе.

  Решение: удалить _sample_training_bet_size_eval и связанный
  _sample_bet_size_eval (строка 1079-1084) — оба не вызываются
  из нового кода.


═══════════════════════════════════════════════════════════════════
  ПУНКТ 9 — НИЗКИЙ
  sizing_q_buffer выделяется при sizing_q_enabled=False
═══════════════════════════════════════════════════════════════════

  Файл: src/core/deep_cfr.py:767-768 (блок else)

  self.sizing_q_buffer = SizingQBuffer(
      cfg_get('sizing_q_buffer_size', 100000), state_dim=input_size
  )

  Буфер ~50+ MB выделяется впустую.

  Решение: self.sizing_q_buffer = None в блоке else.
  Проверить все обращения к sizing_q_buffer на наличие
  проверки sizing_q_enabled перед len()/sample().


═══════════════════════════════════════════════════════════════════
  ПУНКТ 10 — НИЗКИЙ
  Legacy-классы на удаление (после функциональных фиксов)
═══════════════════════════════════════════════════════════════════

  Файлы: src/core/deep_cfr.py, src/core/model.py

  Мёртвый код, не блокирует обучение, можно чистить потом:
    - PrioritizedMemory  (deep_cfr.py:132-196)  — не используется
    - StrategyMemory     (deep_cfr.py:200-231)  — не используется
    - SizingNetwork      (model.py:108-217)     — floating Gaussian, не исп.
    - cfr_traverse       (deep_cfr.py:1426-1430) — deprecated
    - train_advantage_network (deep_cfr.py:1833-1838) — deprecated


═══════════════════════════════════════════════════════════════════
  ЧТО НЕ ТРОГАТЬ — ЖИВЫЕ КОМПОНЕНТЫ
═══════════════════════════════════════════════════════════════════

  - sizing_q_net / SizingQNetwork        — основной sizing-Q
  - strategy_sizing_net / StrategySizingNet — инференс + оппоненты
  - sizing_strategy_buffer               — таргеты для дистилляции
  - strategy_buffer / StrategyBuffer     — DCFR+ policy данные
  - advantage_buffer / AdvantageBuffer   — CFR regret данные
  - q_buffer / QValueBuffer              — variance reduction replay
  - advantage_net / PokerNetwork         — action-сеть
  - strategy_net / PokerNetwork          — action-policy инференс
  - q_net / QValueNetwork                — VR на opponent-узлах
  - target_net / PokerNetwork            — стабилизация advantage


═══════════════════════════════════════════════════════════════════
  ПРИОРИТЕТ РЕАЛИЗАЦИИ
═══════════════════════════════════════════════════════════════════

  1. [КРИТ] anchor_counts — считать все valid anchor_idx
  2. [КРИТ] 4 режима — train_advantage_network → _multi
  3. [ВЫС]  sizing_strategy_buffer — унифицировать smoothing
  4. [ВЫС]  q_buffer в _cfr_traverse_with_opponents
  5. [СРД]  advantage_sizing_net — fallback при !Q-ready
  6. [СРД]  sizing_target_net — удалить
  7. [СРД]  SizingAdvantageBuffer — удалить
  8. [НИЗ]  _sample_training_bet_size_eval — удалить deprecated путь
  9. [НИЗ]  sizing_q_buffer = None при sizing_q_enabled=False
  10.[НИЗ]  Legacy-классы — потом


═══════════════════════════════════════════════════════════════════
  СВЯЗАННЫЕ БАГИ
═══════════════════════════════════════════════════════════════════

  Bug #48  — continuous Q/regret sizing (sizing_q_net, AWR)
  Bug #49  — Hybrid Outcome Sampling + sizing architecture split
  Bug #50  — фиксированная сетка сайзингов (15 анкеров)
  Bug #43  — sizing void (предок sizing-проблем)
  Bug #44  — conditional size advantage
  Bug #46  — detach sizing isolation
  Bug #47  — k-sample within-state advantage + sizing split
═══════════════════════════════════════════════════════════════════
