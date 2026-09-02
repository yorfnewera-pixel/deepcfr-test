═══════════════════════════════════════════════════════════════════
  BUG #58 — Sizing mismatch: Q учится на raw anchor, а среда применяет effective
  Дата: 03.06.2026
  Статус: Исправлено
  Файлы: src/core/deep_cfr.py, src/training/train.py
═══════════════════════════════════════════════════════════════════

ОПИСАНИЕ
────────

При выборе анкера (например, 3.00 pots) алгоритм пишет regret и Q-target
для raw anchor, но среда pokers может молча применить другое значение:

  anchor 3.00 → remaining stack = 80 chips, pot = 50
  → additional = min(150, 80) = 80 chips → effective = 1.60 pots

Или снизу:

  anchor 0.10 → pot = 10, min_raise = 2 chips
  → additional = max(1, 2) = 2 chips → effective = 0.20 pots

В обоих случаях Q-буфер получает:
  - norm_size = (3.00 - 0.1) / (3.0 - 0.1) = 1.0  (raw anchor)
  - anchor_idx = индекс анкера 3.00

Но реально сыграно 1.60 pots (all-in) или 0.20 pots (min-raise).

Последствия:
  - Q-сеть учится ассоциировать outcome с неверным размером ставки
  - advantage_sizing_net получает искажённые Q-targets
  - Политика может предпочитать анкеры, которые всегда урезаются средой

═══════════════════════════════════════════════════════════════════
  РЕШЕНИЕ
═══════════════════════════════════════════════════════════════════

Архитектура не расширяется. 15 анкеров и 3 бакета без изменений.
Добавляется единый resolver эффективного сайзинга.

Шаг 1 — _get_min_raise_increment(state)
  Файл: src/core/deep_cfr.py (новый метод перед action_type_to_pokers_action)

  Вынесена логика расчёта min_raise из action_type_to_pokers_action:

    def _get_min_raise_increment(self, state):
        if hasattr(state, 'last_raise_increment') and state.last_raise_increment:
            return max(1.0, float(state.last_raise_increment))
        if hasattr(state, 'bb') and state.bb:
            return max(1.0, float(state.bb))
        return 1.0

Шаг 2 — _resolve_effective_sizing(state, anchor)
  Файл: src/core/deep_cfr.py (новый метод)

  Возвращает (effective_multiplier, effective_chips, kind):

    pot = max(1.0, float(state.pot))
    mult = clamp(anchor, min_bet_size, max_bet_size)
    raw = pot * mult

    remaining = stake - call_amount
    min_raise = self._get_min_raise_increment(state)

    if remaining < min_raise:
        effective = remaining  # all-in (или 0 если нечего ставить)
    else:
        effective = clamp(raw, min_raise, remaining)

    effective_mult = effective / pot
    kind ∈ {"NORMAL", "MIN_RAISE", "ALL_IN"}

Шаг 3 — action_type_to_pokers_action использует resolver
  Файл: src/core/deep_cfr.py

  Было: скрытый clamp внутри метода
  Стало: _, additional, _ = self._resolve_effective_sizing(state, anchor)

Шаг 4 — _add_sizing_q_sample: effective_multiplier + nearest anchor_idx
  Файлы: src/core/deep_cfr.py (2 места), src/training/train.py (1 место)

  Было:
    self._add_sizing_q_sample(state, sampled_bet_size, target,
                              anchor_idx=sizing_anchor_idx)

  Стало:
    eff_mult, _, _ = self._resolve_effective_sizing(state, sampled_bet_size)
    eff_idx = int(np.argmin(np.abs(self.anchors_arr - eff_mult)))
    self._add_sizing_q_sample(state, eff_mult, target, anchor_idx=eff_idx)

═══════════════════════════════════════════════════════════════════
  ЧТО НЕ МЕНЯЕТСЯ
═══════════════════════════════════════════════════════════════════

- Архитектура: SizingAnchorNet(15), StrategySizingNet(15), SizingQNetwork
- train_sizing_anchor_network — raw anchors (Q интерполирует)
- train_strategy_network — bet_size не используется в обучении
- SizingStrategyBuffer — хранит 15-dim probs, не scalar
- Чекпоинты — обратная совместимость сохранена

═══════════════════════════════════════════════════════════════════
  ИЗВЕСТНЫЙ КОМПРОМИСС
═══════════════════════════════════════════════════════════════════

Дублирование анкеров при all-in остаётся:
  2.00, 2.25, 2.50, 2.75, 3.00 → один effective (all-in)

Но Q теперь учится на фактическом множителе, а не на выдуманном raw-анкоре.
Политика естественно сместится к рабочему диапазону через интерполяцию Q.
