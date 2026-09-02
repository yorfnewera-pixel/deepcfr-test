================================================================================
БАГ-РЕПОРТ #36: Hybrid Outcome Sampling на postflop multiway
================================================================================

ТИП: Новая фича / Performance fix
ДАТА: 2026-05-15
СТАТУС: План согласован, к реализации
ЗАВИСИМОСТИ: #35 (Q-Network VR) — обязательно

================================================================================
ОПИСАНИЕ
================================================================================

Проблема: CFR traversal раздувается до ~5 минут на итерацию при 400 traversals.
Причина: на postflop multiway External Sampling раскрывает ВСЕ legal actions
traversing player'а, а дерево на flop/turn/river с 3+ игроками огромно.

Решение: Hybrid External/Outcome Sampling:
  - Preflop: External Sampling как раньше (full traversal для hero)
  - Postflop heads-up: External Sampling (пока)
  - Postflop multiway (>=2 active opponents): Outcome Sampling

При Outcome Sampling на traversing node сэмплируется ОДНО действие через
sampling policy mu, а regrets для ВСЕХ legal actions восстанавливаются через
Q-baseline + importance correction:

  value_hat[a]           = Q(s, a)                                    для unsampled
  value_hat[a_sampled]    = Q(s, a_sampled) + w * (v_sampled - Q(s, a_sampled))
  w                       = min(1/mu(a_sampled), importance_weight_clip)
  regret_hat[a]           = value_hat[a] - EV_hat
  EV_hat                  = sum(sigma[a] * value_hat[a])

Несмещённость при ЛЮБОМ Q (control variate). Хорошее Q → low variance.

================================================================================
КЛЮЧЕВЫЕ АРХИТЕКТУРНЫЕ РЕШЕНИЯ (консенсус GLM + GPT)
================================================================================

1. ДЕФОЛТ: hybrid_os_enabled = false
   Новый режим выключен по умолчанию. Включать только явно в экспериментальном
   конфиге. Не ломает существующие рабочие процессы.

2. WARMUP: hybrid_os_q_warmup_iterations = 50, hybrid_os_min_q_buffer_size = 5000
   OS не включается, пока Q-сеть не обучена минимально. Причина: не bias
   (оценщик несмещён при любом Q), а variance — плохой Q увеличивает шум.

3. Q-BASELINE: используем СУЩЕСТВУЮЩИЙ q_net
   НЕ добавляем отдельную q_self_net. НЕ используем Q=0 baseline.
   Обоснование:
   - Оценщик несмещён при любом Q → математическая корректность сохранена
   - Существующий q_net уже работает на неконсистентных данных (opponent
     perspective → traversing reward). Traversing-perspective данные УЛУЧШАЮТ
     консистентность, не ухудшают.
   - Q=0 + clipping даёт систематический bias для unsampled actions
   Обязательная диагностика: abs(Q(s, a_sampled) - v_sampled).
   Если качество плохое — отдельная сеть на следующем этапе.

4. OPPONENT NODES: не меняются
   В hybrid OS-режиме opponent nodes остаются sampled/control-variate.
   Возвращаемое значение — continuation estimate для OS-estimator.
   Это нормальная смесь, не конфликт.

5. PG FLOOR: hybrid_os_pg_min_raise_sample_prob = 0.15
   Постоянный floor, НЕ anneal. Иначе PG по sizing голодает при anneal
   min_raise_sample_prob до 0.05 — слишком мало записей в pg_memory.

6. REGRET CLIPPING: optional
   hybrid_os_regret_clip = null по умолчанию. Логировать os_regret_clip_count.
   Включать только если диагностика покажет explosion.

7. CONFIG PREFIX: hybrid_os_
   Все новые параметры с единым префиксом для логической группировки.

================================================================================
ИЗМЕНЯЕМЫЕ ФАЙЛЫ
================================================================================

1. src/core/deep_cfr.py
   - Новые приватные методы DeepCFRAgent:
     * _is_postflop(state) — int(state.stage) > 0
     * _count_active_opponents(state, traversing_player)
     * _should_use_outcome_sampling(state, traversing_player) — с warmup проверкой
     * _compute_regret_matching_strategy(advantages, legal_mask)
     * _build_sampling_policy(strategy, legal_mask, iteration)
     * _cfr_traverse_multi_outcome_node(...) — OS-ветка traversing node
   - cfr_traverse_multi: ветвление на OS/ES для traversing player
   - reset_traversal_stats: новые OS counters
   - get_traversal_stats: новые OS metrics
   - __init__: загрузка hybrid_os_* config параметров

2. src/utils/config.py
   - Новые дефолты с hybrid_os_ prefix (все с безопасными дефолтами)

3. config.yaml
   - Опционально: секция hybrid_os_* для экспериментов

4. src/training/train.py
   - Логирование OS-метрик в TensorBoard и консоль

5. tests/test_hybrid_outcome_sampling.py (НОВЫЙ)
   - Тесты: переключение, sampling policy, OS traversal, warmup, preflop ES

================================================================================
КОНФИГУРАЦИЯ
================================================================================

Параметр                                | Дефолт  | Описание
----------------------------------------|---------|------------------------------------------
hybrid_os_enabled                       | false   | Включить hybrid OS (БЕЗОПАСНЫЙ ДЕФОЛТ)
hybrid_os_min_active_opponents          | 2       | Мин. активных оппонентов для OS
hybrid_os_epsilon_start                 | 0.25    | Начальная exploration probability
hybrid_os_epsilon_end                   | 0.05    | Конечная exploration probability
hybrid_os_epsilon_decay_iterations      | 1000    | Итерации anneal epsilon
hybrid_os_importance_weight_clip        | 20.0    | Max importance weight
hybrid_os_sizing_advantage_clip         | 10.0    | Clip PG sizing advantage
hybrid_os_regret_clip                   | null    | Optional clip для regrets
hybrid_os_pg_min_raise_sample_prob      | 0.15    | Floor для Raise sampling (НЕ anneal!)
hybrid_os_q_warmup_iterations           | 50      | OS не вкл. до этой итерации
hybrid_os_min_q_buffer_size             | 5000    | OS не вкл. пока Q-buffer < размера

================================================================================
ДИАГНОСТИКА (новые counters)
================================================================================

Счётчики узлов:
- outcome_sampling_nodes
- outcome_sampling_traversing_nodes
- postflop_multiway_nodes
- sampled_raise_nodes
- sampled_non_raise_nodes

Importance weights:
- importance_weight_sum, importance_weight_max, importance_weight_count
- (mean вычислять как sum/count)

Q-correction:
- q_correction_abs_sum, q_correction_count
- q_baseline_abs_error_sum, q_baseline_abs_error_max, q_baseline_abs_error_count

Regrets:
- os_regret_abs_sum, os_regret_sq_sum, os_regret_count
- os_max_abs_regret
- (mean и std вычислять из sum/sq_sum/count)
- os_regret_clip_count

PG:
- sizing_advantage_sum, sizing_advantage_sq_sum, sizing_advantage_count
- sampled_raise_ratio

================================================================================
ФОРМУЛЫ
================================================================================

Стратегия:
  σ(a) = regret_matching(advantages, legal_mask)

Sampling policy:
  μ(a) = (1 - ε) * σ(a) + ε * UniformLegal(a)

Raise floor (НЕ anneal):
  μ(Raise) = max(μ(Raise), hybrid_os_pg_min_raise_sample_prob)

Перенормализация:
  μ(a) = μ(a) / Σ_{a∈legal} μ(a)

Importance weight:
  w = min(1 / μ(a_sampled), hybrid_os_importance_weight_clip)

Value estimator:
  value_hat[a]           = Q(s, a)                                    (unsampled)
  value_hat[a_sampled]   = Q(s, a_sampled) + w * (v_sampled - Q(s, a_sampled))

EV:
  EV_hat = Σ_a σ(a) * value_hat[a]

Regret:
  regret_hat[a] = value_hat[a] - EV_hat

Optional regret clip:
  Если hybrid_os_regret_clip задан:
    regret_hat[a] = clip(regret_hat[a], -clip_val, clip_val)

PG для ставок (только если sampled action == Raise):
  sizing_advantage = regret_hat[Raise]
  sizing_advantage = clip(sizing_advantage, -sizing_advantage_clip, sizing_advantage_clip)
  pg_memory.add(state, z_raw, sizing_advantage, sampled_bet_size)

================================================================================
Q-BUFFER UPDATE на traversing OS nodes
================================================================================

При terminal:
  next_state  = zeros_like(encoded_state)
  next_mask   = zeros
  is_terminal = True
  reward      = terminal reward traversing_player

Иначе:
  next_state  = encode_state(new_state, new_state.current_player)  — см. багфикс #35
  next_mask   = get_legal_action_mask(new_state)
  is_terminal = False
  reward      = 0.0

================================================================================
ПОРЯДОК РЕАЛИЗАЦИИ
================================================================================

Шаг 1: Config flags + helper-функции (без изменения обхода)
Шаг 2: Counters и логирование (без изменения стратегической логики)
Шаг 3: _cfr_traverse_multi_outcome_node(...) + ветвление в cfr_traverse_multi
Шаг 4: Q-buffer update для traversing OS node
Шаг 5: PG-сигнал только для sampled Raise
Шаг 6: Быстрые тесты / smoke
Шаг 7: Сравнение диагностики до/после

================================================================================
ОГРАНИЧЕНИЯ (НЕ делать в этом PR)
================================================================================

1. Не переписывать PokerNetwork
2. Не переводить ставки в дискретные бины
3. Не удалять PG sizing
4. Не делать size-aware Q-сеть
5. Не менять suit isomorphism
6. Не переписывать весь CFR traversal
7. Не удалять старый External Sampling код
8. Не добавлять отдельную q_self_net
9. Не использовать Q=0 baseline

================================================================================
КРИТЕРИИ УСПЕХА
================================================================================

1. Preflop работает как раньше (full traversal hero actions)
2. Postflop heads-up работает как раньше (full traversal hero actions)
3. Postflop multiway НЕ раскрывает все hero actions (one sampled action)
4. Iteration time заметно падает (цель: 2-3x ускорение на multiway)
5. advantage_buffer получает full-vector regrets (не только sampled action)
6. pg_memory получает данные только при sampled Raise
7. Q-buffer получает transitions из OS ветки
8. Нет NaN/inf в regrets, PG advantages, Q corrections
9. Importance weights ограничены clipping'ом
10. Диагностика показывает долю OS узлов и sampled Raise
11. Warmup корректно задерживает включение OS
12. Q-baseline quality: abs(Q(s,a_sampled) - v_sampled) мониторится

================================================================================
СВЯЗАННЫЕ БАГ-РЕПОРТЫ
================================================================================

- #35: Q-Network VR — базовая Q-сеть, на которую опирается OS estimator
- #22: Value Head Audit — аудит value head
- #17: DCFR+ Discounting — существующая схема discounting
