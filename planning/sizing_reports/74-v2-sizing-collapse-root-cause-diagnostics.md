# Bug #74-v2: sizing collapse root-cause diagnostics

## Статус: DIAGNOSTICS_IMPLEMENTED (ожидает diagnostic run iter200-300)

## Контекст

#74 mix=0.15 подтвердил PASS на iter200, но FAIL на устойчивость iter300-400:
- `iter100`: OK, raise_freq=54.8%, win_rate=44.0
- `iter200`: OK, raise_freq=34.8%, но `best_sizing_anchor_dist` уже `0.10: 2918/2993`
- `iter300`: WARN, raise_freq=16.9%, `best_sizing_anchor_dist={"0.10":2993}`
- `iter400`: WARN, raise_freq=11.2%, sizing коллапс в small bucket

mix=0.15 — это отсрочка self-suppressing collapse, а не фикс.

Вместо слепого #75 (player-aware backup) сначала нужен диагностический прогон,
чтобы доказательно определить, где замыкается collapse:
- в action-Q bootstrap (opponent vs hero next-node)
- в sizing/OS path (кто кормит 0.10: hero OS, full traversing, opponent)

## Что сделано (диагностический слой, без изменения поведения обучения)

### 1. `QValueBuffer.next_is_hero` (src/core/deep_cfr.py)
- Добавлен массив `_next_is_hero` (float32).
- `add()` принимает `next_is_hero=False` по умолчанию.
- `sample()` возвращает 8-й элемент — `_next_is_hero`.
- При записи в traversing/opponent нодах:
  `next_is_hero = (new_state.current_player == traversing_player)`.
  Терминалы всегда `False`.

### 2. Checkpoint save (src/core/deep_cfr.py)
- `q_buffer_next_is_hero` сохраняется в `_build_checkpoint`.
- `sizing_path_diag` сохраняется в чекпоинт.

### 3. Role-sliced action-Q diagnostics (tools/checkpoint_tools.py)
- В `run_action_q_model_diagnostics` добавлен блок `bootstrap_by_next_role`.
- Разрез по ролям: `hero`, `opponent` (non-terminal), `terminal`.
- Для каждого action метрики: `bootstrap_value`, `target`, `next_strategy_raise_mass`.
- Если `q_buffer_next_is_hero` отсутствует в чекпоинте → `bootstrap_by_next_role: null`.
- Вывод в консоль и JSON full_report.

### 4. Sizing path diagnostics (src/core/deep_cfr.py + tools/checkpoint_tools.py)
- Три счетчика `sizing_path_diag`:
  `hero_os_raise` — OS traversing node raise anchors.
  `hero_full_raise` — full traversing node raise anchors.
  `opponent_raise` — opponent branch raise anchors.
- Сохраняются в checkpoint и выводятся в full_report.

### 5. Тесты
- 57/57 PASS в `test_hybrid_outcome_sampling.py`.
- 91/94 total pass (3 failure в `test_training_regressions.py` предсуществующие).

## Что НЕ менялось

- `q_bootstrap_policy_uniform_mix` остался 0.15.
- Поведение обучения не изменилось — только сбор диагностики.
- `_apply_q_bootstrap_policy_mix` не трогался.
- #75 (player-aware backup) не внедрён.

## Как использовать

1. Запустить training с теми же параметрами #74 до iter200-300.
2. Сгенерировать full_reports для iter100, iter200, iter300:

```
python tools/checkpoint_tools.py --checkpoint models/multi_new/multi_checkpoint_iter_200.pt --mode full --num-states 2993
```

3. Смотреть новые поля в JSON:
   - `action_q_model.bootstrap_by_next_role.hero.raise.bootstrap_value`
   - `action_q_model.bootstrap_by_next_role.opponent.raise.bootstrap_value`
   - `sizing_path_diag.hero_os_raise`
   - `sizing_path_diag.hero_full_raise`
   - `sizing_path_diag.opponent_raise`

## Правила интерпретации

### Case A: opponent bootstrap портит raise
Признаки:
- `bootstrap_by_next_role.opponent.raise.bootstrap_value.mean` сильно ниже hero
- `opponent.raise.next_strategy_raise_mass.mean` низкий/нестабильный
- Collapse начинается до/на iter200

→ Делать #75 player-aware backup: mix только когда `next_is_hero=True`.

### Case B: action bootstrap нормальный, sizing path схлопнулся
Признаки:
- `bootstrap_by_next_role` hero/opponent не показывает сильного перекоса
- `sizing_path_diag.hero_os_raise` или `opponent_raise` почти весь в 0.10
- `sizing_q_buffer.source0` концентрируется в small anchors

→ Копать OS sizing path: почему opponent branch использует `strategy_sizing_net` напрямую.

### Case C: оба слоя портятся
Признаки: Case A + Case B одновременно.

→ Сначала #75 (action-Q влияет на sizing-Q bootstrap).
→ Затем отдельный фикс OS sizing path.

## Файлы

- Изменены: `src/core/deep_cfr.py`, `tools/checkpoint_tools.py`, `tests/test_hybrid_outcome_sampling.py`
- Конфиг: `config.yaml` (без изменений, mix=0.15 как в #74)

## Дата: 2026-06-07
