# 005 - root policy

**Статус:** IMPLEMENTED
**Дата:** 2026-08-28
**Код:** `src/runtime_search/spots.py`, `src/runtime_search/policy.py`, `src/runtime_search/__init__.py`, `tests/test_runtime_search_spots.py`, `tests/test_runtime_search_policy.py`
**Команды проверки:** `python -m pytest tests/test_frozen_blueprint_policy.py tests/test_paired_evaluation.py tests/test_mmds.py tests/test_runtime_search_spots.py tests/test_runtime_search_beliefs.py tests/test_runtime_search_rollouts.py tests/test_runtime_search_policy.py -q`

## Что сделано

- Добавлен deterministic turn response-to-bet spot, в котором legal compact set ровно `fold/call/raise_1pot/all_in`.
- Добавлены `RuntimeSearchConfig`, `SearchDecision` и public `choose_action`.
- Root prior берётся напрямую из blueprint на identity slots, без mapping на новые сайзинги.
- S3 raw EV нормализуется на `max(pot, effective_remaining, epsilon)`, затем обновляется через MMDS с default uniform `rho`.
- `SearchDecision` содержит action, обе policy, raw/scaled values, `eta * values`, latency, particle count и diagnostic flags.
- Если хотя бы один root action не завершил все particles, MMDS не выполняется: solver выбирает из blueprint prior с флагом `rollout_incomplete`.

## Почему так

Response-to-bet узлы отделяют проверку MMDS и rollout mechanics от confounder-а распределения prior на новый sizing grid. Локальный value scale делает `eta` сопоставимым в разных pot/stack размерах. Fallback не превращает частичный Monte Carlo signal в произвольное root update.

## Результаты

Целевая проверка S1-S4: `24 passed in 4.17s`.

Покрыты identity action set, mask, `eta=0`, finite scaled values, uniform belief diagnostic и fallback при incomplete rollout.

## Риски

S4 диагностирует механику, но не доказывает EV improvement: particles пока uniform и отмечены `uniform_beliefs_not_solver_valid`. Serena/Pyright не резолвит imports между новыми `src/runtime_search` файлами как namespace package, однако Python imports и тесты работают.

## Следующий шаг

S5: turn/river validation. Нужны замеры частот/entropy blueprint на настоящем checkpoint и constructed spot diagnostics; положительный paired EV до S6 не будет kill-gate.
