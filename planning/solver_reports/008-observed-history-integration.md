# 008 - observed history integration

**Статус:** IMPLEMENTED
**Дата:** 2026-08-28
**Код:** `src/runtime_search/history.py`, `src/evaluation/paired_harness.py`, `src/runtime_search/__init__.py`, `tests/test_runtime_search_history.py`, `tests/test_paired_evaluation.py`
**Команды проверки:** `python -m pytest tests/test_frozen_blueprint_policy.py tests/test_paired_evaluation.py tests/test_mmds.py tests/test_runtime_search_spots.py tests/test_runtime_search_beliefs.py tests/test_runtime_search_rollouts.py tests/test_runtime_search_policy.py tests/test_runtime_search_diagnostics.py tests/test_runtime_search_history.py -q`; `python -m compileall -q src tools/run_turn_river_diagnostics.py`

## Что сделано

- Добавлен `ActionHistoryRecorder`: до `apply_action` он сохраняет compact opponent action как `ObservedDecision`.
- Добавлен `RuntimeSearchPolicy`: на подходящем turn/river response root он автоматически берёт записанную history и вызывает S6 `choose_action`; на остальных узлах играет frozen blueprint.
- `paired_harness` поддерживает необязательные hooks `set_hero_id`, `reset_hand`, `observe_action` и вызывает их только если policy их реализует. Existing policies остаются совместимы.
- Некомпактный opponent raise не подменяется ближайшим slot: history помечается неполной, а runtime policy передаёт `None`, что приводит к безопасному blueprint-only fallback.
- Training loop, модели и checkpoint format не менялись.

## Почему так

Reach-weighted posterior требует state до фактически сыгранных действий оппонента. Evaluation harness — корректная точка записи, потому что видит state и action до перехода к следующему state. Неявное восстановление history из финального `State` было бы ненадёжным: `pokers.State` не хранит полный action log.

## Результаты

- `36 passed in 6.22s` для целевого S1-S6 плюс history-integration набора.
- `compileall` завершился с exit code `0`.
- Проверены compact action recording, отказ от noncompact raise, blueprint fallback без history и вызов lifecycle/history hooks из paired evaluation.

## Следующий шаг

Собрать отдельный paired-EV runner с `RuntimeSearchPolicy` и `FrozenBlueprintPolicy`, запустить baseline vs runtime policy на двух seed-ах. Это уже meaningful evaluation: history формируется автоматически, а reach-weighted ESS остаётся доступен в `SearchDecision`.
