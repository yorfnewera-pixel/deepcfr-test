# 007 - reach-weighted beliefs

**Статус:** IMPLEMENTED
**Дата:** 2026-08-28
**Код:** `src/runtime_search/beliefs.py`, `src/runtime_search/policy.py`, `src/runtime_search/__init__.py`, `tests/test_runtime_search_beliefs.py`, `tests/test_runtime_search_policy.py`, `tests/test_runtime_search_diagnostics.py`
**Команды проверки:** `python -m pytest tests/test_frozen_blueprint_policy.py tests/test_paired_evaluation.py tests/test_mmds.py tests/test_runtime_search_spots.py tests/test_runtime_search_beliefs.py tests/test_runtime_search_rollouts.py tests/test_runtime_search_policy.py tests/test_runtime_search_diagnostics.py -q`; `python -m compileall -q src tools/run_turn_river_diagnostics.py`

## Что сделано

- Добавлены `ObservedDecision` и `ReachWeightedBeliefs`.
- `sample_reach_weighted_particles` создаёт blocker-aware proposal particles, вычисляет likelihood observed opponent actions через frozen blueprint в log-space, нормализует weights и выполняет resampling.
- Для каждого observed snapshot реальные hidden hands заменяются particle hands; `state.deck` не читается. Hero hand и public cards history валидируются против root condition.
- Считаются `ESS` и `ESS / number_of_particles`; resampled particles помечаются `is_solver_valid=True`.
- `choose_action` принимает `observed_decisions`; `SearchDecision` содержит `belief_ess` и `belief_ess_ratio`.
- При отсутствии history или при ESS ниже `RuntimeSearchConfig.belief_min_ess_ratio` rollout/MMDS не запускаются и возвращается blueprint prior с явным diagnostic flag.

## Почему так

Uniform posterior после S5 нельзя трактовать как solver-valid EV evidence. Reach likelihood действий оппонентов даёт приближение `P_pi(H | h_i)`, а ESS показывает, не сконцентрировался ли вес на слишком малом числе proposal particles. Blueprint fallback лучше частичного или вырожденного Monte Carlo update: он сохраняет уже обученную стратегию вместо создания ложного преимущества solver-а.

## Результаты

- `32 passed in 7.39s` для целевого S1-S6 набора.
- `compileall` завершился с exit code `0`.
- Tests покрывают reproducible reach resampling, отсутствие дубликатов/утечки карт, invalid history slot, mismatch hero hand, successful root path, fallback без history, fallback при low ESS и existing rollout-incomplete behavior.

## Риски и границы

- Автоматический collector history в training loop намеренно не добавлялся. Внешний runtime caller обязан передать корректную последовательность `ObservedDecision`; без неё solver остаётся blueprint-only.
- Это приближение posterior через particles, а не точный enumeration. Низкий ESS не делает результат «плохим» молча — он отключает MMDS.
- Paired EV после S6 уже разрешён как следующий содержательный gate, но в этом этапе не запускался: сначала нужен integration path, который формирует реальные observed history snapshots для solver call.
- Serena/Pyright всё ещё не резолвит sibling namespace import в `src/runtime_search`; реальные Python imports, pytest и compileall проходят.

## Следующий шаг

S7 или небольшой integration adapter перед ним: передавать реальные action snapshots из evaluation/runtime loop в `choose_action`, затем запустить paired EV на двух seed-ах против blueprint. Расширенная sizing grid не нужна до появления этого честного evaluation path.
