# Runtime Search Root Policy Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Выбрать legal response-to-bet action через MMDS поверх прямого compact blueprint prior и общих lockstep rollouts.

**Architecture:** `spots.py` добавит deterministic response-to-bet state с identity action set. `policy.py` создаёт blocker-aware particles, вызывает S3 rollout engine, переводит raw EV в безразмерные values и применяет `mmds_update`; public result хранит action, policies, EV и diagnostics.

**Tech Stack:** Python 3.11, NumPy, локальный `pokers`, pytest.

**Spec:** `planning/solver_reports/solver.md`

## Global Constraints

- Не менять training loop, checkpoint format, `src/core/action_space.py` или `pokers/`.
- S4 поддерживает только root actions `fold/call/raise_1pot/all_in`, являющиеся legal compact slots без prior mapping.
- `value_scale = max(pot, effective_remaining, epsilon)` до root action; не использовать global training reward scale.
- `rho=uniform` остаётся default MMDS magnet policy.
- Uniform particles обязаны отражаться в diagnostics и не являются solver-valid EV evidence до S6.
- При incomplete rollout solver возвращает blueprint fallback с diagnostic flag, а не применяет MMDS к частичным значениям.

---

## File Structure

- `src/runtime_search/spots.py` — constructed response-to-bet spot.
- `src/runtime_search/policy.py` — config, `SearchDecision` и root policy update.
- `src/runtime_search/__init__.py` — public exports S4.
- `tests/test_runtime_search_policy.py` — identity mapping, MMDS policy, fallback и diagnostics.
- `planning/solver_reports/005-root-policy.md` — отчёт S4.

### Task 1: Identity response-to-bet spot

**Files:**
- Modify: `src/runtime_search/spots.py`
- Modify: `tests/test_runtime_search_spots.py`

**Interfaces:**
- Produces: `build_response_to_bet_spot() -> pkrs.State`, где player `1` ходит против ставки player `0`.

- [x] **Step 1: Написать падающий test**

Проверить `current_player == 1`, а legal fixed slots равны `FOLD`, `CALL`, `RAISE_POT`, `ALL_IN`; `CHECK` и `RAISE_HALF_POT` illegal.

- [x] **Step 2: Запустить test и увидеть отсутствие builder**

Run: `python -m pytest tests/test_runtime_search_spots.py::test_response_to_bet_spot_has_identity_compact_action_set -q`

Expected: ImportError или AttributeError для builder.

- [x] **Step 3: Реализовать spot**

Переиспользовать known cards from constructed turn, создать turn state с initial stack `200.0`, `pot_chips=[10.0] * 6`, `bet_chips=[10.0, 0.0, ...]`, `pot=70.0`, `current_player=1`, `last_stage_action=[Raise, None, ...]`. Такой pot-size raise `70.0` меньше remaining `160.0` после call и является отдельным legal action.

- [x] **Step 4: Запустить test**

Run: `python -m pytest tests/test_runtime_search_spots.py::test_response_to_bet_spot_has_identity_compact_action_set -q`

Expected: PASS.

### Task 2: Root policy update

**Files:**
- Create: `src/runtime_search/policy.py`
- Modify: `src/runtime_search/__init__.py`
- Create: `tests/test_runtime_search_policy.py`

**Interfaces:**
- Produces: `RuntimeSearchConfig`, `SearchDecision`, `choose_action(state, hero_id, blueprint, config, rng) -> SearchDecision`.
- `SearchDecision` содержит `action`, `action_label`, `root_blueprint_policy`, `root_search_policy`, `raw_ev`, `mc_values`, `eta_times_values`, `belief_particles`, `latency_seconds`, `diagnostic_flags`.

- [x] **Step 1: Написать падающие S4 tests**

На response-to-bet spot и deterministic uniform batch blueprint проверить: output action legal; policies имеют длину `NUM_ACTIONS`, zero вне identity slots и сумму `1.0`; с `eta=0`, `floor=0` search policy равна blueprint policy; value scale даёт finite `mc_values`; diagnostics содержит uniform belief flag. Отдельно mock continuation policy с zero mass для legal action, ожидая blueprint fallback и `rollout_incomplete` flag.

- [x] **Step 2: Запустить tests и увидеть отсутствие policy module**

Run: `python -m pytest tests/test_runtime_search_policy.py -q`

Expected: collection error для `src.runtime_search.policy`.

- [x] **Step 3: Реализовать config/result и choose_action**

Собрать particles для всех other players и недостающих board cards по street. Получить compact prior от `blueprint.probabilities`, оставить response identity slots и их legal mask, вызвать `evaluate_root_actions`. Только при complete rollout разделить raw EV на local value scale, вызвать `mmds_update`, сэмплировать root slot и вернуть resolved action. При incomplete evaluation вернуть prior action и raw diagnostics без MMDS.

- [x] **Step 4: Запустить S4 tests**

Run: `python -m pytest tests/test_runtime_search_policy.py -q`

Expected: PASS.

### Task 3: Отчёт и целевая проверка

**Files:**
- Create: `planning/solver_reports/005-root-policy.md`

- [x] **Step 1: Запустить целевые S1-S4 tests**

Run: `python -m pytest tests/test_frozen_blueprint_policy.py tests/test_paired_evaluation.py tests/test_mmds.py tests/test_runtime_search_spots.py tests/test_runtime_search_beliefs.py tests/test_runtime_search_rollouts.py tests/test_runtime_search_policy.py -q`

Expected: exit code `0`.

- [x] **Step 2: Заполнить отчёт S4**

Зафиксировать identity set, value scaling, default uniform rho, fallback semantics и observed tests.
