# Runtime Search Lockstep Rollouts Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Оценивать raw EV каждого root action на общих belief particles и runouts, до terminal state и без раскрытия исходного deck.

**Architecture:** `FrozenBlueprintPolicy` получает пакетный inference API. `rollouts.py` materializes одну hidden history на particle, явно применяет каждый root action через `parallel_apply_action`, затем в lockstep продолжает все сценарии compact blueprint policy до terminal или emergency depth guard.

**Tech Stack:** Python 3.11, NumPy, PyTorch, локальный `pokers`, pytest.

**Spec:** `planning/solver_reports/solver.md`

## Global Constraints

- Не менять training loop, checkpoint format, `src/core/action_space.py` или `pokers/`.
- Один particle создаёт одну hidden history и один ordered runout для всех root actions.
- Нельзя использовать `state.deck` как источник future board; particle deck строится из independent full deck.
- Root action применяется явно; continuation использует compact blueprint slots.
- Rollout идёт до terminal state; `max_depth` — только emergency guard с диагностикой failure.
- Неподдерживаемые unequal initial stacks должны вызвать диагностический `ValueError`, а не реконструироваться с неверными stack sizes.

---

## File Structure

- `src/evaluation/blueprint_policy.py` — `probabilities_batch(states)` для batch inference.
- `src/runtime_search/rollouts.py` — particle materialization, lockstep engine и результаты оценки.
- `src/runtime_search/__init__.py` — публичные exports S3.
- `tests/test_frozen_blueprint_policy.py` — batch API совпадает с single-state inference.
- `tests/test_runtime_search_rollouts.py` — CRN particle, terminal completion, depth failure и повторяемость.
- `planning/solver_reports/004-lockstep-rollouts.md` — отчёт S3.

### Task 1: Batch blueprint inference

**Files:**
- Modify: `src/evaluation/blueprint_policy.py`
- Modify: `tests/test_frozen_blueprint_policy.py`

**Interfaces:**
- Produces: `FrozenBlueprintPolicy.probabilities_batch(states: Sequence[pkrs.State]) -> np.ndarray` формы `(len(states), NUM_ACTIONS)`.

- [x] **Step 1: Написать падающий batch test**

Для одного heavy checkpoint и двух deterministic states сравнить строки `probabilities_batch([state_a, state_b])` с двумя вызовами `probabilities`. Проверить shape, finite values, legal mask и сумму каждой строки.

- [x] **Step 2: Запустить test и увидеть отсутствие метода**

Run: `python -m pytest tests/test_frozen_blueprint_policy.py::test_frozen_policy_batch_probabilities_match_single_state_inference -q`

Expected: `AttributeError` для `probabilities_batch`.

- [x] **Step 3: Реализовать общий batch forward**

Закодировать каждый state относительно `current_player`, собрать state/mask tensors через `np.stack`, один раз вызвать `strategy_net`, применить masked softmax и вернуть `float32` matrix. Пустой batch вернуть как `(0, NUM_ACTIONS)`.

- [x] **Step 4: Запустить batch test**

Run: `python -m pytest tests/test_frozen_blueprint_policy.py::test_frozen_policy_batch_probabilities_match_single_state_inference -q`

Expected: PASS.

### Task 2: Particle materialization и lockstep engine

**Files:**
- Create: `src/runtime_search/rollouts.py`
- Modify: `src/runtime_search/__init__.py`
- Create: `tests/test_runtime_search_rollouts.py`

**Interfaces:**
- Produces: `materialize_particle_state(state, hero_id, particle, rng) -> pkrs.State`.
- Produces: `evaluate_root_actions(state, hero_id, root_actions, particles, continuation_policy, rng, max_depth=128) -> RolloutEvaluation`.
- `RolloutEvaluation` содержит `raw_ev_mean`, `raw_ev_std`, `successful_rollouts`, `total_particles`, `failure_messages`, `terminal_completion_rate`.

- [x] **Step 1: Написать падающие rollout tests**

На `build_constructed_spot(Turn)` и particle с пятью opponent hands/одним runout проверить, что materialized state получает particle hands, имеет particle.runout первым в deck и не зависит от перестановки исходного `state.deck`. Для двух legal root actions и deterministic passive batch policy проверить successful rollout каждого action, terminal completion rate `1.0`, finite EV, повторяемость fixed seed. Добавить test `max_depth=0`, ожидающий failure и completion rate `0.0`.

- [x] **Step 2: Запустить rollout tests и увидеть отсутствие модуля**

Run: `python -m pytest tests/test_runtime_search_rollouts.py -q`

Expected: collection error для `src.runtime_search.rollouts`.

- [x] **Step 3: Реализовать materialization**

Заменить все opponent hands картами particle в canonical player order, сохранить hero hand, public cards, bet/pot/active/turn owner/last action. Построить deck как `particle.runout + independently shuffled remaining_deck(known_cards)`. Проверить count particle hands, uniqueness и равенство initial stack всех players; затем вызвать `State.from_mid_hand` и восстановить `from_action`.

- [x] **Step 4: Реализовать lockstep evaluation**

Для каждой пары particle/root action materialize отдельный state и применить root actions одним `pkrs.parallel_apply_action`. На каждом шаге собрать nonterminal states, получить `continuation_policy.probabilities_batch`, сэмплировать slots из CRN stream `(particle_seed, depth, current_player)`, разрешить action через `resolve_action`, применить batch engine call. Неуспешные engine/status/resolve события и depth guard записывать в `failure_messages`; только terminal successes попадают в EV mean/std.

- [x] **Step 5: Запустить rollout tests**

Run: `python -m pytest tests/test_runtime_search_rollouts.py -q`

Expected: PASS; на тестовом particle все root action получают terminal reward.

### Task 3: Отчёт и целевая проверка

**Files:**
- Create: `planning/solver_reports/004-lockstep-rollouts.md`

- [x] **Step 1: Запустить целевые S1-S3 tests**

Run: `python -m pytest tests/test_frozen_blueprint_policy.py tests/test_paired_evaluation.py tests/test_mmds.py tests/test_runtime_search_spots.py tests/test_runtime_search_beliefs.py tests/test_runtime_search_rollouts.py -q`

Expected: exit code `0`.

- [x] **Step 2: Заполнить отчёт S3**

Зафиксировать result schema, particle/runout CRN invariant, terminal completion и ограничение initial uniform stack.
