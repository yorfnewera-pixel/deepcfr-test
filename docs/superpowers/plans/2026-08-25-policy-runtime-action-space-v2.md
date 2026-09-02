# Policy Runtime and Action Space v2 Implementation Plan

> **Для агентных исполнителей:** ОБЯЗАТЕЛЬНЫЙ НАВЫК: применить `superpowers:subagent-driven-development` или `superpowers:executing-plans` и выполнять задачи по очереди.

**Goal:** Перенести runtime policy в новый пакет, ввести action-space v2 и обновить консольный итог обучения.

**Architecture:** `src/core/action_space.py` остаётся единственным контрактом действий для training. Новый `policy_runtime` зеркально хранит необходимый state metadata для такой же маски и загружает только v2 light-checkpoint. GUI использует policy runtime для light-checkpoint; console summary строится в training loop после вычисления времён.

**Tech Stack:** Python, PyTorch, pytest, pokers.

**Spec:** `docs/superpowers/specs/2026-08-25-policy-runtime-action-space-v2-design.md`

## Global Constraints

- Старый пакет `inference` и action-space v1 не сохраняются.
- Проверка должна подтверждать mask training/runtime для новых checkpoint.
- Изменение mask требует новых checkpoint.

---

### Task 1: Ввести контракт action-space v2

**Files:**

- Modify: `src/core/action_space.py`
- Modify: `src/core/deep_cfr.py`
- Test: `tests/test_action_space.py`

**Interfaces:**

- Produces: `ACTION_SPACE_VERSION = "six_fixed_v2"`.
- Produces: helper `has_raise_on_current_street(state) -> bool`.

- [ ] **Step 1: Написать падающий тест рerаise-mask**

```python
def test_half_pot_raise_is_illegal_after_raise_on_current_street(state_after_raise):
    mask = legal_action_mask(state_after_raise)
    assert mask[ActionSlot.RAISE_HALF_POT] == 0.0
    assert mask[ActionSlot.RAISE_POT] == 1.0
    assert mask[ActionSlot.ALL_IN] == 1.0
```

- [ ] **Step 2: Запустить тест**

Run: `pytest tests/test_action_space.py::test_half_pot_raise_is_illegal_after_raise_on_current_street -v`

Expected: FAIL — half-pot ещё доступен.

- [ ] **Step 3: Реализовать helper и mask**

Проверять `last_stage_action` всех `players_state`; при существующем `Raise` выбрасывать `ValueError` в `resolve_action(RAISE_HALF_POT, state)`. Маска автоматически станет согласованной, так как строится через `resolve_action`. Удалить threshold filtering из `DeepCFRAgent.get_policy_distribution` и старое поле runtime threshold из его checkpoint config.

- [ ] **Step 4: Проверить**

Run: `pytest tests/test_action_space.py -v`

Expected: PASS.

### Task 2: Жёстко перенести inference в policy_runtime

**Files:**

- Move: `inference/` → `policy_runtime/`
- Modify: `policy_runtime/__init__.py`, `policy_runtime/core.py`, `policy_runtime/adapters/pokers.py`
- Modify: `scripts/poker_gui.py`, `scripts/eval_vs_random.py`, `tools/checkpoint_tools.py`
- Modify: `config.yaml`, `src/utils/config.py`, `tests/test_action_space.py`

**Interfaces:**

- Produces: `PolicyRuntimeAgent`, `PolicyRuntimeAdapter`, `policy_runtime_min_action_prob`.
- Consumes: v2 light-checkpoint and optional legacy checkpoint config key `inference_min_action_prob`.

- [ ] **Step 1: Написать падающие import/load-тесты**

```python
from policy_runtime.core import PolicyRuntimeAgent

def test_policy_runtime_loads_v2_light_checkpoint(checkpoint):
    assert PolicyRuntimeAgent(str(checkpoint)).validate() == ["OK: чекпоинт совместим"]
```

- [ ] **Step 2: Запустить тест**

Run: `pytest tests/test_action_space.py -v`

Expected: FAIL — пакета и класса ещё нет.

- [ ] **Step 3: Переместить пакет и обновить API**

Переименовать папку и все внутренние импорты. Переименовать класс и GUI adapter. В runtime добавить `last_stage_action` в protocol, dict-state и player-state serialization; применить ту же семантику half-pot mask. В runtime loader требовать `six_fixed_v2`, читать новый threshold с fallback к старому ключу. В GUI передавать `min_action_prob=0.0` при загрузке light-checkpoint.

- [ ] **Step 4: Обновить все imports**

Run: `rg -n "\\binference\\b|InferenceAgent|InferenceAdapter" --glob "*.py"`

Expected: отсутствуют production references.

- [ ] **Step 5: Проверить**

Run: `pytest tests/test_action_space.py -v`

Expected: PASS.

### Task 3: Сделать консольный итог итерации метрик-ориентированным

**Files:**

- Modify: `src/training/train.py`
- Test: `tests/test_training_regressions.py`

**Interfaces:**

- Produces: `_format_iteration_summary(iteration_elapsed, traversal_elapsed, advantage_loss, strategy_loss) -> str`.

- [ ] **Step 1: Написать падающий тест format helper**

```python
def test_iteration_summary_contains_all_requested_metrics():
    summary = _format_iteration_summary(4.5, 3.2, 0.1, 0.2)
    assert "Time/Iteration=4.5s" in summary
    assert "Time/Traversal=3.2s" in summary
    assert "Loss/Advantage=0.100000" in summary
    assert "Loss/Strategy=0.200000" in summary
```

- [ ] **Step 2: Запустить тест**

Run: `pytest tests/test_training_regressions.py::test_iteration_summary_contains_all_requested_metrics -v`

Expected: FAIL — helper отсутствует.

- [ ] **Step 3: Реализовать и использовать helper**

Вычислить `iteration_elapsed` до console print. Убрать старую строку `advantage=... strategy=... memory=...` и финальную строку времени, заменить одной форматированной строкой. TensorBoard output не менять.

- [ ] **Step 4: Проверить**

Run: `pytest tests/test_training_regressions.py -v`

Expected: PASS.

### Task 4: Итоговая проверка

**Files:**

- Verify: `src/core/action_space.py`, `policy_runtime/`, `scripts/poker_gui.py`, `src/training/train.py`, tests.

- [ ] **Step 1: Запустить целевые тесты**

Run: `pytest tests/test_action_space.py tests/test_checkpoint_lifecycle.py tests/test_training_regressions.py -v`

Expected: PASS.

- [ ] **Step 2: Проверить синтаксис**

Run: `python -m compileall -q src policy_runtime scripts tools`

Expected: exit 0.

- [ ] **Step 3: Проверить отсутствие старого пакета**

Run: `Test-Path inference`

Expected: `False`.

