# HU Artifact Safety Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:executing-plans` to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Исключить потерю HU checkpoint при retention, обеспечить загрузку actor-conditioned HU light policy и сделать восстановление mid-hand детерминированным.

**Architecture:** Каждый новый HU training run создаёт собственный каталог с неизменяемым manifest; resume использует каталог manifest исходного checkpoint. `FrozenBlueprintPolicy` отделяет базовый encoder от расширенного входа shared HU strategy. Rust-конструктор отвергает неполные данные raise-history.

**Tech Stack:** Python 3.11, PyTorch, pytest, Rust/PyO3.

**Spec:** `planning/буфер_планов/рефакторинг_от_Astra/план3.txt`

## Global Constraints

- Не менять математику D2CFR, lifecycle replay или порядок обучения P0/P1.
- Сначала добавлять regression test, проверять его ожидаемое падение, затем вносить минимальную production-правку.
- Изменения должны оставаться обратно-совместимыми только там, где это не нарушает контракт точного восстановления состояния.

---

### Task 1: Изолировать артефакты HU training run

**Files:**
- Modify: `src/training/train.py`
- Test: `tests/test_checkpoint_lifecycle.py`, `tests/test_hu_self_play_training.py`

- [ ] **Step 1: Добавить failing tests для нового run и защиты текущей итерации**

```python
def test_hu_retention_keeps_current_checkpoint_when_old_run_has_larger_iterations(tmp_path):
    # 50_000..145_000 принадлежат другой series; 5_000 текущей series сохраняется.
    ...
    assert full_path.is_file()
    assert light_path.is_file()
```

- [ ] **Step 2: Запустить regression tests и зафиксировать RED**

Run: `python -m pytest tests/test_checkpoint_lifecycle.py tests/test_hu_self_play_training.py -q`

- [ ] **Step 3: Реализовать manifest и run-scoped retention**

```python
run_dir = _prepare_hu_run_directory(save_dir, agent, seed, initial_checkpoint)
_save_hu_iteration_checkpoints(agent, run_dir, iteration, seed=seed, run_id=run_id)
```

- [ ] **Step 4: Проверить GREEN**

Run: `python -m pytest tests/test_checkpoint_lifecycle.py tests/test_hu_self_play_training.py -q`

### Task 2: Загружать actor-conditioned HU light checkpoint

**Files:**
- Modify: `src/evaluation/blueprint_policy.py`
- Test: `tests/test_frozen_blueprint_policy.py`

- [ ] **Step 1: Добавить failing tests round-trip P0/P1 и batch**

```python
policy = FrozenBlueprintPolicy.from_checkpoint(path, num_players=2)
assert np.allclose(policy.probabilities(state, player_id=0), expected_p0)
assert np.allclose(policy.probabilities(state, player_id=1), expected_p1)
```

- [ ] **Step 2: Запустить test и зафиксировать RED**

Run: `python -m pytest tests/test_frozen_blueprint_policy.py -q`

- [ ] **Step 3: Реализовать валидацию metadata и actor one-hot**

```python
actor = np.zeros(self.strategy_actor_count, dtype=np.float32)
actor[player_id] = 1.0
return np.concatenate((base, actor))
```

- [ ] **Step 4: Проверить GREEN**

Run: `python -m pytest tests/test_frozen_blueprint_policy.py -q`

### Task 3: Требовать last_raise_increment при from_mid_hand

**Files:**
- Modify: `pokers/src/game_logic.rs`, `pokers/pokers.pyi`
- Test: `pokers/tests/test_mid_hand.py`, `pokers/src/game_logic.rs`

- [ ] **Step 1: Добавить failing validation tests**

```python
with pytest.raises(OSError, match="last_raise_increment must be provided"):
    pkrs.State.from_mid_hand(**mid_hand_arguments(last_raise_increment=None))
```

- [ ] **Step 2: Запустить test и зафиксировать RED**

Run: `python -m pytest pokers/tests/test_mid_hand.py -q`

- [ ] **Step 3: Реализовать строгую Rust-валидацию**

```rust
let increment = last_raise_increment.ok_or_else(|| InitStateError { ... })?;
if !increment.is_finite() || increment + CHIP_EPSILON < bb { return Err(...); }
```

- [ ] **Step 4: Проверить GREEN**

Run: `python -m pytest pokers/tests/test_mid_hand.py -q`

### Task 4: Сквозная проверка

**Files:**
- Test: `tests/test_checkpoint_lifecycle.py`, `tests/test_frozen_blueprint_policy.py`, `tests/test_policy_runtime_history_v3.py`, `pokers/tests/test_mid_hand.py`

- [ ] **Step 1: Запустить целевой Python и Rust test набор**

Run: `python -m pytest tests/test_checkpoint_lifecycle.py tests/test_hu_self_play_training.py tests/test_frozen_blueprint_policy.py tests/test_policy_runtime_history_v3.py pokers/tests/test_mid_hand.py -q`

- [ ] **Step 2: Выполнить Rust проверки**

Run: `cargo test` в `pokers/`

