# D2CFR HU Correctness Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `executing-plans` to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Сделать новый цикл HU D2CFR корректным по правилам покера, lifecycle replay, runtime-контракту и подготовке fresh 6-max transfer.

**Architecture:** Старые checkpoint и продолжение их обучения не поддерживаются. Правила движка исправляются в Rust с регрессиями; Python-контракты используют явные валидные строки reservoir, actor-conditioned HU strategy и pairing по идентификатору частицы. Existing D2CFR HU snapshots и transaction lifecycle сохраняются.

**Tech Stack:** Rust/pyo3, Python 3.11, NumPy, PyTorch, pytest.

**Spec:** `planning/буфер_планов/рефакторинг_от_Astra/план.txt`

## Global Constraints

- Не мигрировать и не загружать прежние checkpoint как совместимые training artifacts.
- Не менять legacy DCFR lifecycle за пределами изолированных D2CFR-веток.
- Каждое изменение поведения начинается с наблюдаемого failing test.
- Сохранять fixed action space из шести действий.

---

### Task 1: Стандартные HU-правила и evaluator

**Files:**
- Modify: `pokers/src/game_logic.rs`
- Test: `pokers/src/game_logic.rs`

- [ ] Добавить deterministic Rust-тесты для blinds/action order при button P0/P1, BB после limp, wheel и wheel straight flush.
- [ ] Запустить `cargo test`, подтвердить красные регрессии.
- [ ] Разделить HU и multiplayer assignment blinds/current player; определять check/call через `call_amount`; добавить `straight_high_rank`.
- [ ] Запустить `cargo test` и `cargo fmt --check`.

### Task 2: Явная модель historical reservoir

**Files:**
- Modify: `src/core/buffers.py`
- Modify: `src/core/hu_self_play.py`
- Test: `tests/test_buffer_reservoir_flags.py`
- Test: `tests/test_d2cfr_model_and_buffer.py`

- [ ] Добавить failing tests, отделяющие `size` от `total_seen` до и после reservoir replacement/restore.
- [ ] Ввести единый механизм выбора reservoir slot и использовать размер валидных строк в `len`/sampling.
- [ ] Запустить адресные pytest tests.

### Task 3: Fresh checkpoint payload и D2 lifecycle

**Files:**
- Modify: `src/core/deep_cfr.py`
- Modify: `src/training/train.py`
- Test: `tests/test_d2cfr_checkpoint.py`
- Test: `tests/test_hu_checkpoint_resume.py`

- [ ] Добавить failing tests для сохранения `capacity`, `size`, `total_seen`, counters и запрета capacity mismatch при resume.
- [ ] Сохранять только valid rows с явно описанной схемой replay; legacy payload не поддерживать.
- [ ] Сохранить существующие D2 HU snapshots, P0/P1 isolation и pending-sample rollback.
- [ ] Запустить checkpoint/HU regression subset.

### Task 4: HU light checkpoint и independent runtime

**Files:**
- Modify: `src/core/deep_cfr.py`
- Modify: `policy_runtime/core.py`
- Test: `tests/test_hu_self_play_training.py`
- Test: `tests/test_policy_runtime_history_v3.py`

- [ ] Добавить failing end-to-end test: HU strategy checkpoint загружается runtime и строит нормированные policy для P0/P1.
- [ ] Добавить explicit actor-conditioning metadata; runtime проверяет форму strategy weights и добавляет actor one-hot.
- [ ] Запустить адресные pytest tests.

### Task 5: Pairing runtime-search particles

**Files:**
- Modify: `src/runtime_search/rollouts.py`
- Test: `tests/test_runtime_search_rollouts.py`

- [ ] Добавить failing test с разным порядком завершения и частично failed particle.
- [ ] Хранить reward/success по `(action_index, particle_index)` и вычислять paired statistics только по общей маске.
- [ ] Запустить runtime-search tests.

### Task 6: Generic multiplayer и fresh HU-to-6-max transfer

**Files:**
- Modify: `src/training/train.py`
- Modify: `src/core/deep_cfr.py`
- Modify: `src/core/teacher_transfer.py`
- Test: `tests/test_teacher_transfer.py`
- Test: `tests/test_checkpoint_lifecycle.py`

- [ ] Добавить tests, что данные всех traversers сохраняются, обучение запускается после collection, а transfer создаёт fresh empty replay.
- [ ] Изолировать D2 historical lifecycle от legacy clearing; закрепить единый frozen profile per iteration.
- [ ] Оставить transfer только card encoder и не переносить HU replay, optimizers, strategy head либо iteration.
- [ ] Запустить transfer/multiplayer regression subset.

### Task 7: Legacy isolation, docs и full verification

**Files:**
- Modify: `src/training/__init__.py`
- Modify: `README.md`
- Test: `tests/test_training_regressions.py`

- [ ] Убрать broken opponent-modeling entrypoint из публичного API либо явно переместить в legacy.
- [ ] Обновить README: active HU D2CFR, fresh 6-max card-encoder transfer, запрет старого resume.
- [ ] Запустить `cargo test`, полный pytest и focused smoke.
