# Teacher Transfer Stage A — Task 2 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Явно включать проверенный HU → six-max warm-start только для `strategy_net.card_encoder` и сохранять его provenance в full checkpoint.

**Architecture:** Конфигурация остаётся opt-in: новый six-max запуск строит `card_context_v1`, один раз переносит проверенный блок и записывает неизменяемый provenance. Resume всегда сначала загружает student checkpoint и никогда повторно не применяет teacher. Старый `teacher_strategy_checkpoint` сохраняет прежнюю изолированную семантику и не включает Stage A.

**Tech Stack:** Python, PyTorch, pytest, YAML.

**Spec:** `planning/буфер_планов/хедзап/план ученик учитель коррекция.txt`

## Global Constraints

- Новые флаги: `teacher_transfer_enabled`, `teacher_transfer_checkpoint`, `teacher_transfer_mode=card_encoder_warmstart`, `teacher_transfer_freeze_card_encoder`, `teacher_hu_aux_distillation_enabled`, `teacher_hu_aux_distillation_weight`.
- Stage B и старая HU projection не реализуются; включение auxiliary distillation завершается ясным `ValueError`.
- Допустим только проверенный versioned HU full checkpoint из `load_card_encoder_from_hu_checkpoint`.
- Переносится только policy `card_encoder`; action/context/advantage, optimizer и replay-buffer teacher не переносятся.
- Ошибки и комментарии — на русском языке; legacy behavior при выключенных флагах сохраняется.

---

### Task 1: Lifecycle/configuration Stage A

**Files:**
- Modify: `config.yaml`
- Modify: `src/utils/config.py`
- Modify: `src/training/train.py`
- Modify: `src/core/deep_cfr.py`
- Test: `tests/test_teacher_transfer.py`
- Test: `tests/test_checkpoint_lifecycle.py`

**Interfaces:**
- Consumes: `DeepCFRAgent.load_card_encoder_from_hu_checkpoint(path, freeze=False)`.
- Produces: six-max full checkpoint with transfer provenance and startup flow without повторного переноса на resume.

- [ ] **Step 1: Write failing lifecycle tests**

Покрыть disabled legacy path, новый enabled six-max warm-start, provenance в full checkpoint, resume без повторного переноса, freeze и несовместимые флаги.

- [ ] **Step 2: Verify RED**

Run: `python -m pytest -q tests/test_teacher_transfer.py tests/test_checkpoint_lifecycle.py`

Expected: failures because lifecycle/configuration is absent.

- [ ] **Step 3: Add explicit configuration contract**

Добавить defaults и YAML flags; отвергать неверный mode, отсутствующий checkpoint при enabled, Stage B и несовместимость HU/current policy или non-six-max запуска.

- [ ] **Step 4: Add one-shot startup and checkpoint provenance**

На fresh six-max запуске создавать `card_context_v1`, выполнять transfer после создания student; при `initial_checkpoint` сперва загрузить student и не вызывать transfer. Сохранять provenance в full checkpoint и восстанавливать его как metadata student, без повторной мутации весов.

- [ ] **Step 5: Verify GREEN**

Run: `python -m pytest -q tests/test_teacher_transfer.py tests/test_checkpoint_lifecycle.py tests/test_hu_checkpoint_resume.py tests/test_card_context_architecture.py tests/test_training_regressions.py`

- [ ] **Step 6: Commit and scoped review**

Commit only Task 1 files, then fix all Important review findings before moving on.
