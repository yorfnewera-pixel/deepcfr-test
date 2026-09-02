# Postflop Identity Runtime Search Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Запускать reach-weighted runtime search во всех compact postflop root nodes.

**Architecture:** `spots.py` создаёт flop constructed state. `policy.py` выбирает direct identity mask по наличию call и поддерживает Flop runout. `history.py` делегирует любые postflop identity roots в runtime search; rollout и beliefs contracts не меняются.

**Tech Stack:** Python 3.11, NumPy, локальный `pokers`, pytest.

**Spec:** `docs/superpowers/specs/2026-08-28-postflop-identity-runtime-design.md`

## Global Constraints

- Не менять training loop, checkpoint format, модель или `src/core/action_space.py`.
- Не использовать mapping на новые runtime sizing slots.
- При incomplete history, low ESS или rollout failure сохранять blueprint fallback.

### Task 1: Flop spot и root masks

**Files:** `src/runtime_search/spots.py`, `src/runtime_search/policy.py`, `tests/test_runtime_search_spots.py`, `tests/test_runtime_search_policy.py`

- [x] Написать падающие tests: `build_constructed_spot(Flop)` и `choose_action` с history на flop возвращают legal action и finite values на identity legal slots.
- [x] Запустить tests и увидеть отсутствие flop/root support.
- [x] Добавить flop board, `runout_cards=2` и public root-mask helper: no-bet `check/halfpot/pot/all-in`, response existing set.
- [x] Запустить tests и получить PASS.

### Task 2: Runtime wrapper и проверка

**Files:** `src/runtime_search/history.py`, `tests/test_runtime_search_history.py`, `planning/solver_reports/011-postflop-identity-runtime.md`

- [x] Написать падающий test: `RuntimeSearchPolicy` вызывает search на flop no-bet root и сохраняет `last_search_decision`.
- [x] Заменить response-only predicate на public postflop identity predicate.
- [x] Запустить целевой S1-S7 набор, compileall и записать report с результатом активации.
