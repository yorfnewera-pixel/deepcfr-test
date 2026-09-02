# Runtime Search Spots and Beliefs Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Создать воспроизводимые turn/river spots и blocker-aware belief particles без утечки будущего board.

**Architecture:** `cards.py` создаёт независимую полную колоду и сравнивает карты по rank/suit. `spots.py` строит валидные 6-max узлы через `State.from_mid_hand`; `beliefs.py` сэмплирует скрытые руки и runout из полной колоды за вычетом известных карт, не читая `state.deck`.

**Tech Stack:** Python 3.11, NumPy, локальный `pokers`, pytest.

**Spec:** `planning/solver_reports/solver.md`

## Global Constraints

- Не менять training loop, checkpoint format, `src/core/action_space.py` и `pokers/`.
- `stake` в `State.from_mid_hand` всегда означает начальный stack, не оставшийся stack.
- Particle не может содержать повторяющихся карт или карт hero/public board.
- Runout выбирать из independently constructed remaining deck; запрещено использовать `state.deck[:k]` как будущий board.
- Uniform blocker-aware sampling — только baseline до S6 и не считается solver-valid belief.

---

## File Structure

- `src/runtime_search/cards.py` — canonical card identity, независимая полная колода и исключение известных карт.
- `src/runtime_search/spots.py` — deterministic constructed turn/river states.
- `src/runtime_search/beliefs.py` — immutable particle и blocker-aware sampler.
- `tests/test_runtime_search_spots.py` — корректность `from_mid_hand`, initial/remaining stack и отсутствие duplicate cards.
- `tests/test_runtime_search_beliefs.py` — blocker/runout/reproducibility/regression против утечки deck.
- `planning/solver_reports/003-belief-spots.md` — отчёт S2.

### Task 1: Карточный инвариант и constructed spots

**Files:**
- Create: `src/runtime_search/cards.py`
- Create: `src/runtime_search/spots.py`
- Create: `tests/test_runtime_search_spots.py`

**Interfaces:**
- Produces: `card_key(card) -> tuple[int, int]`, `full_deck() -> tuple[pkrs.Card, ...]`, `remaining_deck(excluded_cards) -> tuple[pkrs.Card, ...]`, `build_constructed_spot(stage: pkrs.Stage) -> pkrs.State`.

- [x] **Step 1: Написать падающие тесты turn/river spots**

Проверить для Turn и River: stage и количество public cards соответствуют улице; шесть игроков активны; `pot == 180.0`; каждый игрок имеет `stake == 170.0`, потому что при `stake=200.0` уже committed `30.0`; все hero/board/deck cards уникальны.

- [x] **Step 2: Запустить spot tests и убедиться в ожидаемом падении**

Run: `python -m pytest tests/test_runtime_search_spots.py -q`

Expected: collection error, потому что `src.runtime_search.spots` отсутствует.

- [x] **Step 3: Реализовать cards и spots**

Построить 52 карты через `pkrs.Card.from_string(f"{suit}{rank}")`; сравнивать карты только через `(int(rank), int(suit))`. В `build_constructed_spot` вычесть hole/public cards из полной колоды, передать остаток как `deck`, а `stake=200.0`, `pot_chips=[30.0] * 6`, `bet_chips=[0.0] * 6`, `pot=180.0` в `State.from_mid_hand`.

- [x] **Step 4: Запустить spot tests**

Run: `python -m pytest tests/test_runtime_search_spots.py -q`

Expected: tests проходят; constructed state не зависит от checkpoint или training.

### Task 2: Blocker-aware particles

**Files:**
- Create: `src/runtime_search/beliefs.py`
- Create: `tests/test_runtime_search_beliefs.py`
- Modify: `src/runtime_search/__init__.py`

**Interfaces:**
- Produces: `BeliefParticle(opponent_hands, runout, is_solver_valid=False)` и `sample_blocker_aware_particle(hero_hand, public_cards, num_opponents, runout_cards, rng) -> BeliefParticle`.

- [x] **Step 1: Написать падающие particle tests**

С fixed `np.random.default_rng(107)` проверить: пять hand по две карты и один turn runout не пересекаются с hero/board; вызовы с одинаковым seed равны по card keys; замена значения в `state.deck` не меняет sample при тех же известных cards и seed; невозможный sample выбрасывает `ValueError` с диагностикой.

- [x] **Step 2: Запустить particle tests и убедиться в ожидаемом падении**

Run: `python -m pytest tests/test_runtime_search_beliefs.py -q`

Expected: collection error, потому что `src.runtime_search.beliefs` отсутствует.

- [x] **Step 3: Реализовать blocker-aware sampler**

Вычесть `hero_hand + public_cards` через `remaining_deck`, проверить capacity `2 * num_opponents + runout_cards`, создать одну permutation c supplied `np.random.Generator`, разрезать её на пары opponents и runout. Не принимать `State` и не читать `state.deck`; `is_solver_valid=False` явно маркирует отсутствие reach weighting.

- [x] **Step 4: Запустить particle tests**

Run: `python -m pytest tests/test_runtime_search_beliefs.py -q`

Expected: tests проходят, sampler детерминирован от supplied RNG.

### Task 3: Отчёт и целевая проверка

**Files:**
- Create: `planning/solver_reports/003-belief-spots.md`

- [x] **Step 1: Запустить целевые тесты S2**

Run: `python -m pytest tests/test_runtime_search_spots.py tests/test_runtime_search_beliefs.py -q`

Expected: exit code `0`.

- [x] **Step 2: Заполнить отчёт S2**

Зафиксировать constructed state contract, independent runout source, число tests и ограничение uniform particle до S6.
