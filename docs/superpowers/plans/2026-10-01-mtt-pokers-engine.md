# MTT Pokers Engine Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Перевести локальный `pokers/` на точный контракт одной 2--8-max MTT-раздачи с разными стеками, обычным ante и корректными банками.

**Architecture:** За основу берётся проверенная механика эталонного форка: `u64` chip-единицы, состояние ожидающих действий и settlement вложенных банков. В неё переносятся только необходимые для MTT публичная история и API; ante добавляется как отдельный forced contribution, не влияющий на уличную ставку.

**Tech Stack:** Rust 2021, PyO3, maturin, pytest, proptest.

**Spec:** `docs/superpowers/specs/2026-10-01-mtt-pokers-engine-design.md`

## Global Constraints

- Рабочая область: только `C:\Users\Cassmall\Desktop\deepcfr-test`.
- Новый контракт несовместим со старыми checkpoint и action-space.
- Big-blind ante, ICM, payouts и турнирный lifecycle вне этого плана.
- Любая новая логика сначала получает падающий тест.

## Review Focus

- Некратная `chip_unit` сумма должна быть отвергнута, а не округлена.
- Короткий BB сохраняет nominal bring-in в многосторонней торговле, но не
  создаёт непокрытый колл для единственного плательщика.
- Короткое повышение обязано запросить колл, но не обязано открыть рейз.
- Folded contribution остаётся в банке без eligibility.
- Ошибочный переход не должен менять публичную историю успешных действий.

---

### Task 1: Базовый целочисленный контракт и неравные стеки

**Files:**
- Modify: `pokers/src/state.rs`
- Modify: `pokers/src/game_logic.rs`
- Modify: `pokers/pokers.pyi`
- Test: `pokers/tests/test_mtt_contract.py`

**Interfaces:**
- Produces: `State.from_seed(..., chip_unit=..., stakes=...)` и
  `State.from_deck(..., chip_unit=..., stakes=...)`.

- [ ] Написать падающие тесты разных стеков, коротких SB/BB, nominal
  bring-in, единственного плательщика и отклонения некратного `chip_unit`
  значения.
- [ ] Запустить только новые тесты и подтвердить ожидаемое падение.
- [ ] Перенести `u64`-арифметику с checked operations, валидацию и частичную
  оплату blinds из эталона; сохранить Python-величины через getters.
- [ ] Пересобрать локальный PyO3-модуль через maturin, проверить
  `pokers.__file__` и версию игрового контракта, затем запустить новые тесты.

### Task 2: Торговый автомат и side pots

**Files:**
- Modify: `pokers/src/game_logic.rs`
- Modify: `pokers/src/state.rs`
- Test: `pokers/tests/test_mtt_contract.py`

**Interfaces:**
- Consumes: целочисленные вклады Task 1.
- Produces: `min_raise`, корректные legal actions, settlement side pots и
  возврат непокрытого избытка.

- [ ] Написать падающие fixtures short all-in, индивидуального reopening,
  side pots с точными банками/eligible/выплатами, folded eligibility, odd
  chips, uncalled excess и short-BB номинального call.
- [ ] Запустить тесты и подтвердить падение на старом автомате.
- [ ] Перенести `pending`/`acted_at`, minimum full raise, forced runout и
  settlement из эталона, сохранив immutable `apply_action`.
- [ ] Пересобрать PyO3-модуль и запустить тесты Task 2 и property-проверку
  точного сохранения фишек.

### Task 3: Per-player ante и публичная история

**Files:**
- Modify: `pokers/src/game_logic.rs`
- Modify: `pokers/src/state.rs`
- Modify: `pokers/src/state/action.rs`
- Modify: `pokers/pokers.pyi`
- Test: `pokers/tests/test_mtt_contract.py`

**Interfaces:**
- Consumes: forced blinds и settlement Tasks 1--2.
- Produces: `ante` в constructors и публичные записи фактических действий.

- [ ] Написать падающие тесты порядка ante--blinds, all-in только из-за ante,
  side pot с ante и отсутствия записи для illegal action.
- [ ] Запустить тесты и подтвердить падение.
- [ ] Реализовать ante как `pot_chips`, не меняя `bet_chips`, и адаптировать
  публичную историю к новому автомату.
- [ ] Пересобрать PyO3-модуль, затем запустить тесты Task 3, весь
  `pokers/tests` и `cargo test`.

### Task 4: Новый action-space контракт и цепочка раздач

**Files:**
- Modify: `src/core/action_space.py`
- Modify: `tests/test_action_space.py`
- Create: `tests/test_mtt_hand_chain.py`

**Interfaces:**
- Consumes: `State.min_raise`, `stakes`, `ante` и конечные стеки движка.
- Produces: версия action-space MTT и согласованные `legal_action_mask` /
  `resolve_action`.

- [ ] Написать падающие тесты, что min raise движка и маска совпадают,
  half-pot нечётного банка квантуется вниз, сайзинг на границе min raise/all-in
  не получает fallback, а живые конечные стеки становятся `stakes` следующей
  раздачи.
- [ ] Запустить тесты и подтвердить падение из-за legacy
  `last_raise_increment`.
- [ ] Заменить зависимость action-space на `min_raise`, реализовать явное
  квантование через `chip_unit`, повысить версию контракта и реализовать
  тестовую цепочку без раздачи карт выбывшему игроку.
- [ ] Запустить тесты Task 4 и целевые Python-регрессии.

### Task 5: Документация контракта и итоговая верификация

**Files:**
- Modify: `pokers/README.md`
- Modify: `pokers/documentation.md`
- Test: `pokers/tests/test_mtt_contract.py`, `tests/test_mtt_hand_chain.py`

**Interfaces:**
- Consumes: публичный API Tasks 1--4.
- Produces: документированный MTT game-contract version и команды сборки.

- [ ] Обновить документацию и пример 8 разных стеков с ante без ссылок на
  legacy checkpoints.
- [ ] Сверить пример вручную с финальными сигнатурами constructors и
  `min_raise`.
- [ ] Выполнить `cargo test`, `pytest pokers/tests -q` и целевой `pytest`
  набора MTT/action-space; после пересборки PyO3 дополнительно выполнить
  `tests/test_pokers_showdown.py` и карточные тесты абстракции; зафиксировать
  результаты.
