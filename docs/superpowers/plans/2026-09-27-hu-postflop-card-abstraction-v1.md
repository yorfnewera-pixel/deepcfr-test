# HU postflop card abstraction V1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Построить и проверить офлайн HU card abstraction: lossless preflop из 169 классов и deterministic postflop equity-profile с тремя независимыми моделями `k-means(200)`.

**Architecture:** `src/card_abstraction` изолирован от Deep CFR и принимает только карты. Локальный `pokers` получает узкий showdown API; Python-слой канонизирует ситуации, строит feature, обучает модели и атомарно публикует artifacts. Runtime resolver и изменение solver не входят.

**Tech Stack:** Python 3, NumPy, pandas, scikit-learn, joblib, PyArrow, локальный `pokers` (Rust/PyO3), pytest.

**Spec:** `docs/superpowers/specs/2026-09-27-hu-postflop-card-abstraction-v1-design.md`

## Global Constraints

- Поддерживать только HU и board длиной 0, 3, 4 или 5; six-max отклонять.
- Preflop — ровно 169 lossless canonical hand classes: 13 pair, 78 suited, 78 offsuit; не использовать для него equity или `k-means`.
- `CanonicalCardKey` не содержит history, pot, stack, action mask или player ID.
- Использовать только `UniformHeadsUpRange`.
- Feature: 27 конечных `float32`; flop/turn `128 × 128`, river exact enumeration.
- Использовать 24 suit permutations, L2, `KMeans(k=200, k-means++, n_init=20, max_iter=500, tol=1e-4)` и master-seed `20260927`.
- Не менять D2CFR, checkpoint, action space, card context или существующие игровые переходы.
- Не копировать code из reference-репозиториев; stage/commit только task-owned files.

## Review Focus

- Дубликаты hero/board отклоняются до sampling — Task 2.
- Suit-isomorphic ситуации дают один key и feature — Tasks 2–3.
- River не использует RNG и учитывает tie как `0.5` — Tasks 1 и 3.
- Частичный или manifest-несовместимый artifact не публикуется — Task 5.
- Дедупликация не уменьшает production dataset ниже 100 000 без ошибки — Task 4.

---

## Структура файлов

| Файл | Ответственность |
| --- | --- |
| `pokers/src/game_logic.rs`, `pokers/src/lib.rs`, `pokers/pokers.pyi` | PyO3 API сравнения готового showdown. |
| `src/card_abstraction/domain.py`, `canonical.py`, `preflop.py` | Situation/key, validation, 24-permutation canonicalization и lossless 169 table. |
| `src/card_abstraction/equity.py`, `features.py` | Equity sampling и 27-мерный profile. |
| `src/card_abstraction/dataset.py` | Выборка unique canonical states. |
| `src/card_abstraction/artifacts.py`, `pipeline.py` | K-means, validation, manifest и atomic publication. |
| `tools/build_hu_postflop_abstraction.py` | Офлайн CLI, не вызываемый solver. |

### Task 1: Выставить showdown evaluator из локального `pokers`

**Files:** Modify `pokers/src/game_logic.rs`, `pokers/src/lib.rs`, `pokers/pokers.pyi`; create `tests/test_pokers_showdown.py`.

**Interfaces:** Produces `pkrs.compare_showdown(hero: tuple[Card, Card], opponent: tuple[Card, Card], board: list[Card]) -> int`. Return `1` для win hero, `0` для tie, `-1` для loss; принять пять board cards и семь разных карт.

- [ ] **Step 1: Написать failing tests**

```python
def test_compare_showdown_counts_tie_as_zero():
    assert pkrs.compare_showdown(hero, opponent, board) == 0

def test_compare_showdown_rejects_duplicate_card():
    with pytest.raises(ValueError, match="дубликат"):
        pkrs.compare_showdown(hero, opponent, board_with_hero_card)
```

Добавить win/loss cases.

- [ ] **Step 2: Запустить test до реализации**

Run: `python -m pytest -q tests/test_pokers_showdown.py`

Expected: FAIL, `compare_showdown` отсутствует.

- [ ] **Step 3: Реализовать и экспортировать `compare_showdown`**

Проверить размеры/уникальность до ранжирования; сравнить существующий `rank_hand` двух игроков. Не менять `State.apply_action` или семантику рангов.

- [ ] **Step 4: Пересобрать local package и проверить test**

Run: `cd pokers; maturin develop --release; cd ..; python -m pytest -q tests/test_pokers_showdown.py`

Expected: PASS.

- [ ] **Step 5: Commit task-owned files**

Run: `git add pokers/src/game_logic.rs pokers/src/lib.rs pokers/pokers.pyi tests/test_pokers_showdown.py && git commit -m "feat: expose heads-up showdown evaluator"`

### Task 2: Domain types и suit canonicalization

**Files:** Create `src/card_abstraction/__init__.py`, `domain.py`, `canonical.py`, `preflop.py`, `tests/test_card_abstraction_canonical.py`.

**Interfaces:** Consumes `pkrs.Card`, `card_key`, `full_deck`. Produces `Street`, `CardSituation.create(hero, board)`, `CanonicalCardKey`, `canonicalize(situation) -> CanonicalCardKey`, `build_lossless_preflop_table() -> pd.DataFrame`.

- [ ] **Step 1: Написать failing tests canonical contract**

```python
def test_canonicalize_is_invariant_to_all_common_suit_permutations():
    assert canonicalize(source) == canonicalize(permuted_source)

def test_card_situation_rejects_duplicate_card_and_invalid_board_size():
    with pytest.raises(ValueError):
        CardSituation.create(hero, duplicate_or_invalid_board)
```

Добавить порядок hero/board, streets 0/3/4/5 и test, что таблица содержит
1326 physical hands, 169 keys и разбиение 13/78/78.

- [ ] **Step 2: Запустить test до реализации**

Run: `python -m pytest -q tests/test_card_abstraction_canonical.py`

Expected: FAIL, модуль отсутствует.

- [ ] **Step 3: Реализовать immutable domain и canonicalization**

Key содержит только normalised `(rank, suit)` hero/board и street. Перебрать 24 общие suit permutations и вернуть минимальный лексикографический кортеж. `preflop.py` перечисляет 1326 неупорядоченных hands и строит lossless таблицу без sampling.

- [ ] **Step 4: Запустить tests и закоммитить task**

Run: `python -m pytest -q tests/test_card_abstraction_canonical.py`

Expected: PASS.

Run: `git add src/card_abstraction tests/test_card_abstraction_canonical.py && git commit -m "feat: add lossless preflop and canonical card situations"`

### Task 3: Deterministic equity profile

**Files:** Create `src/card_abstraction/equity.py`, `features.py`, `tests/test_card_abstraction_features.py`.

**Interfaces:** Produces `conditional_equities(situation, master_seed=20260927) -> np.ndarray` and `build_feature(situation, master_seed=20260927) -> np.ndarray`. Flop/turn return 128 conditional equities from 128 runout × 128 opponent hands; river returns one exact equity; feature `(27,)`, `float32`.

- [ ] **Step 1: Написать failing feature tests**

```python
def test_river_feature_matches_exact_enumeration_and_uses_tie_half():
    assert feature[0] == pytest.approx(manual_exact_equity)
    assert feature[1] == 0.0

def test_feature_is_reproducible_and_suit_invariant():
    assert np.array_equal(build_feature(state), build_feature(permuted_state))
```

Добавить finite, histogram sum `1 ± 1e-6`, dtype/shape и иной seed на flop.

- [ ] **Step 2: Запустить test до реализации**

Run: `python -m pytest -q tests/test_card_abstraction_features.py`

Expected: FAIL, функции отсутствуют.

- [ ] **Step 3: Реализовать sampling и feature**

Derive seed через SHA-256 от feature version, master-seed, street и canonical key; использовать локальный `numpy.random.Generator`. River перечисляет legal unordered opponent pairs без RNG. Вернуть `[mean, std, q10, q25, q50, q75, q90, hist_0..hist_19]`.

- [ ] **Step 4: Проверить и закоммитить task**

Run: `python -m pytest -q tests/test_card_abstraction_features.py tests/test_pokers_showdown.py`

Expected: PASS.

Run: `git add src/card_abstraction/equity.py src/card_abstraction/features.py tests/test_card_abstraction_features.py && git commit -m "feat: add deterministic equity profiles"`

### Task 4: Dataset builder и dependencies

**Files:** Modify `requirements.txt`; create `src/card_abstraction/dataset.py`, `tests/test_card_abstraction_dataset.py`.

**Interfaces:** Produces `sample_unique_situations(street: Street, count: int, master_seed: int) -> list[CardSituation]`. Production receives `count=100_000` and returns exactly that many unique keys or raises `RuntimeError`.

- [ ] **Step 1: Написать failing reproducibility и exhaustion tests**

```python
def test_sample_unique_situations_is_seed_reproducible_and_deduplicated():
    result = sample_unique_situations(Street.FLOP, 100, 20260927)
    assert len({canonicalize(item) for item in result}) == 100
```

- [ ] **Step 2: Запустить test до реализации**

Run: `python -m pytest -q tests/test_card_abstraction_dataset.py`

Expected: FAIL, builder отсутствует.

- [ ] **Step 3: Добавить dependencies и выборку**

Перед фиксацией versions `scikit-learn`, `joblib`, `pyarrow` в `requirements.txt` сверить API и Python support через Context7. Генерировать legal situations из `full_deck`, дедуплицировать только canonical key и не уменьшать результат молча.

- [ ] **Step 4: Проверить и закоммитить task**

Run: `python -m pytest -q tests/test_card_abstraction_dataset.py`

Expected: PASS.

Run: `git add requirements.txt src/card_abstraction/dataset.py tests/test_card_abstraction_dataset.py && git commit -m "feat: sample reproducible postflop datasets"`

### Task 5: Artifacts, clustering, validation и CLI

**Files:** Create `src/card_abstraction/artifacts.py`, `pipeline.py`, `tools/build_hu_postflop_abstraction.py`, `tests/test_card_abstraction_artifacts.py`, `tests/test_card_abstraction_pipeline.py`.

**Interfaces:** Produces `build_street(street, output_root, master_seed=20260927, sample_count=100_000) -> Path`, `build_lossless_preflop_artifact(output_root) -> Path`, `publish_street_artifact(...) -> Path`, `load_street_artifact(..., expected_manifest) -> StreetArtifact`. Independent scaler + `KMeans(200)` только per postflop street; output включает preflop lossless parquet, models, assignments, validation JSON и общий manifest; publish только при quality pass.

- [ ] **Step 1: Написать failing artifact и pipeline smoke tests**

```python
def test_load_rejects_incompatible_manifest(tmp_path):
    publish_street_artifact(tmp_path, Street.FLOP, payload)
    with pytest.raises(ValueError, match="manifest"):
        load_street_artifact(tmp_path, Street.FLOP, incompatible_manifest)

def test_pipeline_does_not_publish_when_quality_gate_fails(tmp_path, monkeypatch):
    with pytest.raises(RuntimeError, match="quality"):
        build_street(...)
    assert not (tmp_path / "flop").exists()
```

Добавить smoke `build_all(sample_count=300, holdout_count=100, clusters=3)` только для tests/`--smoke`.

- [ ] **Step 2: Запустить tests до реализации**

Run: `python -m pytest -q tests/test_card_abstraction_artifacts.py tests/test_card_abstraction_pipeline.py`

Expected: FAIL, модули отсутствуют.

- [ ] **Step 3: Реализовать atomic artifacts и pipeline**

Записать во временную sibling directory и заменить output только после проверки. Manifest: versions feature/canonicalizer, seed, sampling, k-means params, dimension, model SHA-256 и versions Python/dependencies/evaluator. Train scaler only; holdout 20 000; сохранить distances, bucket sizes, silhouette максимум на 10 000, representatives и stability `256 × 256` на 1 000 points. Production применяет все spec quality gates.

- [ ] **Step 4: Реализовать CLI, прогнать smoke и full suite**

Run: `python tools/build_hu_postflop_abstraction.py --smoke --output <tmpdir>`

Expected: lossless preflop artifact и три postflop street artifacts, явный report; solver не импортируется.

Run: `python -m pytest -q tests/test_pokers_showdown.py tests/test_card_abstraction_*.py`

Expected: PASS.

Run: `git diff --check`

Expected: нет ошибок whitespace.

- [ ] **Step 5: Commit task-owned files**

Run: `git add src/card_abstraction/artifacts.py src/card_abstraction/pipeline.py tools/build_hu_postflop_abstraction.py tests/test_card_abstraction_artifacts.py tests/test_card_abstraction_pipeline.py && git commit -m "feat: build validated HU postflop abstractions"`

### Task 6: Production evidence и review gate

**Files:** Create `docs/superpowers/reports/2026-09-27-hu-postflop-card-abstraction-v1.md`.

**Interfaces:** Consumes manifest и validation JSON; produces report с command, versions, checksums, metrics и `pass/reject`; не меняет solver.

- [ ] **Step 1: Создать report template и выполнить production build только после отдельного подтверждения compute budget**

Run: `python tools/build_hu_postflop_abstraction.py --output artifacts/hu_postflop_abstraction/v1`

Expected: три artifacts опубликованы только при quality pass; при reject сохраняется report без publication.

- [ ] **Step 2: Выполнить финальную проверку и закоммитить docs**

Run: `python -m pytest -q tests/test_pokers_showdown.py tests/test_card_abstraction_*.py`

Expected: PASS.

Run: `git add docs/superpowers/reports/2026-09-27-hu-postflop-card-abstraction-v1.md && git commit -m "docs: record HU abstraction validation"`

## Self-review

- Task 2 покрывает lossless preflop 169 и canonicalization; Tasks 1 и 3 — showdown/equity correctness; Task 4 — fixed-size postflop sampling; Task 5 — artifact publication, clustering и validation; Task 6 — controlled production evidence.
- Цепочка типов согласована: `CardSituation -> CanonicalCardKey -> feature -> StreetArtifact`.
- Все пять Review Focus закреплены в Tasks 1–5.
- Resolver, CFR, six-max и history-dependent range намеренно отсутствуют.
