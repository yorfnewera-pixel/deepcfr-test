# 8-max postflop card abstraction Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Заменить устаревшую HU card-abstraction воспроизводимым 8-max offline pipeline с 189-мерными multiway equity profiles и проверенными artifacts.

**Architecture:** `src/card_abstraction/` переводится на versioned 8-max contract: семь nested common-pot equity blocks, deterministic shard/cache/resume и pilot-first clustering. Старые 27-мерные HU artifacts и CLI удаляются; MTT policy, solver, движок и checkpoints не меняются.

**Tech Stack:** Python 3.11, NumPy, pandas/PyArrow, scikit-learn, joblib, `pokers` PyO3, pytest.

**Spec:** `docs/superpowers/specs/2026-10-01-8max-postflop-card-abstraction-design.md`

## Global Constraints

- Reference scenarios: hero против 1--7 участников общего showdown, включая all-in players.
- Preflop остаётся 169-class; postflop feature имеет `(189,)`, `float32`.
- Один sample раздаёт 14 разных карт, первые `2*k` образуют nested scenario для `k=1..7`; pairwise averaging запрещён.
- River feature для `k=1..7` использует nested MC; exact HU river -- только validation oracle.
- RNG не зависит от workers, chunking и resume; больший budget сохраняет prefix consistency.
- Train/holdout canonical keys дизъюнктны. Pilot фиксирует thresholds до production holdout.
- Runtime resolver, MTT policy, side-pot utility и production build вне этого плана.

## Review Focus

- Повтор карты у opponents или ненулевая доля банка при проигрыше одному из opponents -- Task 2 tests.
- Смена workers/chunk или рост budget меняет feature prefix -- Task 3 tests.
- Exact HU river попадает в feature вместо независимого oracle -- Task 3 tests.
- Holdout пересекается с train -- Task 4 tests.
- Resume принимает shard от другого estimator spec -- Task 5 tests.

---

### Task 1: 8-max spec и artifact compatibility

**Files:** Create `src/card_abstraction/spec.py`; modify `src/card_abstraction/artifacts.py`, `src/card_abstraction/__init__.py`, `tests/test_card_abstraction_artifacts.py`.

**Interfaces:** Produces `AbstractionSpec.default() -> AbstractionSpec` with `max_table_players=8`, `opponent_counts=(1,2,3,4,5,6,7)`, `feature_dimension=189`; `load_street_artifact(root, street, expected_spec)`.

- [ ] **Step 1: Write failing contract tests**

```python
def test_8max_spec_has_fixed_multiway_layout():
    spec = AbstractionSpec.default()
    assert spec.opponent_counts == (1, 2, 3, 4, 5, 6, 7)
    assert spec.feature_dimension == 189

def test_loader_rejects_other_estimator_with_same_dimension(tmp_path):
    publish_street_artifact(tmp_path, Street.FLOP, artifact, spec_a)
    with pytest.raises(ValueError, match="estimator"):
        load_street_artifact(tmp_path, Street.FLOP, spec_b)
```

- [ ] **Step 2: Run RED**

Run: `python -m pytest -q tests/test_card_abstraction_artifacts.py`

Expected: FAIL because `AbstractionSpec` and spec-aware loading do not exist.

- [ ] **Step 3: Implement immutable spec and manifest**

Record estimator/range/layout versions, sampling budgets, scaling policy, cache-key version and fixed thresholds. Replace the old three-field HU manifest; retain SHA-256 and atomic publication.

- [ ] **Step 4: Verify and commit**

Run: `python -m pytest -q tests/test_card_abstraction_artifacts.py`

Expected: PASS.

Commit task-owned files with message `feat: define 8max abstraction contract`.

### Task 2: Common-pot payoff and nested opponent deals

**Files:** Modify `src/card_abstraction/equity.py`; create `tests/test_card_abstraction_multiway_equity.py`.

**Interfaces:** Produces `deal_nested_opponents(available_cards, rng) -> tuple[Hand, ...]`, `common_pot_share(hero, opponents, board) -> np.ndarray`, and `exact_hu_river_equity(situation) -> float`.

- [ ] **Step 1: Write failing payoff tests**

Test sole win, one stronger opponent after weaker opponents, one/two-way split, all 14 cards unique, and `w(k+1) <= w(k)` for each fixed deal.

```python
def test_common_pot_share_is_zero_when_any_opponent_beats_hero():
    assert common_pot_share(hero, opponents, board)[6] == 0.0
```

- [ ] **Step 2: Run RED**

Run: `python -m pytest -q tests/test_card_abstraction_multiway_equity.py`

Expected: FAIL because multiway functions do not exist.

- [ ] **Step 3: Implement common-pot semantics**

Use public `pkrs.compare_showdown` against each opponent: any loss yields zero; when no loss occurs, divide by hero plus tied opponents. Do not average pairwise outcomes. Keep exact HU enumeration independent from MC code.

- [ ] **Step 4: Verify and commit**

Run: `python -m pytest -q tests/test_card_abstraction_multiway_equity.py tests/test_pokers_showdown.py`

Expected: PASS.

Commit with `feat: add multiway common pot equity`.

### Task 3: Deterministic 189-value profiles

**Files:** Modify `src/card_abstraction/equity.py`, `src/card_abstraction/features.py`; replace `tests/test_card_abstraction_features.py`.

**Interfaces:** Produces `conditional_equities(situation, spec, seed) -> np.ndarray` of `(runouts, 7)` or `(1, 7)` on river; `build_feature(situation, spec, seed) -> np.ndarray` of `(189,)` float32.

- [ ] **Step 1: Write failing feature tests**

Assert seven 27-value blocks, finite values, every histogram sum, quantile order, suit invariance, exact same-seed identity, prefix consistency for larger budget, and river `k=1` MC agreement with exact oracle under a fixed statistical tolerance.

- [ ] **Step 2: Run RED**

Run: `python -m pytest -q tests/test_card_abstraction_features.py`

Expected: FAIL because the existing feature is 27-dimensional HU equity.

- [ ] **Step 3: Implement per-runout RNG and profile layout**

Derive one runout stream and one opponent stream per stable runout index from canonical key and estimator version. Process all seven coordinates from each nested deal. River calculates all coordinates through MC; exact HU remains diagnostic only.

- [ ] **Step 4: Verify and commit**

Run: `python -m pytest -q tests/test_card_abstraction_features.py tests/test_card_abstraction_multiway_equity.py`

Expected: PASS.

Commit with `feat: build 8max multiway equity profiles`.

### Task 4: Disjoint datasets and diagnostic corpus

**Files:** Modify `src/card_abstraction/dataset.py`; create `src/card_abstraction/diagnostics.py`; replace `tests/test_card_abstraction_dataset.py`.

**Interfaces:** Produces `sample_unique_situations(street, count, seed, excluded_keys=frozenset())`; `sample_train_holdout(street, train_count, holdout_count, seed)`; `diagnostic_situations()`.

- [ ] **Step 1: Write failing dataset tests**

Assert train and holdout have no common canonical key, impossible exclusion raises, and diagnostic situations include a made hand, weak pair, draw, monotone board, paired board and board-nuts.

- [ ] **Step 2: Run RED**

Run: `python -m pytest -q tests/test_card_abstraction_dataset.py`

Expected: FAIL because cross-split exclusion and diagnostic corpus do not exist.

- [ ] **Step 3: Implement disjoint sampling**

Collect train keys first, pass them as immutable exclusions to holdout, and return a hard error on exhaustion.

- [ ] **Step 4: Verify and commit**

Run: `python -m pytest -q tests/test_card_abstraction_dataset.py tests/test_card_abstraction_canonical.py`

Expected: PASS.

Commit with `feat: sample disjoint 8max card datasets`.

### Task 5: Deterministic shard/cache/resume feature store

**Files:** Create `src/card_abstraction/feature_store.py`, `tests/test_card_abstraction_feature_store.py`.

**Interfaces:** Produces `FeatureStore(root, spec)` and `FeatureStore.build(situations, *, workers, chunk_size, resume) -> FeatureBatch`; batch reports ordered keys, features, elapsed seconds and peak RSS.

- [ ] **Step 1: Write failing store tests**

On a tiny fixed dataset assert resume skips completed shards, worker/chunk choices produce byte-identical ordered features, and a changed spec cannot reuse old shards.

- [ ] **Step 2: Run RED**

Run: `python -m pytest -q tests/test_card_abstraction_feature_store.py`

Expected: FAIL because the store does not exist.

- [ ] **Step 3: Implement atomic shards and workers**

Pass serializable canonical keys to workers; reconstruct cards inside each worker. Name shards by ordered key range plus full spec hash, atomically write them, merge by canonical-key order, and record speed/RSS. Only exact spec hash may resume.

- [ ] **Step 4: Verify and commit**

Run: `python -m pytest -q tests/test_card_abstraction_feature_store.py tests/test_card_abstraction_features.py`

Expected: PASS.

Commit with `feat: add resumable multiway feature batches`.

### Task 6: Pilot-first clustering and validation

**Files:** Modify `src/card_abstraction/pipeline.py`, `src/card_abstraction/artifacts.py`; replace `tests/test_card_abstraction_pipeline.py`.

**Interfaces:** Produces `run_pilot(output_root, spec, ...) -> PilotReport` and `build_all(output_root, spec, *, workers, chunk_size, resume, quality_thresholds) -> dict[Street, Path]`.

- [ ] **Step 1: Write failing pipeline tests**

Assert smoke emits 8-max artifacts; rejected gate does not publish; validation contains original-equity per-opponent errors, scaling contribution diagnostics, enlarged-budget and independent-seed stability, bounded silhouette sample, and production refuses implicit post-hoc thresholds.

- [ ] **Step 2: Run RED**

Run: `python -m pytest -q tests/test_card_abstraction_pipeline.py`

Expected: FAIL because the pipeline assumes HU features and implicit thresholds.

- [ ] **Step 3: Implement pilot and production lifecycle**

Fit scaler/KMeans only on train. Pilot emits immutable thresholds. Production consumes those thresholds with a separate holdout; report exact-HU river oracle error, MC errors `k=1..7`, two stability measures, bucket metrics and scaling diagnostics. Publish all streets atomically only after full pass.

- [ ] **Step 4: Verify and commit**

Run: `python -m pytest -q tests/test_card_abstraction_artifacts.py tests/test_card_abstraction_pipeline.py tests/test_card_abstraction_feature_store.py`

Expected: PASS.

Commit with `feat: validate 8max card abstraction artifacts`.

### Task 7: New CLI and HU-card migration cleanup

**Files:** Create `tools/build_8max_postflop_abstraction.py`, `docs/superpowers/reports/2026-10-01-8max-postflop-card-abstraction-pilot.md`; delete `tools/build_hu_postflop_abstraction.py`; modify `tests/test_card_abstraction_pipeline.py`.

**Interfaces:** CLI accepts `--output`, `--seed`, `--train-count`, `--holdout-count`, `--clusters`, `--runout-samples`, `--opponent-samples`, `--river-samples`, `--workers`, `--chunk-size`, `--resume`, `--pilot`, `--smoke`.

- [ ] **Step 1: Write failing CLI tests**

Assert `--smoke` reports progress and writes only 8-max layout; `--pilot` produces threshold/report candidate; malformed counts or incompatible resume exit nonzero without publication.

- [ ] **Step 2: Run RED**

Run: `python -m pytest -q tests/test_card_abstraction_pipeline.py`

Expected: FAIL because the new CLI does not exist.

- [ ] **Step 3: Implement CLI and remove HU card CLI**

Flush progress after each feature shard/stage. Production requires a pilot threshold file and never derives thresholds from its own holdout. Report command, spec hash, budgets, workers/chunks, speed, memory, metrics, thresholds and pass/reject. Do not run production in this task.

- [ ] **Step 4: Verify and commit**

Run: `python -m pytest -q tests/test_pokers_showdown.py tests/test_card_abstraction_canonical.py tests/test_card_abstraction_multiway_equity.py tests/test_card_abstraction_features.py tests/test_card_abstraction_dataset.py tests/test_card_abstraction_feature_store.py tests/test_card_abstraction_artifacts.py tests/test_card_abstraction_pipeline.py`

Expected: PASS.

Run: `python -m compileall -q src/card_abstraction tools/build_8max_postflop_abstraction.py`

Expected: PASS.

Commit task-owned files with message `feat: build 8max postflop card abstraction`.

## Self-review

- Tasks 1--3 replace HU card semantics with the versioned 8-max estimator and its correctness tests.
- Task 4 owns disjoint splits and diagnostic situations.
- Task 5 owns the performance/resume contract before any large build.
- Task 6 fixes pilot thresholds before production holdout and owns publication safety.
- Task 7 exposes the only supported CLI and removes the HU card entry point.
- MTT policy, resolver, tournament utility and side-pot eligibility are intentionally absent.
