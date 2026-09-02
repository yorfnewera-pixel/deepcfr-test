# Runtime Search Noise Control Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Сделать reach-weighted beliefs независимыми от rollout budget, блокировать policy update без статистически различимого root signal и публиковать paired EV только с измеримой неопределённостью.

**Architecture:** Belief sampler строит 256 proposals и resample 32 rollout particles, поэтому ESS измеряется на distribution proposals. Lockstep rollout возвращает per-action SE и CRN-correct best-vs-runner-up gap; policy делает fallback при недостаточном signal. Paired runner агрегирует per-seed differences в pooled SE, t и CI.

**Tech Stack:** Python 3.11, NumPy, `pokers`, pytest, argparse, JSON.

**Spec:** `docs/superpowers/specs/2026-08-28-runtime-search-noise-control-design.md`

## Global Constraints

- Не менять training loop, checkpoint format, модель или `src/core/action_space.py`.
- Не добавлять runtime sizing/mapping: search остаётся на compact identity actions.
- Default belief budgets: `belief_proposal_count=256`, `belief_particles=32`.
- Fallback при `ESS < 8` или `ESS / proposal_count < 0.03125`.
- Noise gate: `min_root_gap_zscore=1.0`; он использует CRN pairwise difference, не сумму независимых variances.
- Для `rho=uniform`: `eta_effective = eta / (1 + alpha * eta)`.
- Paired report с samples `<10_000` получает `power_status="insufficient_power"`; это не verdict качества.
- В репозитории нет Git, поэтому шаги commit пропускаются.

---

### Task 1: Независимые proposal и resampled particle budgets

**Files:**
- Modify: `src/runtime_search/beliefs.py`
- Modify: `src/runtime_search/policy.py`
- Modify: `tools/run_runtime_paired_evaluation.py`
- Test: `tests/test_runtime_search_beliefs.py`
- Test: `tests/test_runtime_search_policy.py`

**Interfaces:**
- Produces: `sample_reach_weighted_particles` with explicit `proposal_count` and `particle_count` keyword parameters, returning `ReachWeightedBeliefs`.
- Produces: `ReachWeightedBeliefs.proposal_count: int`, `ess: float`, `ess_ratio: float` where `ess_ratio == ess / proposal_count`.
- Produces: `RuntimeSearchConfig.belief_proposal_count`, `belief_min_ess`, `belief_min_ess_ratio`.

- [ ] **Step 1: Write failing beliefs tests for independent counts**

```python
def test_reach_weighted_particles_resample_budget_is_independent_from_proposals():
    beliefs = sample_reach_weighted_particles(
        **valid_history_inputs(),
        proposal_count=16,
        particle_count=4,
        rng=np.random.default_rng(107),
    )

    assert beliefs.proposal_count == 16
    assert len(beliefs.particles) == 4
    assert beliefs.ess_ratio == pytest.approx(beliefs.ess / 16)


def test_reach_weighted_particles_reject_less_proposals_than_particles():
    with pytest.raises(ValueError, match="proposal_count"):
        sample_reach_weighted_particles(
            **valid_history_inputs(),
            proposal_count=3,
            particle_count=4,
            rng=np.random.default_rng(107),
        )
```

- [ ] **Step 2: Run the new tests and verify RED**

Run:

```powershell
python -m pytest tests/test_runtime_search_beliefs.py -q
```

Expected: FAIL because `proposal_count` is not accepted and `ReachWeightedBeliefs` lacks the new field.

- [ ] **Step 3: Implement proposal sampling and config validation**

```python
@dataclass(frozen=True)
class ReachWeightedBeliefs:
    particles: tuple[BeliefParticle, ...]
    proposal_count: int
    ess: float
    ess_ratio: float
    diagnostic_flags: tuple[str, ...]


def sample_reach_weighted_particles(*, proposal_count: int, particle_count: int, **kwargs) -> ReachWeightedBeliefs:
    if proposal_count < particle_count:
        raise ValueError("proposal_count не может быть меньше particle_count")
    proposals = tuple(sample_blocker_aware_particle(**kwargs) for _ in range(proposal_count))
    # log_weights и ESS считаются по proposal_count.
    selected_indices = rng.choice(proposal_count, size=particle_count, replace=True, p=weights)
```

Add to `RuntimeSearchConfig`:

```python
belief_proposal_count: int = 256
belief_particles: int = 32
belief_min_ess: float = 8.0
belief_min_ess_ratio: float = 0.03125
```

Validate positive counts, `belief_proposal_count >= belief_particles`, non-negative absolute ESS and ratio in `[0, 1]`. Pass `belief_proposal_count` to the sampler. In `choose_action`, fallback if either threshold fails and retain `belief_ess`, `belief_ess_ratio`, proposal count and both explicit diagnostic flags.

Add CLI arguments `--proposals`, `--min-ess` and preserve `--min-ess-ratio`; pass them to `RuntimeSearchConfig`.

- [ ] **Step 4: Run focused tests and verify GREEN**

Run:

```powershell
python -m pytest tests/test_runtime_search_beliefs.py tests/test_runtime_search_policy.py -q
```

Expected: PASS; existing reproducibility/card legality tests continue to pass.

### Task 2: CRN-correct root signal statistics

**Files:**
- Modify: `src/runtime_search/rollouts.py`
- Test: `tests/test_runtime_search_rollouts.py`

**Interfaces:**
- Consumes: aligned per-particle rewards produced by `evaluate_root_actions`.
- Produces: `RolloutEvaluation.raw_ev_se: np.ndarray`, `best_gap: float`, `best_gap_se: float`, `best_gap_zscore: float`.

- [ ] **Step 1: Write failing fixture tests for paired difference SE**

```python
def test_rollout_evaluation_uses_common_particle_rewards_for_best_gap_se():
    evaluation = _evaluation_from_rewards(
        np.asarray([[10.0, 20.0, 30.0], [9.0, 19.0, 29.0]])
    )

    assert evaluation.best_gap == pytest.approx(1.0)
    assert evaluation.best_gap_se == pytest.approx(0.0)
    assert math.isinf(evaluation.best_gap_zscore)
```

The helper belongs in the test module; production code must expose observable `RolloutEvaluation` fields, not a test-only API.

- [ ] **Step 2: Run the test and verify RED**

Run:

```powershell
python -m pytest tests/test_runtime_search_rollouts.py -q
```

Expected: FAIL because `RolloutEvaluation` has no signal statistics.

- [ ] **Step 3: Implement rollout statistics from aligned rewards**

```python
raw_ev_se = raw_ev_std / np.sqrt(successful_rollouts)
best_index = int(np.nanargmax(raw_ev_mean))
runner_up_index = int(np.nanargmax(np.where(np.arange(len(raw_ev_mean)) == best_index, np.nan, raw_ev_mean)))
paired_differences = rewards_matrix[best_index] - rewards_matrix[runner_up_index]
best_gap = float(np.mean(paired_differences))
best_gap_se = float(np.std(paired_differences, ddof=1) / np.sqrt(paired_differences.size))
```

Build `rewards_matrix` only after verifying every root action has `len(particles)` rewards; otherwise keep statistics `nan` and existing incomplete-rollout path controls the fallback. For zero paired-difference SE, return `inf` only for a positive gap and `0.0` for a zero gap. Keep `raw_ev_std` behavior unchanged.

- [ ] **Step 4: Run focused tests and verify GREEN**

Run:

```powershell
python -m pytest tests/test_runtime_search_rollouts.py -q
```

Expected: PASS including existing CRN/lockstep tests.

### Task 3: Root noise gate and MMDS transparency

**Files:**
- Modify: `src/runtime_search/policy.py`
- Modify: `src/runtime_search/history.py`
- Test: `tests/test_runtime_search_policy.py`
- Test: `tests/test_runtime_search_history.py`

**Interfaces:**
- Consumes: extended `RolloutEvaluation` from Task 2.
- Produces: `SearchDecision.raw_ev_std`, `raw_ev_se`, `best_gap`, `best_gap_se`, `best_gap_zscore`, `eta_effective`, `belief_proposal_count`.
- Produces: `RuntimeSearchConfig.min_root_gap_zscore: float = 1.0`.

- [ ] **Step 1: Write failing tests for no-signal fallback and eta effective**

```python
def test_choose_action_falls_back_when_root_gap_is_not_above_noise(monkeypatch):
    state = build_response_to_bet_spot()
    monkeypatch.setattr(policy_module, "evaluate_root_actions", lambda **_: noisy_evaluation())
    decision = choose_action(
        state, hero_id=1, blueprint=UniformBlueprint(),
        config=RuntimeSearchConfig(belief_proposal_count=2, belief_particles=2, min_root_gap_zscore=1.0),
        rng=np.random.default_rng(107), observed_decisions=valid_observed_decisions(state),
    )

    assert "rollout_signal_below_noise_blueprint_fallback" in decision.diagnostic_flags
    np.testing.assert_allclose(decision.root_search_policy, decision.root_blueprint_policy)


def test_search_decision_reports_effective_eta_for_uniform_magnet():
    state = build_response_to_bet_spot()
    decision = choose_action(
        state, hero_id=1, blueprint=UniformBlueprint(),
        config=RuntimeSearchConfig(belief_proposal_count=2, belief_particles=2, eta=50.0, alpha=0.05),
        rng=np.random.default_rng(107), observed_decisions=valid_observed_decisions(state),
    )

    assert decision.eta_effective == pytest.approx(50.0 / 3.5)
```

- [ ] **Step 2: Run the tests and verify RED**

Run:

```powershell
python -m pytest tests/test_runtime_search_policy.py tests/test_runtime_search_history.py -q
```

Expected: FAIL because SearchDecision does not expose the statistics and no noise fallback exists.

- [ ] **Step 3: Implement decision fields and gate**

```python
eta_effective = config.eta / (1.0 + config.alpha * config.eta)
if evaluation.best_gap_zscore < config.min_root_gap_zscore:
    return _blueprint_fallback_decision(
        state,
        root_prior=root_prior,
        belief_particles=len(beliefs.particles),
        belief_ess=beliefs.ess,
        belief_ess_ratio=beliefs.ess_ratio,
        diagnostic_flags=(*beliefs.diagnostic_flags, "rollout_signal_below_noise_blueprint_fallback"),
        rng=rng,
        started_at=started_at,
    )
```

Populate the new fields on successful and fallback decisions. Fallbacks without a rollout use `np.nan` for EV/signal values; they never fabricate evidence. Extend `RuntimeSearchPolicy` counters with `noise_gate_fallback_decision_count` for paired reports.

- [ ] **Step 4: Run focused tests and verify GREEN**

Run:

```powershell
python -m pytest tests/test_runtime_search_policy.py tests/test_runtime_search_history.py -q
```

Expected: PASS; blueprint fallback preserves the incoming RNG contract.

### Task 4: Pooled paired statistics and CLI report

**Files:**
- Modify: `src/runtime_search/paired_evaluation.py`
- Modify: `tools/run_runtime_paired_evaluation.py`
- Test: `tests/test_runtime_search_paired_evaluation.py`

**Interfaces:**
- Produces: `RuntimePairedEvaluation.pooled_statistics: PairedStatistics` with `mean_difference`, `std_difference`, `standard_error`, `bb_per_100`, `standard_error_bb_per_100`, `t_statistic`, `ci95_low_bb_per_100`, `ci95_high_bb_per_100`, `power_status`.
- Produces: JSON key `pooled_statistics`.

- [ ] **Step 1: Write a failing known-statistics test**

```python
def test_paired_evaluation_reports_pooled_se_t_ci_and_small_sample_status():
    seed_result = RuntimePairedSeedResult(
        seed=1, samples=2, mean_difference=2.0, std_difference=2.0, bb_per_100=100.0,
        search_decisions=0, reach_weighted_decisions=0, blueprint_fallback_decisions=0,
        action_comparisons=0, action_changes=0,
        search_decisions_by_stage={"flop": 0, "turn": 0, "river": 0},
        action_changes_by_stage={"flop": 0, "turn": 0, "river": 0},
        policy_l1_sum=0.0, policy_l1_max=0.0, policy_argmax_changes=0,
        policy_l1_sum_by_stage={"flop": 0.0, "turn": 0.0, "river": 0.0},
        policy_argmax_changes_by_stage={"flop": 0, "turn": 0, "river": 0},
    )
    evaluation = RuntimePairedEvaluation(runs=(
        seed_result,
        dataclasses.replace(seed_result, seed=2, mean_difference=4.0, bb_per_100=200.0),
    ))

    stats = evaluation.pooled_statistics
    assert stats.mean_difference == pytest.approx(3.0)
    assert stats.power_status == "insufficient_power"
    assert evaluation.to_dict()["pooled_statistics"]["t_statistic"] == pytest.approx(stats.t_statistic)
```

- [ ] **Step 2: Run the test and verify RED**

Run:

```powershell
python -m pytest tests/test_runtime_search_paired_evaluation.py -q
```

Expected: FAIL because `pooled_statistics` is absent.

- [ ] **Step 3: Implement pooled variance and reporting**

```python
pooled_variance = (
    sum((run.samples - 1) * run.std_difference**2 + run.samples * (run.mean_difference - mean)**2 for run in runs)
    / (total_samples - 1)
)
standard_error = math.sqrt(pooled_variance / total_samples)
t_statistic = mean / standard_error if standard_error > 0 else math.nan
power_status = "reported" if total_samples >= 10_000 else "insufficient_power"
```

Validate at least two total samples. Convert chip statistics to BB/100 with `* 100 / bb`. Use `1.96` for the documented normal 95% CI. Add CLI line with BB/100, SE, t, CI and power status. Add `--proposals`, `--min-ess`, `--min-ess-ratio`, `--min-root-gap-zscore` arguments if Task 1/3 has not already wired them.

- [ ] **Step 4: Run focused tests and verify GREEN**

Run:

```powershell
python -m pytest tests/test_runtime_search_paired_evaluation.py -q
```

Expected: PASS with JSON-serializable finite statistics for normal input.

### Task 5: Contract documentation and regression verification

**Files:**
- Modify: `planning/solver_reports/solver.md`
- Create: `planning/solver_reports/020-noise-control-branch-closure.md`
- Test: `tests/test_frozen_blueprint_policy.py`
- Test: `tests/test_paired_evaluation.py`
- Test: `tests/test_mmds.py`
- Test: `tests/test_runtime_search_spots.py`
- Test: `tests/test_runtime_search_beliefs.py`
- Test: `tests/test_runtime_search_rollouts.py`
- Test: `tests/test_runtime_search_policy.py`
- Test: `tests/test_runtime_search_diagnostics.py`
- Test: `tests/test_runtime_search_history.py`
- Test: `tests/test_runtime_search_paired_evaluation.py`

**Interfaces:**
- Consumes: implemented config, SearchDecision, rollout and paired statistics from Tasks 1–4.
- Produces: updated solver contract and branch-closure report without claiming small-sample EV quality.

- [ ] **Step 1: Update `solver.md` contracts**

Add proposal/resample budgets, dual ESS gate, CRN signal gate, `eta_effective`, per-action MC SE and mandatory paired SE/t/CI/power status. Replace wording that treats `eta * mc_values` as the only applied scale with both raw and effective coefficients.

- [ ] **Step 2: Write report 020**

State that 013–019 found a low-particle noise-amplification issue. Preserve raw measurements, retract the unsupported conclusion that eta 20/30 is harmful, and record eta 50 only as a statistically negative low-particle candidate.

- [ ] **Step 3: Run complete targeted regression and compilation**

Run:

```powershell
python -m pytest tests/test_frozen_blueprint_policy.py tests/test_paired_evaluation.py tests/test_mmds.py tests/test_runtime_search_spots.py tests/test_runtime_search_beliefs.py tests/test_runtime_search_rollouts.py tests/test_runtime_search_policy.py tests/test_runtime_search_diagnostics.py tests/test_runtime_search_history.py tests/test_runtime_search_paired_evaluation.py -q
python -m compileall -q src tools\run_turn_river_diagnostics.py tools\run_runtime_paired_evaluation.py
```

Expected: all tests pass and compilation exits zero.

### Task 6: Post-change checkpoint diagnostics

**Files:**
- Create: `planning/solver_reports/021-runtime-policy-delta-noise-control-checkpoint-3000.md`
- Create: `planning/solver_reports/021-runtime-policy-delta-noise-control-checkpoint-3000.json`

**Interfaces:**
- Consumes: CLI from Tasks 1–4 and `models/test110/light_checkpoint_iter_3000.pt` read-only.
- Produces: diagnostic evidence only; no EV quality verdict under 10 000 samples.

- [ ] **Step 1: Run policy-delta diagnostic**

Run:

```powershell
python tools\run_runtime_paired_evaluation.py --checkpoint models\test110\light_checkpoint_iter_3000.pt --output planning\solver_reports\021-runtime-policy-delta-noise-control-checkpoint-3000.json --deals 8 --seeds 107 109 --particles 32 --proposals 256 --eta 10 --min-ess 8 --min-ess-ratio 0.03125 --min-root-gap-zscore 1
```

- [ ] **Step 2: Record only mechanical diagnostics**

Report proposals, ESS/ESS ratio, noise-gate fallback count, policy L1, argmax/action changes, latency and power status. Explicitly label any paired EV as `insufficient_power`.

## Plan Self-Review

- [x] Spec coverage: Tasks 1–4 cover independent budgets, ESS, CRN noise gate, eta effective and paired statistics; Task 5 updates contracts; Task 6 verifies checkpoint behavior.
- [x] No placeholders: every implementation and test step gives concrete names, formulas and commands.
- [x] Interface consistency: Task 1 feeds `ReachWeightedBeliefs` and config into Task 3; Task 2 feeds `RolloutEvaluation` into Task 3; Task 4 serializes the runner used by Task 6.
