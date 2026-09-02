# Reach-Weighted Beliefs Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Сэмплировать solver-valid opponent particles по reach вероятности frozen blueprint и безопасно отключать MMDS при низком ESS.

**Architecture:** `beliefs.py` получает immutable историю observed decisions и выполняет proposal, log-space weighting, ESS и resampling. `policy.py` передаёт историю в sampler и возвращает blueprint-only decision при отсутствии истории или низком ESS; rollout contract не меняется.

**Tech Stack:** Python 3.11, NumPy, локальный `pokers`, pytest.

**Spec:** `docs/superpowers/specs/2026-08-28-reach-weighted-beliefs-design.md`

## Global Constraints

- Не менять training loop, checkpoint format, сеть, `src/core/action_space.py` или `pokers/`.
- Не читать hidden opponent cards или `state.deck` при belief sampling.
- History отсутствует или ESS низок: runtime search не применяет MMDS и возвращает blueprint prior с diagnostic flag.
- Одна resampled particle обслуживает все root actions в существующем CRN rollout.
- Рабочая папка не является Git repository; коммиты не выполнять.

---

### Task 1: History-aware weighted sampler

**Files:**
- Modify: `src/runtime_search/beliefs.py`
- Modify: `tests/test_runtime_search_beliefs.py`

**Interfaces:**
- Produces: `ObservedDecision(state, actor_id, action_slot)`.
- Produces: `ReachWeightedBeliefs(particles, ess, ess_ratio, diagnostic_flags)`.
- Produces: `sample_reach_weighted_particles(hero_id, hero_hand, public_cards, num_opponents, runout_cards, particle_count, observed_decisions, blueprint, rng, reach_probability_floor) -> ReachWeightedBeliefs`.

- [x] **Step 1: Write failing sampler tests**

```python
def test_reach_weighted_particles_are_solver_valid_and_reproducible():
    result = sample_reach_weighted_particles(..., observed_decisions=(decision,), ...)
    assert result.ess < 8
    assert result.ess_ratio < 1.0
    assert all(particle.is_solver_valid for particle in result.particles)

def test_reach_weighted_particles_reject_illegal_history_slot():
    with pytest.raises(ValueError, match="недопустимый"):
        sample_reach_weighted_particles(..., observed_decisions=(bad_decision,), ...)
```

- [x] **Step 2: Run sampler tests and verify import failure**

Run: `python -m pytest tests/test_runtime_search_beliefs.py -q`

Expected: FAIL because the reach-weighted public API is absent.

- [x] **Step 3: Implement proposal, likelihood and resampling**

```python
log_weights[index] += np.log(max(probabilities[action_slot], reach_probability_floor))
weights = np.exp(log_weights - np.logaddexp.reduce(log_weights))
ess = 1.0 / float(np.dot(weights, weights))
indices = rng.choice(len(proposals), size=len(proposals), replace=True, p=weights)
```

Build temporary snapshot states with particle opponent hands and an independently shuffled remaining deck; retain only hero hand from source state. Validate `actor_id == state.current_player` and the compact action slot against `legal_action_mask`.

- [x] **Step 4: Run sampler tests and verify pass**

Run: `python -m pytest tests/test_runtime_search_beliefs.py -q`

Expected: PASS.

### Task 2: S6 policy integration and fallback

**Files:**
- Modify: `src/runtime_search/policy.py`
- Modify: `src/runtime_search/__init__.py`
- Modify: `tests/test_runtime_search_policy.py`

**Interfaces:**
- Consumes: `ObservedDecision`, `ReachWeightedBeliefs`, weighted sampler from Task 1.
- Produces: `RuntimeSearchConfig.belief_min_ess_ratio`, `RuntimeSearchConfig.reach_probability_floor`.
- Produces: `choose_action(..., observed_decisions: Sequence[ObservedDecision] | None = None) -> SearchDecision` with `belief_ess` and `belief_ess_ratio`.

- [x] **Step 1: Write failing policy tests**

```python
def test_root_policy_falls_back_without_observed_history():
    decision = choose_action(state, ..., observed_decisions=None)
    assert "belief_history_missing_blueprint_fallback" in decision.diagnostic_flags

def test_root_policy_falls_back_when_reach_ess_is_low():
    decision = choose_action(state, ..., observed_decisions=history)
    assert "belief_low_ess_blueprint_fallback" in decision.diagnostic_flags
    assert np.array_equal(decision.root_search_policy, decision.root_blueprint_policy)
```

- [x] **Step 2: Run policy tests and verify failure**

Run: `python -m pytest tests/test_runtime_search_policy.py -q`

Expected: FAIL because `observed_decisions` and ESS diagnostics are absent.

- [x] **Step 3: Integrate only valid beliefs**

```python
if not observed_decisions:
    return blueprint_only_decision(..., "belief_history_missing_blueprint_fallback")
beliefs = sample_reach_weighted_particles(...)
if beliefs.ess_ratio < config.belief_min_ess_ratio:
    return blueprint_only_decision(..., "belief_low_ess_blueprint_fallback")
```

Run existing S3 rollouts only after the two guards. Use weighted resampled particles and add `reach_weighted_beliefs` to successful-path diagnostics.

- [x] **Step 4: Run policy tests and verify pass**

Run: `python -m pytest tests/test_runtime_search_policy.py -q`

Expected: PASS.

### Task 3: Report and validation

**Files:**
- Create: `planning/solver_reports/007-reach-weighted-beliefs.md`

- [x] **Step 1: Run targeted S1-S6 tests**

Run: `python -m pytest tests/test_frozen_blueprint_policy.py tests/test_paired_evaluation.py tests/test_mmds.py tests/test_runtime_search_spots.py tests/test_runtime_search_beliefs.py tests/test_runtime_search_rollouts.py tests/test_runtime_search_policy.py tests/test_runtime_search_diagnostics.py -q`

Expected: exit code `0`.

- [x] **Step 2: Run static compilation**

Run: `python -m compileall -q src tools/run_turn_river_diagnostics.py`

Expected: exit code `0`.

- [x] **Step 3: Record S6 scope and risks**

Document action-history contract, log-space weighting, ESS threshold behavior, validation output and that paired EV is now eligible but not yet executed.
