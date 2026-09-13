import numpy as np
import pokers as pkrs
import pytest
from types import SimpleNamespace

from src.core.action_space import ActionSlot, NUM_ACTIONS, legal_action_mask
from src.runtime_search.beliefs import ObservedDecision
from src.runtime_search.beliefs import ReachWeightedBeliefs
from src.runtime_search.policy import RuntimeSearchConfig, choose_action
import src.runtime_search.policy as policy_module
from src.runtime_search.spots import build_constructed_spot, build_response_to_bet_spot


class UniformBatchBlueprint:
    def probabilities(self, state, player_id=None):
        del player_id
        mask = legal_action_mask(state)
        return mask / mask.sum()

    def probabilities_batch(self, states):
        return np.stack([self.probabilities(state) for state in states])


class BrokenContinuationBlueprint(UniformBatchBlueprint):
    def probabilities_batch(self, states):
        return np.zeros((len(states), NUM_ACTIONS), dtype=np.float64)


class HandSensitiveReachBlueprint(UniformBatchBlueprint):
    def probabilities(self, state, player_id=None):
        actor_id = int(state.current_player if player_id is None else player_id)
        mask = legal_action_mask(state).astype(bool)
        probabilities = np.zeros(NUM_ACTIONS, dtype=np.float64)
        legal_slots = np.flatnonzero(mask)
        check_slot = int(ActionSlot.CHECK)
        if actor_id == 0 and mask[check_slot]:
            high_rank = int(state.players_state[actor_id].hand[0].rank) >= 8
            probabilities[check_slot] = 0.95 if high_rank else 0.05
            other_slots = legal_slots[legal_slots != check_slot]
            probabilities[other_slots] = (1.0 - probabilities[check_slot]) / len(other_slots)
            return probabilities
        probabilities[legal_slots] = 1.0 / len(legal_slots)
        return probabilities


def _observed_history():
    history_state = build_constructed_spot(pkrs.Stage.Turn)
    return (ObservedDecision(state=history_state, actor_id=0, action_slot=ActionSlot.CHECK),)


def _rollout_evaluation(
    *,
    particles=8,
    successful_rollouts=None,
    raw_ev_mean=None,
    best_gap=1.0,
    best_gap_se=0.25,
    best_gap_zscore=4.0,
    best_gap_sample_count=8,
):
    raw_ev_mean = np.asarray(
        (4.0, 2.0, 1.0, 0.0) if raw_ev_mean is None else raw_ev_mean,
        dtype=np.float64,
    )
    successful_rollouts = np.full(
        raw_ev_mean.size,
        particles if successful_rollouts is None else successful_rollouts,
        dtype=np.int64,
    )
    return SimpleNamespace(
        raw_ev_mean=raw_ev_mean,
        raw_ev_std=np.zeros_like(raw_ev_mean),
        raw_ev_se=np.zeros_like(raw_ev_mean),
        best_gap=best_gap,
        best_gap_se=best_gap_se,
        best_gap_zscore=best_gap_zscore,
        best_gap_sample_count=best_gap_sample_count,
        successful_rollouts=successful_rollouts,
        total_particles=particles,
        failure_messages=(),
        terminal_completion_rate=1.0,
    )


def _stubbed_search(monkeypatch, evaluation):
    particles = tuple(object() for _ in range(evaluation.total_particles))
    beliefs = ReachWeightedBeliefs(
        particles=particles,
        proposal_count=len(particles),
        ess=float(len(particles)),
        ess_ratio=1.0,
        resample_count=0,
        diagnostic_flags=("reach_weighted_beliefs",),
    )
    monkeypatch.setattr(policy_module, "sample_reach_weighted_particles", lambda **_: beliefs)
    monkeypatch.setattr(policy_module, "evaluate_root_actions", lambda *_args, **_kwargs: evaluation)
    return choose_action(
        build_response_to_bet_spot(),
        hero_id=1,
        blueprint=UniformBatchBlueprint(),
        config=RuntimeSearchConfig(belief_particles=evaluation.total_particles, policy_floor=0.0),
        rng=np.random.default_rng(107),
        observed_decisions=_observed_history(),
    )


def test_runtime_config_rejects_root_gap_sample_threshold_below_two():
    with pytest.raises(ValueError, match="min_root_gap_samples"):
        RuntimeSearchConfig(min_root_gap_samples=1)


@pytest.mark.parametrize("threshold", (float("nan"), float("inf"), float("-inf"), -0.1, True))
def test_runtime_config_rejects_nonfinite_or_invalid_root_gap_zscore(threshold):
    with pytest.raises(ValueError, match="min_root_gap_zscore"):
        RuntimeSearchConfig(min_root_gap_zscore=threshold)


@pytest.mark.parametrize(
    "sample_count",
    (True, 1.0, 1.5, float("nan"), float("inf"), float("-inf")),
)
def test_runtime_config_rejects_noninteger_root_gap_sample_threshold(sample_count):
    with pytest.raises(ValueError, match="min_root_gap_samples"):
        RuntimeSearchConfig(min_root_gap_samples=sample_count)


def test_invalid_runtime_config_is_rejected_before_policy_update(monkeypatch):
    def unexpected_update(*_args, **_kwargs):
        pytest.fail("недопустимый config не должен запускать обновление policy")

    monkeypatch.setattr(policy_module, "mmds_update", unexpected_update)

    with pytest.raises(ValueError, match="min_root_gap_zscore"):
        RuntimeSearchConfig(min_root_gap_zscore=float("nan"))


@pytest.mark.parametrize(
    ("evaluation", "expected_flag"),
    [
        (_rollout_evaluation(successful_rollouts=7), "rollout_incomplete"),
        (_rollout_evaluation(best_gap_sample_count=7), "rollout_insufficient_gap_samples"),
        (_rollout_evaluation(best_gap=np.nan), "rollout_gap_undefined"),
        (_rollout_evaluation(best_gap_se=np.nan), "rollout_gap_se_undefined"),
        (_rollout_evaluation(best_gap_zscore=np.nan), "rollout_zscore_undefined"),
        (_rollout_evaluation(best_gap_zscore=-np.inf), "rollout_signal_below_noise"),
        (_rollout_evaluation(best_gap=0.0, best_gap_se=0.0, best_gap_zscore=0.0), "rollout_zscore_undefined"),
        (_rollout_evaluation(best_gap=0.0, best_gap_se=0.0, best_gap_zscore=np.inf), "rollout_zscore_undefined"),
        (_rollout_evaluation(best_gap_zscore=0.5), "rollout_signal_below_noise"),
        (_rollout_evaluation(raw_ev_mean=(np.nan, 2.0, 1.0, 0.0)), "rollout_nonfinite_values"),
    ],
)
def test_root_policy_rejects_invalid_runtime_signal_with_blueprint_identity(
    monkeypatch, evaluation, expected_flag
):
    decision = _stubbed_search(monkeypatch, evaluation)

    assert np.array_equal(decision.root_search_policy, decision.root_blueprint_policy)
    assert expected_flag in decision.diagnostic_flags


def test_root_policy_rejects_positive_infinity_when_paired_samples_are_undersampled(monkeypatch):
    decision = _stubbed_search(
        monkeypatch,
        _rollout_evaluation(best_gap=1.0, best_gap_se=0.0, best_gap_zscore=np.inf, best_gap_sample_count=7),
    )

    assert np.array_equal(decision.root_search_policy, decision.root_blueprint_policy)
    assert "rollout_insufficient_gap_samples" in decision.diagnostic_flags


def test_root_policy_updates_only_for_valid_positive_infinite_signal(monkeypatch):
    decision = _stubbed_search(
        monkeypatch,
        _rollout_evaluation(best_gap=1.0, best_gap_se=0.0, best_gap_zscore=np.inf),
    )
    legal = legal_action_mask(build_response_to_bet_spot()).astype(bool)

    assert "rollout_zero_variance_positive_signal" in decision.diagnostic_flags
    assert not np.array_equal(decision.root_search_policy, decision.root_blueprint_policy)
    assert np.isclose(decision.root_search_policy.sum(), 1.0)
    assert np.all(np.isfinite(decision.root_search_policy))
    assert np.all(decision.root_search_policy >= 0.0)
    assert np.all(decision.root_search_policy[~legal] == 0.0)


def test_root_policy_submits_only_finite_values_to_mmds(monkeypatch):
    evaluation = _rollout_evaluation()
    original_update = policy_module.mmds_update

    def finite_mmds_update(pi, mc_values, mask, eta, alpha, **kwargs):
        assert np.all(np.isfinite(pi))
        assert np.all(np.isfinite(mc_values))
        assert np.all(np.isfinite(mc_values[mask]))
        return original_update(pi, mc_values, mask, eta, alpha, **kwargs)

    monkeypatch.setattr(policy_module, "mmds_update", finite_mmds_update)
    decision = _stubbed_search(monkeypatch, evaluation)

    assert not np.array_equal(decision.root_search_policy, decision.root_blueprint_policy)


def test_root_policy_falls_back_when_default_sample_gate_rejects_two_particles():
    state = build_response_to_bet_spot()
    decision = choose_action(
        state,
        hero_id=1,
        blueprint=UniformBatchBlueprint(),
        config=RuntimeSearchConfig(belief_particles=2, eta=0.0, policy_floor=0.0),
        rng=np.random.default_rng(107),
        observed_decisions=_observed_history(),
    )
    mask = legal_action_mask(state).astype(bool)

    assert decision.action.action in state.legal_actions
    assert decision.root_blueprint_policy.shape == (NUM_ACTIONS,)
    assert decision.root_search_policy.shape == (NUM_ACTIONS,)
    assert np.isclose(decision.root_blueprint_policy.sum(), 1.0)
    assert np.isclose(decision.root_search_policy.sum(), 1.0)
    assert np.all(decision.root_blueprint_policy[~mask] == 0.0)
    assert np.all(decision.root_search_policy[~mask] == 0.0)
    assert np.allclose(decision.root_search_policy, decision.root_blueprint_policy)
    assert np.all(np.isnan(decision.mc_values[mask]))
    assert "rollout_insufficient_gap_samples" in decision.diagnostic_flags
    assert decision.belief_particles == 2
    assert "reach_weighted_beliefs" in decision.diagnostic_flags
    assert decision.belief_ess is not None
    assert decision.belief_ess_ratio is not None
    assert decision.belief_resample_count == 0


def test_root_policy_falls_back_when_default_sample_gate_rejects_one_particle():
    state = build_constructed_spot(pkrs.Stage.Flop)
    history = (ObservedDecision(state=state, actor_id=0, action_slot=ActionSlot.CHECK),)
    decision = choose_action(
        state,
        hero_id=0,
        blueprint=UniformBatchBlueprint(),
        config=RuntimeSearchConfig(belief_particles=1, eta=0.0, policy_floor=0.0),
        rng=np.random.default_rng(107),
        observed_decisions=history,
    )
    mask = legal_action_mask(state).astype(bool)

    assert decision.action.action in state.legal_actions
    assert np.isclose(decision.root_search_policy.sum(), 1.0)
    assert np.all(decision.root_search_policy[~mask] == 0.0)
    assert np.all(np.isnan(decision.mc_values[mask]))
    assert "rollout_insufficient_gap_samples" in decision.diagnostic_flags
    assert "reach_weighted_beliefs" in decision.diagnostic_flags


def test_root_policy_falls_back_to_blueprint_when_rollout_is_incomplete():
    state = build_response_to_bet_spot()
    decision = choose_action(
        state,
        hero_id=1,
        blueprint=BrokenContinuationBlueprint(),
        config=RuntimeSearchConfig(belief_particles=1),
        rng=np.random.default_rng(107),
        observed_decisions=_observed_history(),
    )

    assert decision.action.action in state.legal_actions
    assert np.array_equal(decision.root_search_policy, decision.root_blueprint_policy)
    assert "rollout_incomplete" in decision.diagnostic_flags


def test_root_policy_falls_back_without_observed_history():
    state = build_response_to_bet_spot()
    decision = choose_action(
        state,
        hero_id=1,
        blueprint=UniformBatchBlueprint(),
        config=RuntimeSearchConfig(belief_particles=2),
        rng=np.random.default_rng(107),
    )

    assert decision.action.action in state.legal_actions
    assert np.array_equal(decision.root_search_policy, decision.root_blueprint_policy)
    assert decision.belief_particles == 0
    assert decision.belief_ess is None
    assert "belief_history_missing_blueprint_fallback" in decision.diagnostic_flags


def test_root_policy_falls_back_when_reach_ess_is_low():
    state = build_response_to_bet_spot()
    decision = choose_action(
        state,
        hero_id=1,
        blueprint=HandSensitiveReachBlueprint(),
        config=RuntimeSearchConfig(belief_particles=8, belief_min_ess_ratio=0.99),
        rng=np.random.default_rng(107),
        observed_decisions=_observed_history(),
    )

    assert decision.action.action in state.legal_actions
    assert np.array_equal(decision.root_search_policy, decision.root_blueprint_policy)
    assert decision.belief_ess is not None
    assert decision.belief_ess_ratio < 0.99
    assert "belief_low_ess_blueprint_fallback" in decision.diagnostic_flags


def test_runtime_config_rejects_invalid_sequential_resampling_threshold():
    with pytest.raises(ValueError, match="belief_resample_ess_ratio"):
        RuntimeSearchConfig(belief_resample_ess_ratio=0.0)
