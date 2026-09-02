import numpy as np
import pokers as pkrs
import pytest

from src.core.action_space import ActionSlot, NUM_ACTIONS, legal_action_mask
from src.runtime_search.beliefs import ObservedDecision
from src.runtime_search.policy import RuntimeSearchConfig, choose_action
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


def test_root_policy_uses_identity_compact_prior_and_value_scaling():
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
    assert np.all(np.isfinite(decision.mc_values[mask]))
    assert decision.belief_particles == 2
    assert "reach_weighted_beliefs" in decision.diagnostic_flags
    assert decision.belief_ess is not None
    assert decision.belief_ess_ratio is not None
    assert decision.belief_resample_count == 0


def test_root_policy_supports_flop_no_bet_identity_actions():
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
    assert np.all(np.isfinite(decision.mc_values[mask]))
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
