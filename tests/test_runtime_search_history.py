import numpy as np
import pokers as pkrs

from src.core.action_space import ActionSlot, NUM_ACTIONS, legal_action_mask, resolve_action
from src.runtime_search.history import ActionHistoryRecorder, RuntimeSearchPolicy
from src.runtime_search.policy import RuntimeSearchConfig
from src.runtime_search.spots import build_constructed_spot, build_response_to_bet_spot


class UniformBlueprint:
    def probabilities(self, state, player_id=None):
        del player_id
        mask = legal_action_mask(state)
        return mask / mask.sum()

    def probabilities_batch(self, states):
        return np.stack([self.probabilities(state) for state in states])

    def choose_action(self, state, rng):
        slot = int(rng.choice(NUM_ACTIONS, p=self.probabilities(state)))
        return resolve_action(slot, state).action


def test_history_recorder_keeps_compact_opponent_decision():
    state = build_constructed_spot(pkrs.Stage.Turn)
    recorder = ActionHistoryRecorder(hero_id=1)
    action = resolve_action(ActionSlot.CHECK, state).action

    recorder.observe_action(state, action)

    assert recorder.is_complete
    assert len(recorder.observed_decisions) == 1
    decision = recorder.observed_decisions[0]
    assert decision.state is state
    assert decision.actor_id == 0
    assert decision.action_slot == ActionSlot.CHECK


def test_history_recorder_marks_noncompact_opponent_raise_incomplete():
    state = build_constructed_spot(pkrs.Stage.Turn)
    recorder = ActionHistoryRecorder(hero_id=1)

    recorder.observe_action(state, pkrs.Action(pkrs.ActionEnum.Raise, 10.0))

    assert not recorder.is_complete
    assert recorder.observed_decisions == ()
    assert recorder.diagnostic_flags == ("history_contains_noncompact_opponent_action",)


def test_runtime_policy_uses_blueprint_when_no_history_is_available():
    state = build_response_to_bet_spot()
    policy = RuntimeSearchPolicy(
        hero_id=1,
        blueprint=UniformBlueprint(),
        config=RuntimeSearchConfig(belief_particles=2),
    )
    policy.reset_hand()

    action = policy.choose_action(state, np.random.default_rng(107))

    assert action.action in state.legal_actions
    assert policy.last_search_decision is not None
    assert "belief_history_missing_blueprint_fallback" in policy.last_search_decision.diagnostic_flags


def test_runtime_policy_activates_on_flop_no_bet_root():
    state = build_constructed_spot(pkrs.Stage.Flop)
    policy = RuntimeSearchPolicy(
        hero_id=0,
        blueprint=UniformBlueprint(),
        config=RuntimeSearchConfig(belief_particles=1),
    )
    policy.reset_hand()

    action = policy.choose_action(state, np.random.default_rng(107))

    assert action.action in state.legal_actions
    assert policy.last_search_decision is not None
    assert "belief_history_missing_blueprint_fallback" in policy.last_search_decision.diagnostic_flags


def test_runtime_policy_compares_blueprint_action_without_changing_rng():
    state = build_constructed_spot(pkrs.Stage.Flop)
    blueprint = UniformBlueprint()
    policy = RuntimeSearchPolicy(
        hero_id=0,
        blueprint=blueprint,
        config=RuntimeSearchConfig(belief_particles=1),
    )
    expected_rng = np.random.default_rng(107)
    expected_action = blueprint.choose_action(state, expected_rng)
    expected_next_random = expected_rng.integers(0, 1_000_000)

    rng = np.random.default_rng(107)
    action = policy.choose_action(state, rng)

    assert action.action == expected_action.action
    assert policy.action_comparison_count == 1
    assert policy.action_change_count == 0
    assert policy.search_decisions_by_stage == {"flop": 1, "turn": 0, "river": 0}
    assert policy.action_changes_by_stage == {"flop": 0, "turn": 0, "river": 0}
    assert policy.policy_l1_sum == 0.0
    assert policy.policy_l1_max == 0.0
    assert policy.policy_argmax_change_count == 0
    assert policy.policy_l1_sum_by_stage == {"flop": 0.0, "turn": 0.0, "river": 0.0}
    assert policy.policy_argmax_changes_by_stage == {"flop": 0, "turn": 0, "river": 0}
    assert rng.integers(0, 1_000_000) == expected_next_random
