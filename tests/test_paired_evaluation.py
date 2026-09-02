import numpy as np
import pokers as pkrs

from src.evaluation.paired_harness import (
    DeterministicLegalPolicy,
    PairedEvaluation,
    evaluate_paired,
)


class RngLegalPolicy:
    """Uses supplied CRN, so equal duplicate arms must still be identical."""

    def choose_action(self, state, rng):
        legal = list(state.legal_actions)
        legal = [action for action in legal if action != pkrs.ActionEnum.Raise]
        action = legal[int(rng.integers(len(legal)))]
        return pkrs.Action(action)


class HookTrackingPolicy(RngLegalPolicy):
    def __init__(self):
        self.reset_calls = 0
        self.hero_ids: list[int] = []
        self.observed_players: list[int] = []

    def reset_hand(self):
        self.reset_calls += 1

    def set_hero_id(self, hero_id):
        self.hero_ids.append(hero_id)

    def observe_action(self, state, action):
        del action
        self.observed_players.append(int(state.current_player))


def test_identity_baseline_has_exact_zero_paired_difference_with_rotation_and_crn():
    policy = RngLegalPolicy()
    result = evaluate_paired(
        policy,
        policy,
        num_deals=4,
        seed=107,
        num_players=6,
        rotate_seats=True,
    )

    assert result.deals == 4
    assert result.seats == 6
    assert result.samples == 24
    assert np.array_equal(result.differences, np.zeros(24))
    assert result.mean_difference == 0.0
    assert result.std_difference == 0.0
    assert result.bb_per_100 == 0.0


def test_harness_is_reproducible_and_rotation_can_be_disabled():
    policy = DeterministicLegalPolicy()
    first = evaluate_paired(policy, policy, num_deals=3, seed=5, rotate_seats=False)
    second = evaluate_paired(policy, policy, num_deals=3, seed=5, rotate_seats=False)

    assert first.seats == 1
    assert first.samples == 3
    assert np.array_equal(first.baseline_rewards, second.baseline_rewards)
    assert np.array_equal(first.candidate_rewards, second.candidate_rewards)
    assert np.array_equal(first.differences, second.differences)


def test_harness_rejects_illegal_policy_action():
    class IllegalPolicy:
        def choose_action(self, state, rng):
            return pkrs.Action(pkrs.ActionEnum.Raise, amount=0.0)

    try:
        evaluate_paired(IllegalPolicy(), IllegalPolicy(), num_deals=1, seed=1)
    except RuntimeError as exc:
        assert "engine rejected" in str(exc)
    else:
        raise AssertionError("illegal policy action was accepted")


def test_bb_per_100_normalizes_reward_difference_by_big_blind():
    result = PairedEvaluation(
        differences=np.array([4.0, 8.0]),
        baseline_rewards=np.array([0.0, 0.0]),
        candidate_rewards=np.array([4.0, 8.0]),
        deals=2,
        seats=1,
        bb=2.0,
    )

    assert result.bb_per_100 == 300.0


def test_paired_evaluation_notifies_optional_history_hooks():
    tracking_policy = HookTrackingPolicy()
    result = evaluate_paired(
        DeterministicLegalPolicy(),
        tracking_policy,
        num_deals=1,
        seed=107,
        num_players=2,
        rotate_seats=False,
    )

    assert result.samples == 1
    assert tracking_policy.reset_calls == 1
    assert tracking_policy.hero_ids == [0]
    assert tracking_policy.observed_players
