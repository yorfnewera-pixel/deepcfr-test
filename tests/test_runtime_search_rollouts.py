import numpy as np
import pokers as pkrs

from src.core.action_space import ActionSlot, legal_action_mask, resolve_action
from src.runtime_search.beliefs import sample_blocker_aware_particle
from src.runtime_search.cards import card_key
from src.runtime_search.rollouts import _root_signal_statistics, evaluate_root_actions, materialize_particle_state
from src.runtime_search.spots import build_constructed_spot


class PassiveBatchPolicy:
    """Выбирает первый допустимый passive slot без зависимости от rollout branch."""

    def probabilities_batch(self, states):
        probabilities = np.zeros((len(states), 6), dtype=np.float64)
        for index, state in enumerate(states):
            mask = legal_action_mask(state)
            for slot in (ActionSlot.CHECK, ActionSlot.CALL, ActionSlot.FOLD):
                if mask[int(slot)]:
                    probabilities[index, int(slot)] = 1.0
                    break
        return probabilities


def test_root_signal_uses_paired_particle_differences():
    gap, gap_se, zscore = _root_signal_statistics(
        (
            [10.0, 20.0, 30.0],
            [9.0, 19.0, 29.0],
        )
    )

    assert gap == 1.0
    assert gap_se == 0.0
    assert np.isinf(zscore)


def test_root_signal_selects_pair_on_other_particle_half():
    gap, gap_se, zscore = _root_signal_statistics(
        (
            [5.0, 5.0, 0.0, 0.0],
            [0.0, 0.0, 2.0, 2.0],
        )
    )

    assert gap == -2.0
    assert gap_se == 0.0
    assert zscore == -np.inf


def _particle(state, seed):
    return sample_blocker_aware_particle(
        hero_hand=state.players_state[0].hand,
        public_cards=state.public_cards,
        num_opponents=5,
        runout_cards=1,
        rng=np.random.default_rng(seed),
    )


def test_materialized_particle_uses_particle_cards_and_not_source_deck():
    state = build_constructed_spot(pkrs.Stage.Turn)
    particle = _particle(state, 107)
    rebuilt = materialize_particle_state(state, hero_id=0, particle=particle, rng=np.random.default_rng(211))

    reordered_source = build_constructed_spot(pkrs.Stage.Turn)
    reordered_source.deck = list(reversed(reordered_source.deck))
    rebuilt_from_reordered = materialize_particle_state(
        reordered_source,
        hero_id=0,
        particle=particle,
        rng=np.random.default_rng(211),
    )

    actual_hands = tuple(
        tuple(card_key(card) for card in rebuilt.players_state[player].hand)
        for player in range(1, 6)
    )
    expected_hands = tuple(tuple(card_key(card) for card in hand) for hand in particle.opponent_hands)

    assert actual_hands == expected_hands
    assert card_key(rebuilt.deck[0]) == card_key(particle.runout[0])
    assert [card_key(card) for card in rebuilt.deck] == [card_key(card) for card in rebuilt_from_reordered.deck]


def test_lockstep_rollouts_complete_for_each_root_action_and_are_reproducible():
    state = build_constructed_spot(pkrs.Stage.Turn)
    particles = (_particle(state, 107), _particle(state, 211))
    root_actions = (
        resolve_action(ActionSlot.CHECK, state).action,
        resolve_action(ActionSlot.ALL_IN, state).action,
    )
    policy = PassiveBatchPolicy()

    first = evaluate_root_actions(
        state,
        hero_id=0,
        root_actions=root_actions,
        particles=particles,
        continuation_policy=policy,
        rng=np.random.default_rng(313),
    )
    second = evaluate_root_actions(
        state,
        hero_id=0,
        root_actions=root_actions,
        particles=particles,
        continuation_policy=policy,
        rng=np.random.default_rng(313),
    )

    assert np.array_equal(first.successful_rollouts, np.array([2, 2]))
    assert first.terminal_completion_rate == 1.0
    assert not first.failure_messages
    assert np.all(np.isfinite(first.raw_ev_mean))
    assert np.array_equal(first.raw_ev_mean, second.raw_ev_mean)


def test_lockstep_rollouts_report_emergency_depth_failure():
    state = build_constructed_spot(pkrs.Stage.Turn)
    result = evaluate_root_actions(
        state,
        hero_id=0,
        root_actions=(resolve_action(ActionSlot.CHECK, state).action,),
        particles=(_particle(state, 107),),
        continuation_policy=PassiveBatchPolicy(),
        rng=np.random.default_rng(313),
        max_depth=0,
    )

    assert np.array_equal(result.successful_rollouts, np.array([0]))
    assert result.terminal_completion_rate == 0.0
    assert any("max_depth" in message for message in result.failure_messages)
