import numpy as np
import pokers as pkrs
import pytest

from src.runtime_search.beliefs import (
    ObservedDecision,
    _materialize_observed_state,
    sample_blocker_aware_particle,
    sample_reach_weighted_particles,
)
from src.core.action_space import ActionSlot, NUM_ACTIONS, legal_action_mask, resolve_action
from src.runtime_search.cards import card_key
from src.runtime_search.spots import build_constructed_spot


def _particle_keys(particle):
    return (
        tuple(tuple(card_key(card) for card in hand) for hand in particle.opponent_hands),
        tuple(card_key(card) for card in particle.runout),
    )


class HandSensitiveBlueprint:
    """Blueprint, у которого check likelihood зависит от hidden hand actor-а."""

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

    def probabilities_batch(self, states):
        return np.stack([self.probabilities(state) for state in states])


def test_blocker_aware_particle_has_no_known_or_duplicate_cards():
    state = build_constructed_spot(pkrs.Stage.Turn)
    particle = sample_blocker_aware_particle(
        hero_hand=state.players_state[0].hand,
        public_cards=state.public_cards,
        num_opponents=5,
        runout_cards=1,
        rng=np.random.default_rng(107),
    )
    known = {card_key(card) for card in (*state.players_state[0].hand, *state.public_cards)}
    sampled = [
        *(card for hand in particle.opponent_hands for card in hand),
        *particle.runout,
    ]

    assert len(particle.opponent_hands) == 5
    assert all(len(hand) == 2 for hand in particle.opponent_hands)
    assert len(particle.runout) == 1
    assert not particle.is_solver_valid
    assert not known.intersection(card_key(card) for card in sampled)
    assert len({card_key(card) for card in sampled}) == len(sampled)


def test_reach_weighted_materialization_rejects_showdown_before_reconstruction():
    state = build_constructed_spot(pkrs.Stage.River)
    state.stage = pkrs.Stage.Showdown
    particle = sample_blocker_aware_particle(
        hero_hand=state.players_state[0].hand,
        public_cards=state.public_cards,
        num_opponents=5,
        runout_cards=0,
        rng=np.random.default_rng(107),
    )

    with pytest.raises(ValueError, match="Showdown"):
        _materialize_observed_state(
            state,
            hero_id=0,
            particle=particle,
            conditioned_public_cards=state.public_cards,
            rng=np.random.default_rng(211),
        )


def test_blocker_aware_particle_is_reproducible_and_does_not_depend_on_state_deck():
    state = build_constructed_spot(pkrs.Stage.Turn)
    inputs = dict(
        hero_hand=state.players_state[0].hand,
        public_cards=state.public_cards,
        num_opponents=5,
        runout_cards=1,
    )

    first = sample_blocker_aware_particle(**inputs, rng=np.random.default_rng(107))
    deck_mutated = list(state.deck)
    deck_mutated.reverse()
    second = sample_blocker_aware_particle(**inputs, rng=np.random.default_rng(107))

    assert deck_mutated != list(state.deck)
    assert _particle_keys(first) == _particle_keys(second)


def test_blocker_aware_particle_reports_impossible_sample():
    state = build_constructed_spot(pkrs.Stage.River)

    with pytest.raises(ValueError, match="Недостаточно неизвестных карт"):
        sample_blocker_aware_particle(
            hero_hand=state.players_state[0].hand,
            public_cards=state.public_cards,
            num_opponents=24,
            runout_cards=0,
            rng=np.random.default_rng(107),
        )


def test_reach_weighted_particles_are_solver_valid_and_reproducible():
    state = build_constructed_spot(pkrs.Stage.Turn)
    history = (
        ObservedDecision(state=state, actor_id=0, action_slot=ActionSlot.CHECK),
    )
    inputs = dict(
        hero_id=1,
        hero_hand=state.players_state[1].hand,
        public_cards=state.public_cards,
        num_opponents=5,
        runout_cards=1,
        proposal_count=16,
        particle_count=4,
        observed_decisions=history,
        blueprint=HandSensitiveBlueprint(),
    )

    first = sample_reach_weighted_particles(**inputs, rng=np.random.default_rng(107))
    second = sample_reach_weighted_particles(**inputs, rng=np.random.default_rng(107))

    assert first.proposal_count == 16
    assert len(first.particles) == 4
    assert first.ess_ratio == pytest.approx(first.ess / 16)
    assert first.diagnostic_flags == ("reach_weighted_beliefs",)
    assert all(particle.is_solver_valid for particle in first.particles)
    assert tuple(_particle_keys(particle) for particle in first.particles) == tuple(
        _particle_keys(particle) for particle in second.particles
    )


def test_reach_weighted_particles_resample_between_observed_actions():
    first_state = build_constructed_spot(pkrs.Stage.Turn)
    second_state = first_state.apply_action(
        resolve_action(ActionSlot.CHECK, first_state).action
    )
    history = (
        ObservedDecision(state=first_state, actor_id=0, action_slot=ActionSlot.CHECK),
        ObservedDecision(state=second_state, actor_id=1, action_slot=ActionSlot.CHECK),
    )
    inputs = dict(
        hero_id=2,
        hero_hand=first_state.players_state[2].hand,
        public_cards=first_state.public_cards,
        num_opponents=5,
        runout_cards=1,
        proposal_count=16,
        particle_count=4,
        observed_decisions=history,
        blueprint=HandSensitiveBlueprint(),
        resample_ess_ratio=0.9,
    )

    first = sample_reach_weighted_particles(**inputs, rng=np.random.default_rng(107))
    second = sample_reach_weighted_particles(**inputs, rng=np.random.default_rng(107))

    assert first.resample_count == 1
    assert "reach_sequential_resampling" in first.diagnostic_flags
    assert first.ess == pytest.approx(16.0)
    assert tuple(_particle_keys(particle) for particle in first.particles) == tuple(
        _particle_keys(particle) for particle in second.particles
    )


def test_reach_weighted_particles_reject_illegal_history_slot():
    state = build_constructed_spot(pkrs.Stage.Turn)
    history = (
        ObservedDecision(state=state, actor_id=0, action_slot=ActionSlot.CALL),
    )

    with pytest.raises(ValueError, match="недопустимый"):
        sample_reach_weighted_particles(
            hero_id=1,
            hero_hand=state.players_state[1].hand,
            public_cards=state.public_cards,
            num_opponents=5,
            runout_cards=1,
            particle_count=4,
            observed_decisions=history,
            blueprint=HandSensitiveBlueprint(),
            rng=np.random.default_rng(107),
        )


def test_reach_weighted_particles_reject_history_for_another_hero_hand():
    state = build_constructed_spot(pkrs.Stage.Turn)
    history = (
        ObservedDecision(state=state, actor_id=0, action_slot=ActionSlot.CHECK),
    )

    with pytest.raises(ValueError, match="Hero hand истории не совпадает"):
        sample_reach_weighted_particles(
            hero_id=1,
            hero_hand=state.players_state[0].hand,
            public_cards=state.public_cards,
            num_opponents=5,
            runout_cards=1,
            particle_count=4,
            observed_decisions=history,
            blueprint=HandSensitiveBlueprint(),
            rng=np.random.default_rng(107),
        )
