"""Blocker-aware particle sampling до reach-weighting этапа."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Sequence

import numpy as np
import pokers as pkrs

from src.core.action_space import NUM_ACTIONS, legal_action_mask
from src.runtime_search.cards import card_key, remaining_deck


@dataclass(frozen=True)
class BeliefParticle:
    """Одна скрытая конфигурация для общего CRN rollout particle."""

    opponent_hands: tuple[tuple[pkrs.Card, pkrs.Card], ...]
    runout: tuple[pkrs.Card, ...]
    is_solver_valid: bool = False


@dataclass(frozen=True)
class ObservedDecision:
    """Наблюдаемое compact решение до root, без чтения hidden hands при replay."""

    state: pkrs.State
    actor_id: int
    action_slot: int


@dataclass(frozen=True)
class ReachWeightedBeliefs:
    """Resampled particles и качество reach-weighting до rollout-а."""

    particles: tuple[BeliefParticle, ...]
    proposal_count: int
    ess: float
    ess_ratio: float
    resample_count: int
    diagnostic_flags: tuple[str, ...]


class ReachPolicy(Protocol):
    """Минимальный frozen blueprint contract для likelihood истории."""

    def probabilities(self, state: pkrs.State, player_id: int | None = None) -> np.ndarray:
        """Возвращает compact policy для заданного игрока."""
        ...


def sample_reach_weighted_particles(
    *,
    hero_id: int,
    hero_hand: Sequence[pkrs.Card],
    public_cards: Sequence[pkrs.Card],
    num_opponents: int,
    runout_cards: int,
    proposal_count: int | None = None,
    particle_count: int,
    observed_decisions: Sequence[ObservedDecision],
    blueprint: ReachPolicy,
    rng: np.random.Generator,
    reach_probability_floor: float = 1e-12,
    resample_ess_ratio: float = 0.5,
) -> ReachWeightedBeliefs:
    """Сэмплирует posterior particles с последовательным resampling между решениями."""
    if particle_count <= 0:
        raise ValueError("particle_count должен быть положительным")
    proposal_count = particle_count if proposal_count is None else int(proposal_count)
    if proposal_count < particle_count:
        raise ValueError("proposal_count не может быть меньше particle_count")
    if not 0.0 < reach_probability_floor <= 1.0:
        raise ValueError("reach_probability_floor должен быть в интервале (0, 1]")
    if not 0.0 < resample_ess_ratio <= 1.0:
        raise ValueError("resample_ess_ratio должен быть в интервале (0, 1]")
    _validate_observed_decisions(observed_decisions, hero_id, hero_hand)

    proposals = tuple(
        sample_blocker_aware_particle(
            hero_hand=hero_hand,
            public_cards=public_cards,
            num_opponents=num_opponents,
            runout_cards=runout_cards,
            rng=rng,
        )
        for _ in range(proposal_count)
    )
    opponent_decisions = tuple(
        decision for decision in observed_decisions if decision.actor_id != hero_id
    )
    weights = np.full(proposal_count, 1.0 / proposal_count, dtype=np.float64)
    resample_count = 0
    for decision_index, decision in enumerate(opponent_decisions):
        log_likelihoods = np.empty(proposal_count, dtype=np.float64)
        for particle_index, particle in enumerate(proposals):
            replay_state = _materialize_observed_state(
                decision.state,
                hero_id=hero_id,
                particle=particle,
                conditioned_public_cards=public_cards,
                rng=rng,
            )
            probability = _observed_action_probability(
                blueprint.probabilities(replay_state, player_id=decision.actor_id),
                decision,
                replay_state,
            )
            log_likelihoods[particle_index] = np.log(
                max(probability, reach_probability_floor)
            )

        weights = _normalize_log_weights(np.log(weights) + log_likelihoods)
        ess = 1.0 / float(np.dot(weights, weights))
        has_next_opponent_decision = decision_index + 1 < len(opponent_decisions)
        if has_next_opponent_decision and ess / proposal_count < resample_ess_ratio:
            resampled_indices = rng.choice(
                proposal_count,
                size=proposal_count,
                replace=True,
                p=weights,
            )
            proposals = tuple(proposals[int(index)] for index in resampled_indices)
            weights.fill(1.0 / proposal_count)
            resample_count += 1

    ess = 1.0 / float(np.dot(weights, weights))
    selected_indices = rng.choice(
        proposal_count,
        size=particle_count,
        replace=True,
        p=weights / float(weights.sum()),
    )
    particles = tuple(
        BeliefParticle(
            opponent_hands=proposals[int(index)].opponent_hands,
            runout=proposals[int(index)].runout,
            is_solver_valid=True,
        )
        for index in selected_indices
    )
    return ReachWeightedBeliefs(
        particles=particles,
        proposal_count=proposal_count,
        ess=ess,
        ess_ratio=ess / proposal_count,
        resample_count=resample_count,
        diagnostic_flags=(
            "reach_weighted_beliefs",
            *(("reach_sequential_resampling",) if resample_count else ()),
        ),
    )


def sample_blocker_aware_particle(
    *,
    hero_hand: Sequence[pkrs.Card],
    public_cards: Sequence[pkrs.Card],
    num_opponents: int,
    runout_cards: int,
    rng: np.random.Generator,
) -> BeliefParticle:
    """Сэмплирует uniform blocker-aware baseline, не используя state.deck."""
    if len(hero_hand) != 2:
        raise ValueError("Hero hand должна состоять из двух карт")
    if num_opponents < 0 or runout_cards < 0:
        raise ValueError("Число opponents и runout cards не может быть отрицательным")

    available_cards = remaining_deck((*hero_hand, *public_cards))
    required_cards = 2 * num_opponents + runout_cards
    if required_cards > len(available_cards):
        raise ValueError(
            f"Недостаточно неизвестных карт: нужно {required_cards}, доступно {len(available_cards)}"
        )

    selected_indices = rng.permutation(len(available_cards))[:required_cards]
    selected_cards = tuple(available_cards[int(index)] for index in selected_indices)
    hands_end = 2 * num_opponents
    opponent_hands = tuple(
        (selected_cards[index], selected_cards[index + 1])
        for index in range(0, hands_end, 2)
    )
    return BeliefParticle(opponent_hands=opponent_hands, runout=selected_cards[hands_end:])


def _validate_observed_decisions(
    observed_decisions: Sequence[ObservedDecision],
    hero_id: int,
    hero_hand: Sequence[pkrs.Card],
) -> None:
    expected_hero_hand = {card_key(card) for card in hero_hand}
    for decision in observed_decisions:
        if decision.actor_id < 0 or decision.actor_id >= len(decision.state.players_state):
            raise ValueError("actor_id истории вне диапазона игроков")
        if decision.actor_id != int(decision.state.current_player):
            raise ValueError("actor_id истории не совпадает с current_player")
        observed_hero_hand = {card_key(card) for card in decision.state.players_state[hero_id].hand}
        if observed_hero_hand != expected_hero_hand:
            raise ValueError("Hero hand истории не совпадает с root hero hand")
        if decision.actor_id == hero_id:
            continue
        mask = legal_action_mask(decision.state).astype(bool)
        if decision.action_slot < 0 or decision.action_slot >= NUM_ACTIONS or not mask[decision.action_slot]:
            raise ValueError("История содержит недопустимый compact action slot")


def _materialize_observed_state(
    state: pkrs.State,
    *,
    hero_id: int,
    particle: BeliefParticle,
    conditioned_public_cards: Sequence[pkrs.Card],
    rng: np.random.Generator,
) -> pkrs.State:
    if state.stage == pkrs.Stage.Showdown:
        raise ValueError("Runtime search cannot reconstruct unresolved Showdown state")

    players = tuple(state.players_state)
    if hero_id < 0 or hero_id >= len(players):
        raise ValueError("hero_id вне диапазона игроков")
    if len(particle.opponent_hands) != len(players) - 1:
        raise ValueError("Число opponent hands не соответствует числу игроков")

    conditioned_keys = {card_key(card) for card in conditioned_public_cards}
    if not {card_key(card) for card in state.public_cards}.issubset(conditioned_keys):
        raise ValueError("Public cards истории не являются подмножеством root public cards")

    initial_stacks = np.asarray([
        float(player.stake + player.bet_chips + player.pot_chips)
        for player in players
    ])
    if not np.allclose(initial_stacks, initial_stacks[0]):
        raise ValueError("Reach-weighted beliefs не поддерживают unequal initial stacks")

    opponent_hands = iter(particle.opponent_hands)
    hole_cards: list[tuple[pkrs.Card, pkrs.Card]] = []
    for player_index, player in enumerate(players):
        if player_index == hero_id:
            hero_cards = tuple(player.hand)
            if len(hero_cards) != 2:
                raise ValueError("Hero hand истории должна состоять из двух карт")
            hole_cards.append((hero_cards[0], hero_cards[1]))
        else:
            hole_cards.append(next(opponent_hands))

    known_cards = [
        *hole_cards[hero_id],
        *(card for hand in particle.opponent_hands for card in hand),
        *conditioned_public_cards,
    ]
    independent_deck = list(remaining_deck(known_cards))
    deck_indices = rng.permutation(len(independent_deck))
    rebuilt = pkrs.State.from_mid_hand(
        n_players=len(players),
        button=int(state.button),
        sb=float(state.sb),
        bb=float(state.bb),
        stake=float(initial_stacks[0]),
        deck=[independent_deck[int(index)] for index in deck_indices],
        hole_cards=hole_cards,
        public_cards=list(state.public_cards),
        stage=state.stage,
        pot=float(state.pot),
        bet_chips=[float(player.bet_chips) for player in players],
        pot_chips=[float(player.pot_chips) for player in players],
        active=[bool(player.active) for player in players],
        last_stage_action=[player.last_stage_action for player in players],
        current_player=int(state.current_player),
        last_raise_increment=float(state.last_raise_increment),
        verbose=bool(state.verbose),
    )
    rebuilt.from_action = state.from_action
    if state.action_history_complete:
        rebuilt.copy_public_history_from(state)
    return rebuilt


def _observed_action_probability(
    probabilities: np.ndarray,
    decision: ObservedDecision,
    replay_state: pkrs.State,
) -> float:
    values = np.asarray(probabilities, dtype=np.float64)
    mask = legal_action_mask(replay_state).astype(bool)
    if values.shape != (NUM_ACTIONS,) or not np.all(np.isfinite(values)) or np.any(values < 0.0):
        raise ValueError("Blueprint вернул некорректную policy для reach weighting")
    if np.any(values[~mask] != 0.0) or not np.isclose(float(values.sum()), 1.0, rtol=1e-6, atol=1e-6):
        raise ValueError("Blueprint policy не согласована с legal mask истории")
    return float(values[decision.action_slot])


def _normalize_log_weights(log_weights: np.ndarray) -> np.ndarray:
    maximum = float(np.max(log_weights))
    shifted = np.exp(log_weights - maximum)
    total = float(shifted.sum())
    if not np.isfinite(total) or total <= 0.0:
        raise RuntimeError("Не удалось нормализовать reach weights")
    return shifted / total


__all__ = [
    "BeliefParticle",
    "ObservedDecision",
    "ReachWeightedBeliefs",
    "sample_blocker_aware_particle",
    "sample_reach_weighted_particles",
]
