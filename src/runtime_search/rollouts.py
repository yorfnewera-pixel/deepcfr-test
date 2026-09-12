"""Lockstep rollout evaluation с общими belief particles и runouts."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Sequence

import numpy as np
import pokers as pkrs

from src.core.action_space import NUM_ACTIONS, legal_action_mask, resolve_action
from src.runtime_search.beliefs import BeliefParticle
from src.runtime_search.cards import remaining_deck


class BatchContinuationPolicy(Protocol):
    """Компактная blueprint policy, умеющая infer несколько состояний за forward pass."""

    def probabilities_batch(self, states: Sequence[pkrs.State]) -> np.ndarray:
        """Возвращает одну distribution по fixed action slots на каждое состояние."""
        ...


@dataclass(frozen=True)
class RolloutEvaluation:
    """Агрегированная оценка root actions на одном наборе particles."""

    raw_ev_mean: np.ndarray
    raw_ev_std: np.ndarray
    raw_ev_se: np.ndarray
    best_gap: float
    best_gap_se: float
    best_gap_zscore: float
    successful_rollouts: np.ndarray
    total_particles: int
    failure_messages: tuple[str, ...]
    terminal_completion_rate: float


@dataclass
class _Scenario:
    state: pkrs.State
    action_index: int
    particle_index: int
    particle_seed: int


def materialize_particle_state(
    state: pkrs.State,
    *,
    hero_id: int,
    particle: BeliefParticle,
    rng: np.random.Generator,
) -> pkrs.State:
    """Пересобирает state с particle hands и independent ordered runout."""
    players = tuple(state.players_state)
    if hero_id < 0 or hero_id >= len(players):
        raise ValueError("hero_id вне диапазона игроков")
    if len(particle.opponent_hands) != len(players) - 1:
        raise ValueError("Число opponent hands не соответствует числу игроков")

    initial_stacks = np.asarray([
        float(player.stake + player.bet_chips + player.pot_chips)
        for player in players
    ])
    if not np.allclose(initial_stacks, initial_stacks[0]):
        raise ValueError("Rollout не поддерживает unequal initial stacks")

    opponent_hands = iter(particle.opponent_hands)
    hole_cards = []
    for player_index, player in enumerate(players):
        if player_index == hero_id:
            hole_cards.append(tuple(player.hand))
        else:
            hole_cards.append(next(opponent_hands))

    known_cards = [
        *hole_cards[hero_id],
        *(card for hand in particle.opponent_hands for card in hand),
        *state.public_cards,
        *particle.runout,
    ]
    remaining_cards = list(remaining_deck(known_cards))
    shuffled_indices = rng.permutation(len(remaining_cards))
    shuffled_remaining = [remaining_cards[int(index)] for index in shuffled_indices]
    rebuilt = pkrs.State.from_mid_hand(
        n_players=len(players),
        button=int(state.button),
        sb=float(state.sb),
        bb=float(state.bb),
        stake=float(initial_stacks[0]),
        deck=[*particle.runout, *shuffled_remaining],
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


def evaluate_root_actions(
    state: pkrs.State,
    *,
    hero_id: int,
    root_actions: Sequence[pkrs.Action],
    particles: Sequence[BeliefParticle],
    continuation_policy: BatchContinuationPolicy,
    rng: np.random.Generator,
    max_depth: int = 128,
) -> RolloutEvaluation:
    """Оценивает все root actions на общих hidden histories и ordered runouts."""
    if not root_actions:
        raise ValueError("Нужен хотя бы один root action")
    if not particles:
        raise ValueError("Нужен хотя бы один belief particle")
    if max_depth < 0:
        raise ValueError("max_depth не может быть отрицательным")

    particle_seeds = rng.integers(0, 2**31 - 1, size=len(particles), dtype=np.int64)
    scenarios: list[_Scenario] = []
    root_states: list[pkrs.State] = []
    root_batch_actions: list[pkrs.Action] = []
    for particle_index, particle in enumerate(particles):
        particle_seed = int(particle_seeds[particle_index])
        for action_index, root_action in enumerate(root_actions):
            root_states.append(materialize_particle_state(
                state,
                hero_id=hero_id,
                particle=particle,
                rng=np.random.default_rng(particle_seed),
            ))
            root_batch_actions.append(root_action)
            scenarios.append(_Scenario(
                state=root_states[-1],
                action_index=action_index,
                particle_index=particle_index,
                particle_seed=particle_seed,
            ))

    failures: list[str] = []
    rewards = np.full((len(root_actions), len(particles)), np.nan, dtype=np.float64)
    success = np.zeros((len(root_actions), len(particles)), dtype=bool)
    evolved_root_states = pkrs.parallel_apply_action(root_states, root_batch_actions)
    active_scenarios: list[_Scenario] = []
    for scenario, evolved in zip(scenarios, evolved_root_states):
        if evolved.status != pkrs.StateStatus.Ok:
            failures.append(_failure_message(scenario, "root action rejected by engine"))
        elif evolved.final_state:
            rewards[scenario.action_index, scenario.particle_index] = float(
                evolved.players_state[hero_id].reward
            )
            success[scenario.action_index, scenario.particle_index] = True
        else:
            scenario.state = evolved
            active_scenarios.append(scenario)

    for depth in range(max_depth):
        if not active_scenarios:
            break
        states = [scenario.state for scenario in active_scenarios]
        probabilities = np.asarray(continuation_policy.probabilities_batch(states), dtype=np.float64)
        if probabilities.shape != (len(states), NUM_ACTIONS):
            raise ValueError("continuation_policy вернула distribution неверной формы")

        next_scenarios: list[_Scenario] = []
        batch_states: list[pkrs.State] = []
        batch_actions: list[pkrs.Action] = []
        for scenario, action_probabilities in zip(active_scenarios, probabilities):
            try:
                action = _sample_continuation_action(scenario, action_probabilities, depth)
            except ValueError as error:
                failures.append(_failure_message(scenario, str(error)))
                continue
            batch_states.append(scenario.state)
            batch_actions.append(action)
            next_scenarios.append(scenario)

        evolved_states = pkrs.parallel_apply_action(batch_states, batch_actions)
        active_scenarios = []
        for scenario, evolved in zip(next_scenarios, evolved_states):
            if evolved.status != pkrs.StateStatus.Ok:
                failures.append(_failure_message(scenario, "continuation action rejected by engine"))
            elif evolved.final_state:
                rewards[scenario.action_index, scenario.particle_index] = float(
                    evolved.players_state[hero_id].reward
                )
                success[scenario.action_index, scenario.particle_index] = True
            else:
                scenario.state = evolved
                active_scenarios.append(scenario)

    for scenario in active_scenarios:
        failures.append(_failure_message(scenario, f"max_depth={max_depth} reached before terminal state"))

    successful_rollouts = success.sum(axis=1, dtype=np.int64)
    raw_ev_mean = np.asarray([
        float(np.mean(rewards[action_index, success[action_index]]))
        if successful_rollouts[action_index] else np.nan
        for action_index in range(len(root_actions))
    ], dtype=np.float64)
    raw_ev_std = np.asarray([
        float(np.std(rewards[action_index, success[action_index]], ddof=1))
        if successful_rollouts[action_index] > 1
        else 0.0 if successful_rollouts[action_index] else np.nan
        for action_index in range(len(root_actions))
    ], dtype=np.float64)
    raw_ev_se = np.asarray([
        value / np.sqrt(count) if count > 0 else np.nan
        for value, count in zip(raw_ev_std, successful_rollouts)
    ], dtype=np.float64)
    best_gap, best_gap_se, best_gap_zscore = _root_signal_statistics(rewards, success)
    total_scenarios = len(root_actions) * len(particles)
    return RolloutEvaluation(
        raw_ev_mean=raw_ev_mean,
        raw_ev_std=raw_ev_std,
        raw_ev_se=raw_ev_se,
        best_gap=best_gap,
        best_gap_se=best_gap_se,
        best_gap_zscore=best_gap_zscore,
        successful_rollouts=successful_rollouts,
        total_particles=len(particles),
        failure_messages=tuple(failures),
        terminal_completion_rate=float(successful_rollouts.sum() / total_scenarios),
    )


def _root_signal_statistics(
    rewards: np.ndarray,
    success: np.ndarray,
) -> tuple[float, float, float]:
    """Возвращает CRN-correct gap лучшего и второго root action."""
    rewards = np.asarray(rewards, dtype=np.float64)
    success = np.asarray(success, dtype=bool)
    if rewards.ndim != 2 or rewards.shape != success.shape:
        raise ValueError("rewards и success должны быть двумерными массивами одинаковой формы")
    if rewards.shape[0] < 2 or rewards.shape[1] < 2:
        return np.nan, np.nan, np.nan
    split_index = rewards.shape[1] // 2
    selection_mask = np.all(success[:, :split_index], axis=0)
    if not np.any(selection_mask):
        return np.nan, np.nan, np.nan
    means = rewards[:, :split_index][:, selection_mask].mean(axis=1)
    best_index = int(np.argmax(means))
    runner_values = means.copy()
    runner_values[best_index] = -np.inf
    runner_index = int(np.argmax(runner_values))
    paired_mask = success[best_index, split_index:] & success[runner_index, split_index:]
    differences = (
        rewards[best_index, split_index:][paired_mask]
        - rewards[runner_index, split_index:][paired_mask]
    )
    if differences.size == 0:
        return np.nan, np.nan, np.nan
    gap = float(differences.mean())
    if differences.size <= 1:
        return gap, np.nan, np.nan
    gap_se = float(differences.std(ddof=1) / np.sqrt(differences.size))
    if gap_se == 0.0:
        return gap, 0.0, float(np.inf) if gap > 0.0 else float(-np.inf) if gap < 0.0 else 0.0
    return gap, gap_se, gap / gap_se


def _sample_continuation_action(
    scenario: _Scenario,
    probabilities: np.ndarray,
    depth: int,
) -> pkrs.Action:
    if not np.all(np.isfinite(probabilities)) or np.any(probabilities < 0.0):
        raise ValueError("continuation policy вернула некорректные вероятности")
    mask = legal_action_mask(scenario.state)
    legal_probabilities = probabilities * mask
    total_probability = float(legal_probabilities.sum())
    if total_probability <= 0.0:
        raise ValueError("continuation policy не дала массу допустимым действиям")
    legal_probabilities /= total_probability
    decision_rng = np.random.default_rng(
        np.random.SeedSequence([scenario.particle_seed, depth, int(scenario.state.current_player)])
    )
    action_slot = int(decision_rng.choice(NUM_ACTIONS, p=legal_probabilities))
    return resolve_action(action_slot, scenario.state).action


def _failure_message(scenario: _Scenario, reason: str) -> str:
    return f"particle={scenario.particle_index}, action={scenario.action_index}: {reason}"
