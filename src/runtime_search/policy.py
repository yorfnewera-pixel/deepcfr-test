"""Root policy improvement поверх compact blueprint и MMDS."""
from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from numbers import Real
from time import perf_counter
from typing import Protocol, Sequence

import numpy as np
import pokers as pkrs

from src.core.action_space import ACTION_LABELS, ActionSlot, NUM_ACTIONS, call_amount, legal_action_mask, remaining_after_call, resolve_action
from src.runtime_search.beliefs import ObservedDecision, sample_reach_weighted_particles
from src.runtime_search.mmds import mmds_update
from src.runtime_search.rollouts import BatchContinuationPolicy, evaluate_root_actions

_RESPONSE_SLOTS = (
    ActionSlot.FOLD,
    ActionSlot.CALL,
    ActionSlot.RAISE_POT,
    ActionSlot.ALL_IN,
)
_NO_BET_SLOTS = (
    ActionSlot.CHECK,
    ActionSlot.RAISE_HALF_POT,
    ActionSlot.RAISE_POT,
    ActionSlot.ALL_IN,
)


class BlueprintPolicy(BatchContinuationPolicy, Protocol):
    """Frozen blueprint contract для root и batch continuation inference."""

    def probabilities(self, state: pkrs.State, player_id: int | None = None) -> np.ndarray:
        """Возвращает fixed-slot policy для root state."""
        ...


@dataclass(frozen=True)
class RuntimeSearchConfig:
    """Минимальный набор параметров postflop identity root policy improvement."""

    belief_proposal_count: int = 256
    belief_particles: int = 32
    belief_min_ess: float = 8.0
    eta: float = 10.0
    alpha: float = 0.05
    min_root_gap_zscore: float = 1.0
    min_root_gap_samples: int = 8
    policy_floor: float = 0.001
    belief_min_ess_ratio: float = 0.25
    belief_resample_ess_ratio: float = 0.5
    reach_probability_floor: float = 1e-12
    max_depth: int = 128
    value_scale_epsilon: float = 1e-8

    def __post_init__(self) -> None:
        if self.belief_particles <= 0 or self.belief_proposal_count <= 0:
            raise ValueError("belief budgets должны быть положительными")
        if self.belief_proposal_count < self.belief_particles:
            raise ValueError("belief_proposal_count не может быть меньше belief_particles")
        if self.eta < 0.0 or self.alpha < 0.0 or self.policy_floor < 0.0:
            raise ValueError("eta, alpha и policy_floor не могут быть отрицательными")
        if (
            isinstance(self.min_root_gap_zscore, bool)
            or not isinstance(self.min_root_gap_zscore, Real)
            or not isfinite(self.min_root_gap_zscore)
            or self.min_root_gap_zscore < 0.0
        ):
            raise ValueError("min_root_gap_zscore должен быть конечным числом не меньше 0")
        if type(self.min_root_gap_samples) is not int or self.min_root_gap_samples < 2:
            raise ValueError("min_root_gap_samples должен быть целым числом не меньше 2")
        if self.belief_min_ess < 0.0 or not 0.0 <= self.belief_min_ess_ratio <= 1.0:
            raise ValueError("belief_min_ess и belief_min_ess_ratio имеют недопустимое значение")
        if not 0.0 < self.belief_resample_ess_ratio <= 1.0:
            raise ValueError("belief_resample_ess_ratio имеет недопустимое значение")
        if not 0.0 < self.reach_probability_floor <= 1.0:
            raise ValueError("reach_probability_floor должен быть в интервале (0, 1]")
        if self.max_depth < 0 or self.value_scale_epsilon <= 0.0:
            raise ValueError("max_depth и value_scale_epsilon имеют недопустимое значение")


@dataclass(frozen=True)
class SearchDecision:
    """Результат root policy improvement с достаточной диагностикой."""

    action: pkrs.Action
    action_label: str
    root_blueprint_policy: np.ndarray
    root_search_policy: np.ndarray
    raw_ev: np.ndarray
    mc_values: np.ndarray
    eta_times_values: np.ndarray
    belief_particles: int
    belief_ess: float | None
    belief_ess_ratio: float | None
    belief_resample_count: int | None
    latency_seconds: float
    diagnostic_flags: tuple[str, ...]
    raw_ev_std: np.ndarray | None = None
    raw_ev_se: np.ndarray | None = None
    best_gap: float | None = None
    best_gap_se: float | None = None
    best_gap_zscore: float | None = None
    value_scale: float | None = None
    eta_effective: float | None = None


def choose_action(
    state: pkrs.State,
    *,
    hero_id: int,
    blueprint: BlueprintPolicy,
    config: RuntimeSearchConfig,
    rng: np.random.Generator,
    observed_decisions: Sequence[ObservedDecision] | None = None,
) -> SearchDecision:
    """Выбирает postflop identity action с beliefs или blueprint fallback."""
    started_at = perf_counter()
    if hero_id != int(state.current_player):
        raise ValueError("Runtime root search требует hero_id равный current_player")
    runout_cards = _runout_cards_for_stage(state.stage)

    root_mask = _root_mask(state)
    root_prior = _normalized_response_prior(blueprint.probabilities(state, player_id=hero_id), root_mask)
    if not observed_decisions:
        return _blueprint_fallback_decision(
            state,
            root_prior=root_prior,
            belief_particles=0,
            belief_ess=None,
            belief_ess_ratio=None,
            diagnostic_flags=("belief_history_missing_blueprint_fallback",),
            rng=rng,
            started_at=started_at,
        )

    beliefs = sample_reach_weighted_particles(
        hero_id=hero_id,
        hero_hand=state.players_state[hero_id].hand,
        public_cards=state.public_cards,
        num_opponents=len(state.players_state) - 1,
        runout_cards=runout_cards,
        proposal_count=config.belief_proposal_count,
        particle_count=config.belief_particles,
        observed_decisions=observed_decisions,
        blueprint=blueprint,
        rng=rng,
        reach_probability_floor=config.reach_probability_floor,
        resample_ess_ratio=config.belief_resample_ess_ratio,
    )
    if beliefs.ess < config.belief_min_ess or beliefs.ess_ratio < config.belief_min_ess_ratio:
        ess_flags = [*beliefs.diagnostic_flags, "belief_low_ess_blueprint_fallback"]
        if beliefs.ess < config.belief_min_ess:
            ess_flags.append("belief_low_absolute_ess_blueprint_fallback")
        if beliefs.ess_ratio < config.belief_min_ess_ratio:
            ess_flags.append("belief_low_ess_ratio_blueprint_fallback")
        return _blueprint_fallback_decision(
            state,
            root_prior=root_prior,
            belief_particles=len(beliefs.particles),
            belief_ess=beliefs.ess,
            belief_ess_ratio=beliefs.ess_ratio,
            belief_resample_count=beliefs.resample_count,
            diagnostic_flags=tuple(ess_flags),
            rng=rng,
            started_at=started_at,
        )

    root_slots = np.flatnonzero(root_mask)
    root_actions = tuple(resolve_action(int(slot), state).action for slot in root_slots)
    evaluation = evaluate_root_actions(
        state,
        hero_id=hero_id,
        root_actions=root_actions,
        particles=beliefs.particles,
        continuation_policy=blueprint,
        rng=rng,
        max_depth=config.max_depth,
    )

    raw_ev = np.full(NUM_ACTIONS, np.nan, dtype=np.float64)
    raw_ev[root_slots] = evaluation.raw_ev_mean
    raw_ev_std = np.full(NUM_ACTIONS, np.nan, dtype=np.float64)
    raw_ev_std[root_slots] = evaluation.raw_ev_std
    raw_ev_se = np.full(NUM_ACTIONS, np.nan, dtype=np.float64)
    raw_ev_se[root_slots] = evaluation.raw_ev_se
    mc_values = np.full(NUM_ACTIONS, np.nan, dtype=np.float64)
    eta_times_values = np.full(NUM_ACTIONS, np.nan, dtype=np.float64)
    flags = list(beliefs.diagnostic_flags)
    value_scale: float | None = None
    if np.any(evaluation.successful_rollouts != config.belief_particles):
        flags.append("rollout_incomplete")
        search_policy = root_prior.copy()
    elif evaluation.best_gap_sample_count < config.min_root_gap_samples:
        flags.append("rollout_insufficient_gap_samples")
        search_policy = root_prior.copy()
    elif not np.isfinite(evaluation.best_gap):
        flags.append("rollout_gap_undefined")
        search_policy = root_prior.copy()
    elif not np.isfinite(evaluation.best_gap_se) or evaluation.best_gap_se < 0.0:
        flags.append("rollout_gap_se_undefined")
        search_policy = root_prior.copy()
    elif (
        evaluation.best_gap_se == 0.0
        and evaluation.best_gap > 0.0
        and evaluation.best_gap_zscore == np.inf
    ):
        flags.append("rollout_zero_variance_positive_signal")
        search_policy = _updated_root_policy(
            state,
            root_prior=root_prior,
            root_mask=root_mask,
            root_slots=root_slots,
            raw_ev_mean=evaluation.raw_ev_mean,
            config=config,
            mc_values=mc_values,
            eta_times_values=eta_times_values,
        )
        if search_policy is None:
            flags.append("rollout_nonfinite_values")
            search_policy = root_prior.copy()
        else:
            value_scale = max(
                float(state.pot),
                float(remaining_after_call(state)),
                config.value_scale_epsilon,
            )
    elif evaluation.best_gap_zscore == -np.inf:
        flags.extend(("rollout_signal_below_noise", "rollout_signal_below_noise_blueprint_fallback"))
        search_policy = root_prior.copy()
    elif (
        evaluation.best_gap_se == 0.0
        or np.isnan(evaluation.best_gap_zscore)
        or evaluation.best_gap_zscore == np.inf
    ):
        flags.append("rollout_zscore_undefined")
        search_policy = root_prior.copy()
    elif evaluation.best_gap_zscore < config.min_root_gap_zscore:
        flags.extend(("rollout_signal_below_noise", "rollout_signal_below_noise_blueprint_fallback"))
        search_policy = root_prior.copy()
    else:
        search_policy = _updated_root_policy(
            state,
            root_prior=root_prior,
            root_mask=root_mask,
            root_slots=root_slots,
            raw_ev_mean=evaluation.raw_ev_mean,
            config=config,
            mc_values=mc_values,
            eta_times_values=eta_times_values,
        )
        if search_policy is None:
            flags.append("rollout_nonfinite_values")
            search_policy = root_prior.copy()
        else:
            value_scale = max(
                float(state.pot),
                float(remaining_after_call(state)),
                config.value_scale_epsilon,
            )

    return _search_decision(
        state,
        root_prior=root_prior,
        search_policy=search_policy,
        raw_ev=raw_ev,
        mc_values=mc_values,
        eta_times_values=eta_times_values,
        belief_particles=len(beliefs.particles),
        belief_ess=beliefs.ess,
        belief_ess_ratio=beliefs.ess_ratio,
        belief_resample_count=beliefs.resample_count,
        diagnostic_flags=tuple(flags),
        rng=rng,
        started_at=started_at,
        raw_ev_std=raw_ev_std,
        raw_ev_se=raw_ev_se,
        best_gap=evaluation.best_gap,
        best_gap_se=evaluation.best_gap_se,
        best_gap_zscore=evaluation.best_gap_zscore,
        value_scale=value_scale,
        eta_effective=config.eta / (1.0 + config.alpha * config.eta),
    )


def _updated_root_policy(
    state: pkrs.State,
    *,
    root_prior: np.ndarray,
    root_mask: np.ndarray,
    root_slots: np.ndarray,
    raw_ev_mean: np.ndarray,
    config: RuntimeSearchConfig,
    mc_values: np.ndarray,
    eta_times_values: np.ndarray,
) -> np.ndarray | None:
    """Возвращает MMDS policy только для конечных rollout значений."""
    if not np.all(np.isfinite(raw_ev_mean)):
        return None
    value_scale = max(
        float(state.pot),
        float(remaining_after_call(state)),
        config.value_scale_epsilon,
    )
    if not np.isfinite(value_scale):
        return None
    mc_values[root_slots] = raw_ev_mean / value_scale
    eta_times_values[root_slots] = config.eta * mc_values[root_slots]
    values_for_update = np.zeros(NUM_ACTIONS, dtype=np.float64)
    values_for_update[root_slots] = mc_values[root_slots]
    if not np.all(np.isfinite(mc_values[root_slots])) or not np.all(np.isfinite(values_for_update)):
        return None
    return mmds_update(
        root_prior,
        values_for_update,
        root_mask,
        config.eta,
        config.alpha,
        floor=config.policy_floor,
    )


def _blueprint_fallback_decision(
    state: pkrs.State,
    *,
    root_prior: np.ndarray,
    belief_particles: int,
    belief_ess: float | None,
    belief_ess_ratio: float | None,
    belief_resample_count: int | None = None,
    diagnostic_flags: tuple[str, ...],
    rng: np.random.Generator,
    started_at: float,
) -> SearchDecision:
    unavailable_values = np.full(NUM_ACTIONS, np.nan, dtype=np.float64)
    return _search_decision(
        state,
        root_prior=root_prior,
        search_policy=root_prior.copy(),
        raw_ev=unavailable_values,
        mc_values=unavailable_values.copy(),
        eta_times_values=unavailable_values.copy(),
        belief_particles=belief_particles,
        belief_ess=belief_ess,
        belief_ess_ratio=belief_ess_ratio,
        belief_resample_count=belief_resample_count,
        diagnostic_flags=diagnostic_flags,
        rng=rng,
        started_at=started_at,
    )


def _search_decision(
    state: pkrs.State,
    *,
    root_prior: np.ndarray,
    search_policy: np.ndarray,
    raw_ev: np.ndarray,
    mc_values: np.ndarray,
    eta_times_values: np.ndarray,
    belief_particles: int,
    belief_ess: float | None,
    belief_ess_ratio: float | None,
    belief_resample_count: int | None = None,
    diagnostic_flags: tuple[str, ...],
    rng: np.random.Generator,
    started_at: float,
    raw_ev_std: np.ndarray | None = None,
    raw_ev_se: np.ndarray | None = None,
    best_gap: float | None = None,
    best_gap_se: float | None = None,
    best_gap_zscore: float | None = None,
    value_scale: float | None = None,
    eta_effective: float | None = None,
) -> SearchDecision:
    action_slot = int(rng.choice(NUM_ACTIONS, p=search_policy))
    action = resolve_action(action_slot, state).action
    return SearchDecision(
        action=action,
        action_label=ACTION_LABELS[action_slot],
        root_blueprint_policy=root_prior,
        root_search_policy=search_policy,
        raw_ev=raw_ev,
        mc_values=mc_values,
        eta_times_values=eta_times_values,
        belief_particles=belief_particles,
        belief_ess=belief_ess,
        belief_ess_ratio=belief_ess_ratio,
        belief_resample_count=belief_resample_count,
        latency_seconds=perf_counter() - started_at,
        diagnostic_flags=diagnostic_flags,
        raw_ev_std=raw_ev_std,
        raw_ev_se=raw_ev_se,
        best_gap=best_gap,
        best_gap_se=best_gap_se,
        best_gap_zscore=best_gap_zscore,
        value_scale=value_scale,
        eta_effective=eta_effective,
    )


def is_postflop_identity_root(state: pkrs.State) -> bool:
    """Проверяет, поддерживает ли state прямой compact runtime root."""
    if state.stage not in (pkrs.Stage.Flop, pkrs.Stage.Turn, pkrs.Stage.River):
        return False
    return bool(_root_mask(state).any())


def _root_mask(state: pkrs.State) -> np.ndarray:
    legal_mask = legal_action_mask(state).astype(bool)
    supported_slots = _RESPONSE_SLOTS if call_amount(state) > 0.0 else _NO_BET_SLOTS
    root_mask = np.zeros(NUM_ACTIONS, dtype=bool)
    for slot in supported_slots:
        root_mask[int(slot)] = legal_mask[int(slot)]
    if not root_mask.any():
        raise ValueError("В root state нет допустимых identity compact actions")
    return root_mask


def _runout_cards_for_stage(stage: pkrs.Stage) -> int:
    if stage == pkrs.Stage.Flop:
        return 2
    if stage == pkrs.Stage.Turn:
        return 1
    if stage == pkrs.Stage.River:
        return 0
    raise ValueError("Runtime search поддерживает только Flop, Turn и River root states")


def _normalized_response_prior(probabilities: np.ndarray, response_mask: np.ndarray) -> np.ndarray:
    prior = np.asarray(probabilities, dtype=np.float64)
    if prior.shape != (NUM_ACTIONS,) or not np.all(np.isfinite(prior)) or np.any(prior < 0.0):
        raise ValueError("Blueprint вернул некорректную root policy")
    masked_prior = np.where(response_mask, prior, 0.0)
    total = float(masked_prior.sum())
    if total <= 0.0:
        raise ValueError("Blueprint не дал массу identity response actions")
    return masked_prior / total
