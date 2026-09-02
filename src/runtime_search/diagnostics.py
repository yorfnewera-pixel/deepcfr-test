"""Воспроизводимая диагностика turn/river blueprint и runtime search."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from time import perf_counter
from typing import Protocol

import numpy as np
import pokers as pkrs

from src.core.action_space import ACTION_LABELS, NUM_ACTIONS, legal_action_mask, resolve_action
from src.runtime_search.policy import BlueprintPolicy, RuntimeSearchConfig, choose_action
from src.runtime_search.spots import build_constructed_spot, build_response_to_bet_spot

_NUM_PLAYERS = 6
_MAX_ACTIONS_PER_DEAL = 256


class RootBlueprintPolicy(BlueprintPolicy, Protocol):
    """Blueprint, достаточный для natural-trajectory диагностики."""


@dataclass(frozen=True)
class StreetDiagnostics:
    """Агрегированная статистика решений на одной улице или spot."""

    decisions: int
    mean_probability_mass: tuple[float, ...]
    chosen_action_frequency: tuple[float, ...]
    mean_entropy: float
    mean_latency_milliseconds: float


@dataclass(frozen=True)
class NaturalLateStreetDiagnostics:
    """Результат дешёвого прогона естественно достигнутых поздних улиц."""

    completed_deals: int
    failed_deals: int
    failure_messages: tuple[str, ...]
    streets: dict[str, StreetDiagnostics]


@dataclass(frozen=True)
class ConstructedSpotDiagnostics:
    """Один deterministic spot, не зависящий от частоты достижения улицы."""

    legal_action_labels: tuple[str, ...]
    decisions: int
    mean_probability_mass: tuple[float, ...]
    mean_entropy: float
    latency_milliseconds: float


@dataclass(frozen=True)
class SearchProbeDiagnostics:
    """Короткий S4 probe, который не интерпретирует EV как solver-quality proof."""

    action_label: str
    action_is_legal: bool
    belief_particles: int
    latency_milliseconds: float
    root_blueprint_policy: tuple[float, ...]
    root_search_policy: tuple[float, ...]
    raw_ev: tuple[float | None, ...]
    diagnostic_flags: tuple[str, ...]


@dataclass(frozen=True)
class TurnRiverDiagnostics:
    """Полный результат S5, пригодный для сохранения в JSON."""

    seed: int
    requested_deals: int
    natural: NaturalLateStreetDiagnostics
    constructed_spots: dict[str, ConstructedSpotDiagnostics]
    search_probe: SearchProbeDiagnostics

    def to_dict(self) -> dict:
        """Преобразует результат в JSON-совместимый словарь."""
        return asdict(self)


def collect_turn_river_diagnostics(
    blueprint: RootBlueprintPolicy,
    *,
    num_deals: int,
    belief_particles: int,
    seed: int,
) -> TurnRiverDiagnostics:
    """Собирает natural/constructed метрики и короткий response-to-bet probe."""
    if num_deals <= 0:
        raise ValueError("num_deals должен быть положительным")
    if belief_particles <= 0:
        raise ValueError("belief_particles должен быть положительным")

    natural = _collect_natural_late_street_diagnostics(blueprint, num_deals=num_deals, seed=seed)
    constructed = {
        "turn": _diagnose_constructed_spot(blueprint, build_constructed_spot(pkrs.Stage.Turn)),
        "river": _diagnose_constructed_spot(blueprint, build_constructed_spot(pkrs.Stage.River)),
        "response_to_bet_turn": _diagnose_constructed_spot(blueprint, build_response_to_bet_spot()),
    }
    search_probe = _run_search_probe(blueprint, belief_particles=belief_particles, seed=seed)
    return TurnRiverDiagnostics(
        seed=int(seed),
        requested_deals=int(num_deals),
        natural=natural,
        constructed_spots=constructed,
        search_probe=search_probe,
    )


def _collect_natural_late_street_diagnostics(
    blueprint: RootBlueprintPolicy,
    *,
    num_deals: int,
    seed: int,
) -> NaturalLateStreetDiagnostics:
    accumulators = {"turn": _StreetAccumulator(), "river": _StreetAccumulator()}
    completed_deals = 0
    failures: list[str] = []

    for deal_index in range(num_deals):
        deal_seed = _derived_seed(seed, deal_index)
        state = pkrs.State.from_seed(
            n_players=_NUM_PLAYERS,
            button=deal_seed % _NUM_PLAYERS,
            sb=1.0,
            bb=2.0,
            stake=200.0,
            seed=deal_seed,
        )
        try:
            for action_index in range(_MAX_ACTIONS_PER_DEAL):
                if state.final_state:
                    completed_deals += 1
                    break
                probabilities, latency_milliseconds = _timed_probabilities(blueprint, state)
                action_slot = _sample_slot(
                    probabilities,
                    seed=_derived_seed(seed, deal_index, action_index, int(state.current_player)),
                )
                street_name = _street_name(state.stage)
                if street_name is not None:
                    accumulators[street_name].add(probabilities, action_slot, latency_milliseconds)
                action = resolve_action(action_slot, state).action
                if action.action not in state.legal_actions:
                    raise RuntimeError("Blueprint выбрал недопустимое действие")
                state = state.apply_action(action)
                if state.status != pkrs.StateStatus.Ok:
                    raise RuntimeError(f"Движок отклонил действие: {state.status}")
            else:
                raise RuntimeError(f"Превышен лимит {_MAX_ACTIONS_PER_DEAL} действий")
        except (RuntimeError, TypeError, ValueError) as error:
            failures.append(f"deal={deal_index}: {error}")

    return NaturalLateStreetDiagnostics(
        completed_deals=completed_deals,
        failed_deals=len(failures),
        failure_messages=tuple(failures),
        streets={name: accumulator.freeze() for name, accumulator in accumulators.items()},
    )


def _diagnose_constructed_spot(
    blueprint: RootBlueprintPolicy,
    state: pkrs.State,
) -> ConstructedSpotDiagnostics:
    probabilities, latency_milliseconds = _timed_probabilities(blueprint, state)
    legal_mask = legal_action_mask(state).astype(bool)
    return ConstructedSpotDiagnostics(
        legal_action_labels=tuple(
            ACTION_LABELS[slot]
            for slot in np.flatnonzero(legal_mask)
        ),
        decisions=1,
        mean_probability_mass=tuple(float(value) for value in probabilities),
        mean_entropy=_entropy(probabilities),
        latency_milliseconds=latency_milliseconds,
    )


def _run_search_probe(
    blueprint: RootBlueprintPolicy,
    *,
    belief_particles: int,
    seed: int,
) -> SearchProbeDiagnostics:
    state = build_response_to_bet_spot()
    decision = choose_action(
        state,
        hero_id=int(state.current_player),
        blueprint=blueprint,
        config=RuntimeSearchConfig(belief_particles=belief_particles),
        rng=np.random.default_rng(_derived_seed(seed, 999)),
    )
    return SearchProbeDiagnostics(
        action_label=decision.action_label,
        action_is_legal=decision.action.action in state.legal_actions,
        belief_particles=decision.belief_particles,
        latency_milliseconds=decision.latency_seconds * 1_000.0,
        root_blueprint_policy=tuple(float(value) for value in decision.root_blueprint_policy),
        root_search_policy=tuple(float(value) for value in decision.root_search_policy),
        raw_ev=tuple(_finite_or_none(value) for value in decision.raw_ev),
        diagnostic_flags=decision.diagnostic_flags,
    )


@dataclass
class _StreetAccumulator:
    probability_sum: np.ndarray | None = None
    action_counts: np.ndarray | None = None
    entropy_sum: float = 0.0
    latency_sum_milliseconds: float = 0.0
    decisions: int = 0

    def add(self, probabilities: np.ndarray, action_slot: int, latency_milliseconds: float) -> None:
        if self.probability_sum is None:
            self.probability_sum = np.zeros(NUM_ACTIONS, dtype=np.float64)
            self.action_counts = np.zeros(NUM_ACTIONS, dtype=np.float64)
        probability_sum = self.probability_sum
        action_counts = self.action_counts
        assert probability_sum is not None
        assert action_counts is not None
        probability_sum += probabilities
        action_counts[action_slot] += 1.0
        self.entropy_sum += _entropy(probabilities)
        self.latency_sum_milliseconds += latency_milliseconds
        self.decisions += 1

    def freeze(self) -> StreetDiagnostics:
        if self.decisions == 0:
            zeros = tuple(0.0 for _ in range(NUM_ACTIONS))
            return StreetDiagnostics(0, zeros, zeros, 0.0, 0.0)
        assert self.probability_sum is not None
        assert self.action_counts is not None
        return StreetDiagnostics(
            decisions=self.decisions,
            mean_probability_mass=tuple(float(value / self.decisions) for value in self.probability_sum),
            chosen_action_frequency=tuple(float(value / self.decisions) for value in self.action_counts),
            mean_entropy=self.entropy_sum / self.decisions,
            mean_latency_milliseconds=self.latency_sum_milliseconds / self.decisions,
        )


def _timed_probabilities(blueprint: RootBlueprintPolicy, state: pkrs.State) -> tuple[np.ndarray, float]:
    started_at = perf_counter()
    probabilities = _validated_probabilities(blueprint.probabilities(state), state)
    return probabilities, (perf_counter() - started_at) * 1_000.0


def _validated_probabilities(probabilities: np.ndarray, state: pkrs.State) -> np.ndarray:
    values = np.asarray(probabilities, dtype=np.float64)
    mask = legal_action_mask(state).astype(bool)
    if values.shape != (NUM_ACTIONS,):
        raise ValueError("Blueprint вернул policy неверной формы")
    if not np.all(np.isfinite(values)) or np.any(values < 0.0):
        raise ValueError("Blueprint вернул неконечную или отрицательную policy")
    if np.any(values[~mask] != 0.0):
        raise ValueError("Blueprint дал массу недопустимому действию")
    total = float(values.sum())
    if not np.isclose(total, 1.0, rtol=1e-6, atol=1e-6):
        raise ValueError("Blueprint policy должна суммироваться в единицу")
    return values


def _sample_slot(probabilities: np.ndarray, *, seed: int) -> int:
    normalized = probabilities / float(probabilities.sum())
    return int(np.random.default_rng(seed).choice(NUM_ACTIONS, p=normalized))


def _street_name(stage: pkrs.Stage) -> str | None:
    if stage == pkrs.Stage.Turn:
        return "turn"
    if stage == pkrs.Stage.River:
        return "river"
    return None


def _entropy(probabilities: np.ndarray) -> float:
    positive = probabilities[probabilities > 0.0]
    return float(sum(-float(value) * float(np.log(value)) for value in positive))


def _derived_seed(*values: int) -> int:
    return int(np.random.SeedSequence(values).generate_state(1, dtype=np.uint32)[0])


def _finite_or_none(value: float) -> float | None:
    numeric = float(value)
    return numeric if np.isfinite(numeric) else None


__all__ = [
    "ConstructedSpotDiagnostics",
    "NaturalLateStreetDiagnostics",
    "SearchProbeDiagnostics",
    "StreetDiagnostics",
    "TurnRiverDiagnostics",
    "collect_turn_river_diagnostics",
]
