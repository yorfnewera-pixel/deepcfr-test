"""Deterministic paired evaluation for multiway poker policies.

S2 harness: duplicate deals, common random numbers, seat rotation, and paired
hero-reward differences.  It is intentionally independent of Deep CFR so its
identity baseline can prove that the evaluation pipeline itself is sound.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Sequence

import numpy as np
import pokers as pkrs


class Policy(Protocol):
    def choose_action(self, state: pkrs.State, rng: np.random.Generator) -> pkrs.Action:
        """Choose an engine-legal action using only the supplied RNG."""
        ...


@dataclass(frozen=True)
class PairedEvaluation:
    differences: np.ndarray
    baseline_rewards: np.ndarray
    candidate_rewards: np.ndarray
    deals: int
    seats: int
    bb: float

    def __post_init__(self) -> None:
        if not np.isfinite(self.bb) or self.bb <= 0.0:
            raise ValueError("bb должен быть положительным конечным числом")

    @property
    def samples(self) -> int:
        return int(self.differences.size)

    @property
    def mean_difference(self) -> float:
        return float(np.mean(self.differences)) if self.samples else 0.0

    @property
    def std_difference(self) -> float:
        return float(np.std(self.differences, ddof=1)) if self.samples > 1 else 0.0

    @property
    def bb_per_100(self) -> float:
        return self.mean_difference / self.bb * 100.0


def _play_deal(
    *,
    deal_seed: int,
    hero_seat: int,
    hero_policy: Policy,
    opponent_policy: Policy,
    num_players: int,
    bb: float,
) -> float:
    """Play one fixed deal. Every action receives a fresh CRN-derived RNG."""
    state = pkrs.State.from_seed(
        n_players=num_players,
        button=deal_seed % num_players,
        sb=1.0,
        bb=bb,
        stake=200.0,
        seed=deal_seed,
    )
    _prepare_hand((hero_policy, opponent_policy), hero_seat)
    action_index = 0
    while not state.final_state:
        player = int(state.current_player)
        # Identical per-decision random stream across duplicate arms. It is
        # intentionally independent of the arm and hero seat.
        rng = np.random.default_rng(np.random.SeedSequence([deal_seed, action_index, player]))
        policy = hero_policy if player == hero_seat else opponent_policy
        action = policy.choose_action(state, rng)
        if action.action not in state.legal_actions:
            raise ValueError(f"policy returned illegal action {action.action} for player {player}")
        _observe_action((hero_policy, opponent_policy), state, action)
        # `Raise` also has a size constraint that is not represented by the
        # enum-only action mask. The policy must produce a fully legal action.
        state = state.apply_action(action)
        if state.status != pkrs.StateStatus.Ok:
            raise RuntimeError(f"engine rejected legal action: {state.status}")
        action_index += 1
    return float(state.players_state[hero_seat].reward)


def _prepare_hand(policies: tuple[Policy, Policy], hero_seat: int) -> None:
    for policy in _unique_policies(policies):
        set_hero_id = getattr(policy, "set_hero_id", None)
        if callable(set_hero_id):
            set_hero_id(hero_seat)
        reset_hand = getattr(policy, "reset_hand", None)
        if callable(reset_hand):
            reset_hand()


def _observe_action(policies: tuple[Policy, Policy], state: pkrs.State, action: pkrs.Action) -> None:
    for policy in _unique_policies(policies):
        observe_action = getattr(policy, "observe_action", None)
        if callable(observe_action):
            observe_action(state, action)


def _unique_policies(policies: tuple[Policy, Policy]) -> tuple[Policy, ...]:
    unique: list[Policy] = []
    seen_ids: set[int] = set()
    for policy in policies:
        if id(policy) not in seen_ids:
            unique.append(policy)
            seen_ids.add(id(policy))
    return tuple(unique)


def evaluate_paired(
    baseline_policy: Policy,
    candidate_policy: Policy,
    *,
    num_deals: int,
    seed: int,
    num_players: int = 6,
    rotate_seats: bool = True,
    bb: float = 2.0,
) -> PairedEvaluation:
    """Evaluate candidate minus baseline with duplicate deals and CRN.

    Each deal is played twice for every hero seat: once with baseline hero and
    once with candidate hero. All non-hero seats use the same opponent policy.
    With identical policies this must produce an exactly zero vector.
    """
    if num_deals <= 0:
        raise ValueError("num_deals must be positive")
    if num_players < 2:
        raise ValueError("num_players must be at least two")
    if not np.isfinite(bb) or bb <= 0.0:
        raise ValueError("bb must be a positive finite number")

    root_rng = np.random.default_rng(seed)
    seats: Sequence[int] = tuple(range(num_players)) if rotate_seats else (0,)
    baseline_rewards: list[float] = []
    candidate_rewards: list[float] = []

    for deal_index in range(num_deals):
        deal_seed = int(root_rng.integers(0, 2**31 - 1))
        for hero_seat in seats:
            baseline_rewards.append(_play_deal(
                deal_seed=deal_seed,
                hero_seat=hero_seat,
                hero_policy=baseline_policy,
                opponent_policy=baseline_policy,
                num_players=num_players,
                bb=bb,
            ))
            candidate_rewards.append(_play_deal(
                deal_seed=deal_seed,
                hero_seat=hero_seat,
                hero_policy=candidate_policy,
                opponent_policy=baseline_policy,
                num_players=num_players,
                bb=bb,
            ))

    baseline = np.asarray(baseline_rewards, dtype=np.float64)
    candidate = np.asarray(candidate_rewards, dtype=np.float64)
    return PairedEvaluation(
        differences=candidate - baseline,
        baseline_rewards=baseline,
        candidate_rewards=candidate,
        deals=num_deals,
        seats=len(seats),
        bb=float(bb),
    )


class DeterministicLegalPolicy:
    """Minimal policy used by S2 to prove duplicate/CRN determinism."""

    def choose_action(self, state: pkrs.State, rng: np.random.Generator) -> pkrs.Action:
        del rng
        for action in (pkrs.ActionEnum.Check, pkrs.ActionEnum.Call, pkrs.ActionEnum.Fold):
            if action is not None and action in state.legal_actions:
                resolved_action = pkrs.Action(action)
                if resolved_action is not None:
                    return resolved_action
        raise RuntimeError("state has no passive legal action")
