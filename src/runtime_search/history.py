"""Автоматическая запись observed history для S6 runtime policy."""
from __future__ import annotations

from copy import deepcopy
from typing import Protocol

import numpy as np
import pokers as pkrs

from src.core.action_space import ActionSlot, resolve_action
from src.runtime_search.beliefs import ObservedDecision
from src.runtime_search.policy import (
    BlueprintPolicy,
    RuntimeSearchConfig,
    SearchDecision,
    choose_action,
    is_postflop_identity_root,
)


class BlueprintActionPolicy(BlueprintPolicy, Protocol):
    """Frozen blueprint, пригодный для fallback за пределами runtime root."""

    def choose_action(self, state: pkrs.State, rng: np.random.Generator) -> pkrs.Action:
        """Выбирает compact blueprint action."""
        ...


class ActionHistoryRecorder:
    """Хранит compact решения оппонентов или явно отмечает неполную историю."""

    def __init__(self, hero_id: int):
        self._hero_id = int(hero_id)
        self._observed_decisions: list[ObservedDecision] = []
        self._diagnostic_flags: list[str] = []

    @property
    def observed_decisions(self) -> tuple[ObservedDecision, ...]:
        return tuple(self._observed_decisions)

    @property
    def diagnostic_flags(self) -> tuple[str, ...]:
        return tuple(self._diagnostic_flags)

    @property
    def is_complete(self) -> bool:
        return not self._diagnostic_flags

    def set_hero_id(self, hero_id: int) -> None:
        self._hero_id = int(hero_id)
        self.reset_hand()

    def reset_hand(self) -> None:
        self._observed_decisions.clear()
        self._diagnostic_flags.clear()

    def observe_action(self, state: pkrs.State, action: pkrs.Action) -> None:
        """Записывает opponent action до перехода к следующему engine state."""
        actor_id = int(state.current_player)
        if actor_id == self._hero_id:
            return
        action_slot = _compact_slot_for_action(state, action)
        if action_slot is None:
            self._add_flag("history_contains_noncompact_opponent_action")
            return
        self._observed_decisions.append(
            ObservedDecision(state=state, actor_id=actor_id, action_slot=action_slot)
        )

    def _add_flag(self, flag: str) -> None:
        if flag not in self._diagnostic_flags:
            self._diagnostic_flags.append(flag)


class RuntimeSearchPolicy:
    """Evaluation policy, которая сама получает history от paired harness hooks."""

    action_comparison_count: int
    action_change_count: int
    search_decisions_by_stage: dict[str, int]
    action_changes_by_stage: dict[str, int]
    policy_l1_sum: float
    policy_l1_max: float
    policy_argmax_change_count: int
    policy_l1_sum_by_stage: dict[str, float]
    policy_argmax_changes_by_stage: dict[str, int]

    def __init__(
        self,
        *,
        hero_id: int,
        blueprint: BlueprintActionPolicy,
        config: RuntimeSearchConfig,
    ) -> None:
        self._hero_id = int(hero_id)
        self._blueprint = blueprint
        self._config = config
        self._history = ActionHistoryRecorder(hero_id)
        self.last_search_decision: SearchDecision | None = None
        self.search_decisions: list[SearchDecision] = []
        self.search_decision_count = 0
        self.reach_weighted_decision_count = 0
        self.blueprint_fallback_decision_count = 0
        self.action_comparison_count = 0
        self.action_change_count = 0
        self.search_decisions_by_stage = {"flop": 0, "turn": 0, "river": 0}
        self.action_changes_by_stage = {"flop": 0, "turn": 0, "river": 0}
        self.policy_l1_sum = 0.0
        self.policy_l1_max = 0.0
        self.policy_argmax_change_count = 0
        self.policy_l1_sum_by_stage = {"flop": 0.0, "turn": 0.0, "river": 0.0}
        self.policy_argmax_changes_by_stage = {"flop": 0, "turn": 0, "river": 0}

    def set_hero_id(self, hero_id: int) -> None:
        self._hero_id = int(hero_id)
        self._history.set_hero_id(hero_id)
        self.last_search_decision = None

    def reset_hand(self) -> None:
        self._history.reset_hand()
        self.last_search_decision = None

    def observe_action(self, state: pkrs.State, action: pkrs.Action) -> None:
        self._history.observe_action(state, action)

    def choose_action(self, state: pkrs.State, rng: np.random.Generator) -> pkrs.Action:
        if int(state.current_player) != self._hero_id:
            raise ValueError("RuntimeSearchPolicy вызвана не для hero player")
        if not is_postflop_identity_root(state):
            return self._blueprint.choose_action(state, rng)

        blueprint_action = _sample_blueprint_action_without_advancing_rng(
            self._blueprint,
            state,
            rng,
        )
        observed_decisions = self._history.observed_decisions if self._history.is_complete else None
        decision = choose_action(
            state,
            hero_id=self._hero_id,
            blueprint=self._blueprint,
            config=self._config,
            rng=rng,
            observed_decisions=observed_decisions,
        )
        self.last_search_decision = decision
        self.search_decisions.append(decision)
        self.search_decision_count += 1
        self.action_comparison_count += 1
        stage_name = _postflop_stage_name(state)
        self.search_decisions_by_stage[stage_name] += 1
        policy_l1_distance = float(
            np.abs(decision.root_blueprint_policy - decision.root_search_policy).sum()
        )
        self.policy_l1_sum += policy_l1_distance
        self.policy_l1_max = max(self.policy_l1_max, policy_l1_distance)
        self.policy_l1_sum_by_stage[stage_name] += policy_l1_distance
        if int(np.argmax(decision.root_blueprint_policy)) != int(
            np.argmax(decision.root_search_policy)
        ):
            self.policy_argmax_change_count += 1
            self.policy_argmax_changes_by_stage[stage_name] += 1
        if _compact_slot_for_action(state, blueprint_action) != _compact_slot_for_action(
            state,
            decision.action,
        ):
            self.action_change_count += 1
            self.action_changes_by_stage[stage_name] += 1
        if "reach_weighted_beliefs" in decision.diagnostic_flags:
            self.reach_weighted_decision_count += 1
        if any(flag.endswith("blueprint_fallback") for flag in decision.diagnostic_flags):
            self.blueprint_fallback_decision_count += 1
        return decision.action


def _sample_blueprint_action_without_advancing_rng(
    blueprint: BlueprintActionPolicy,
    state: pkrs.State,
    rng: np.random.Generator,
) -> pkrs.Action:
    """Сэмплирует baseline action, сохраняя RNG для фактического runtime решения."""
    rng_state = deepcopy(rng.bit_generator.state)
    try:
        return blueprint.choose_action(state, rng)
    finally:
        rng.bit_generator.state = rng_state


def _postflop_stage_name(state: pkrs.State) -> str:
    """Преобразует поддерживаемую postflop улицу в ключ отчёта."""
    if state.stage == pkrs.Stage.Flop:
        return "flop"
    if state.stage == pkrs.Stage.Turn:
        return "turn"
    if state.stage == pkrs.Stage.River:
        return "river"
    raise ValueError("Runtime telemetry поддерживает только postflop streets")


def _compact_slot_for_action(state: pkrs.State, action: pkrs.Action) -> int | None:
    for slot in ActionSlot:
        try:
            resolved = resolve_action(slot, state).action
        except ValueError:
            continue
        if resolved.action != action.action:
            continue
        if resolved.action != pkrs.ActionEnum.Raise or np.isclose(resolved.amount, action.amount):
            return int(slot)
    return None


__all__ = ["ActionHistoryRecorder", "RuntimeSearchPolicy"]
