"""Единый контракт дискретных действий покерного агента.

Размер рейза всегда измеряется как добавка сверх обязательного колла. Пот для
относительных рейзов берётся до применения текущего действия.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum

import numpy as np
import pokers as pkrs


class ActionSlot(IntEnum):
    FOLD = 0
    CHECK = 1
    CALL = 2
    RAISE_HALF_POT = 3
    RAISE_POT = 4
    ALL_IN = 5


NUM_ACTIONS = len(ActionSlot)
ACTION_SPACE_VERSION = "six_fixed_v2"
ACTION_LABELS = (
    "fold",
    "check",
    "call",
    "raise_0.5pot",
    "raise_1pot",
    "all_in",
)
_RAISE_SLOTS = frozenset((
    ActionSlot.RAISE_HALF_POT,
    ActionSlot.RAISE_POT,
    ActionSlot.ALL_IN,
))


@dataclass(frozen=True)
class ResolvedAction:
    """Дискретный слот и его точное представление для движка."""

    slot: ActionSlot
    action: pkrs.Action
    call_amount: float = 0.0
    additional_amount: float = 0.0


def is_raise_slot(slot: int | ActionSlot) -> bool:
    """Возвращает ``True`` только для трёх независимых raise-действий."""
    try:
        return ActionSlot(slot) in _RAISE_SLOTS
    except ValueError:
        return False


def raise_slots() -> tuple[ActionSlot, ...]:
    return (
        ActionSlot.RAISE_HALF_POT,
        ActionSlot.RAISE_POT,
        ActionSlot.ALL_IN,
    )


def _current_player_state(state):
    return state.players_state[int(state.current_player)]


def has_raise_on_current_street(state) -> bool:
    """Проверяет, был ли добровольный рейз на текущей улице."""
    return any(
        getattr(player_state, "last_stage_action", None) == pkrs.ActionEnum.Raise
        for player_state in state.players_state
    )


def call_amount(state) -> float:
    """Чипсы, нужные игроку для колла до применения действия."""
    player_state = _current_player_state(state)
    return max(0.0, float(state.min_bet) - float(player_state.bet_chips))


def remaining_after_call(state) -> float:
    """Остаток стека после обязательного колла."""
    return max(0.0, float(_current_player_state(state).stake) - call_amount(state))


def min_raise_increment(state) -> float:
    """Минимальная добавка сверх колла по доступному состоянию движка."""
    last_increment = getattr(state, "last_raise_increment", None)
    if last_increment is not None and float(last_increment) > 0.0:
        return max(1.0, float(last_increment))
    big_blind = getattr(state, "bb", None)
    if big_blind is not None and float(big_blind) > 0.0:
        return max(1.0, float(big_blind))
    return 1.0


def _raise_amount(slot: ActionSlot, state) -> float:
    if slot is ActionSlot.RAISE_HALF_POT:
        return 0.5 * float(state.pot)
    if slot is ActionSlot.RAISE_POT:
        return float(state.pot)
    if slot is ActionSlot.ALL_IN:
        return remaining_after_call(state)
    raise ValueError(f"Слот {slot!r} не является рейзом")


def resolve_action(slot: int | ActionSlot, state) -> ResolvedAction:
    """Преобразует слот в допустимое действие движка без скрытых fallback.

    Полпота и пот не округляются до min-raise и не преобразуются в all-in:
    если размер не является самостоятельным действием, соответствующий слот
    считается нелегальным. Это сохраняет однозначность CFR-regret.
    """
    try:
        action_slot = ActionSlot(slot)
    except ValueError as error:
        raise ValueError(f"Неизвестный слот действия: {slot}") from error

    legal_actions = tuple(getattr(state, "legal_actions", ()))
    if action_slot is ActionSlot.FOLD:
        if pkrs.ActionEnum.Fold not in legal_actions:
            raise ValueError("Fold недоступен в текущем состоянии")
        return ResolvedAction(action_slot, pkrs.Action(pkrs.ActionEnum.Fold))
    if action_slot is ActionSlot.CHECK:
        if pkrs.ActionEnum.Check not in legal_actions:
            raise ValueError("Check недоступен в текущем состоянии")
        return ResolvedAction(action_slot, pkrs.Action(pkrs.ActionEnum.Check))
    if action_slot is ActionSlot.CALL:
        if pkrs.ActionEnum.Call not in legal_actions:
            raise ValueError("Call недоступен в текущем состоянии")
        return ResolvedAction(action_slot, pkrs.Action(pkrs.ActionEnum.Call))

    if action_slot is ActionSlot.RAISE_HALF_POT and has_raise_on_current_street(state):
        raise ValueError("Рейз в полпота недоступен после рейза на текущей улице")

    if pkrs.ActionEnum.Raise not in legal_actions:
        raise ValueError("Raise недоступен в текущем состоянии")

    amount_to_call = call_amount(state)
    remaining = remaining_after_call(state)
    amount = _raise_amount(action_slot, state)
    if amount <= 0.0:
        raise ValueError("Для рейза не осталось фишек после колла")

    if action_slot is ActionSlot.ALL_IN:
        return ResolvedAction(
            action_slot,
            pkrs.Action(pkrs.ActionEnum.Raise, amount),
            call_amount=amount_to_call,
            additional_amount=amount,
        )

    if amount < min_raise_increment(state):
        raise ValueError("Размер рейза меньше минимального рейза")
    if amount >= remaining:
        raise ValueError("Размер рейза совпадает с all-in или превышает его")

    return ResolvedAction(
        action_slot,
        pkrs.Action(pkrs.ActionEnum.Raise, amount),
        call_amount=amount_to_call,
        additional_amount=amount,
    )


def legal_action_mask(state) -> np.ndarray:
    """Возвращает маску шести слотов, согласованную с ``resolve_action``."""
    mask = np.zeros(NUM_ACTIONS, dtype=np.float32)
    for slot in ActionSlot:
        try:
            resolve_action(slot, state)
        except (AttributeError, TypeError, ValueError):
            continue
        mask[int(slot)] = 1.0
    return mask


__all__ = [
    "ACTION_LABELS",
    "ACTION_SPACE_VERSION",
    "ActionSlot",
    "NUM_ACTIONS",
    "ResolvedAction",
    "call_amount",
    "has_raise_on_current_street",
    "is_raise_slot",
    "legal_action_mask",
    "min_raise_increment",
    "raise_slots",
    "remaining_after_call",
    "resolve_action",
]
