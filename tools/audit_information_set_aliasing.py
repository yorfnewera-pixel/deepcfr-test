"""Read-only аудит совпадений tensors information set в seeded игровом дереве."""
from __future__ import annotations

import argparse
import math
import sys
from collections import defaultdict, deque
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pokers as pkrs

# Скрипт должен одинаково работать через ``python tools/...`` и как модуль.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.core.action_space import ACTION_LABELS, ActionSlot, legal_action_mask, resolve_action
from src.core.model import encode_state, encode_state_history_summary_v3


ActionTrace = tuple[ActionSlot, ...]


@dataclass(frozen=True)
class TraceEncoding:
    """Байтовые представления float32 обеих версий encoder для одного trace."""

    legacy_v2: bytes
    history_summary_v3: bytes


@dataclass(frozen=True)
class EncodingAudit:
    """Группирует traces с одинаковым tensor одной версии encoder."""

    tensor_groups: Mapping[bytes, tuple[ActionTrace, ...]]

    @property
    def distinct_tensor_count(self) -> int:
        return len(self.tensor_groups)

    @property
    def collision_buckets(self) -> tuple[tuple[ActionTrace, ...], ...]:
        return tuple(group for group in self.tensor_groups.values() if len(group) > 1)

    @property
    def collision_bucket_count(self) -> int:
        return len(self.collision_buckets)


@dataclass(frozen=True)
class InformationSetAliasingAudit:
    """Детерминированный отчёт о достижимых нетерминальных состояниях до depth."""

    decision_nodes: int
    encodings_by_trace: Mapping[ActionTrace, TraceEncoding]
    legacy_v2: EncodingAudit
    history_summary_v3: EncodingAudit

    def encoding_for_trace(self, trace: Iterable[int | ActionSlot]) -> TraceEncoding:
        """Возвращает encoding, записанный для точного воспроизводимого trace."""
        normalized_trace = tuple(ActionSlot(slot) for slot in trace)
        try:
            return self.encodings_by_trace[normalized_trace]
        except KeyError as error:
            raise KeyError(f"Trace не достигнут в заданном дереве: {format_trace(normalized_trace)}") from error


def _tensor_key(tensor: np.ndarray) -> bytes:
    return np.asarray(tensor, dtype=np.float32).tobytes()


def _freeze_groups(
    groups: Mapping[bytes, list[ActionTrace]],
) -> Mapping[bytes, tuple[ActionTrace, ...]]:
    return {tensor: tuple(traces) for tensor, traces in groups.items()}


def audit_information_set_aliasing(
    *,
    players: int,
    stack: float,
    seed: int,
    max_depth: int,
) -> InformationSetAliasingAudit:
    """Обходит только legal discrete actions и группирует decision-node encodings.

    ``State.apply_action`` возвращает evolved state, поэтому функция не хранит
    ссылки на training trajectories и не изменяет training data.
    """
    if players < 2:
        raise ValueError("Для аудита требуется минимум два игрока")
    if not math.isfinite(stack) or stack <= 0.0:
        raise ValueError("Стек должен быть конечным и положительным")
    if max_depth < 0:
        raise ValueError("max_depth не может быть отрицательным")

    root = pkrs.State.from_seed(
        n_players=players,
        button=0,
        sb=1.0,
        bb=2.0,
        stake=float(stack),
        seed=seed,
    )
    pending: deque[tuple[pkrs.State, ActionTrace]] = deque(((root, ()),))
    encodings_by_trace: dict[ActionTrace, TraceEncoding] = {}
    legacy_groups: dict[bytes, list[ActionTrace]] = defaultdict(list)
    v3_groups: dict[bytes, list[ActionTrace]] = defaultdict(list)

    while pending:
        state, trace = pending.popleft()
        if state.status != pkrs.StateStatus.Ok:
            continue

        legal_slots = np.flatnonzero(legal_action_mask(state))
        if not len(legal_slots):
            continue

        player_id = int(state.current_player)
        legacy_key = _tensor_key(encode_state(state, player_id=player_id))
        v3_key = _tensor_key(encode_state_history_summary_v3(state, player_id=player_id))
        encodings_by_trace[trace] = TraceEncoding(legacy_key, v3_key)
        legacy_groups[legacy_key].append(trace)
        v3_groups[v3_key].append(trace)

        if len(trace) == max_depth:
            continue

        for slot_index in legal_slots:
            slot = ActionSlot(int(slot_index))
            resolved = resolve_action(slot, state)
            next_state = state.apply_action(resolved.action)
            pending.append((next_state, (*trace, slot)))

    return InformationSetAliasingAudit(
        decision_nodes=len(encodings_by_trace),
        encodings_by_trace=encodings_by_trace,
        legacy_v2=EncodingAudit(_freeze_groups(legacy_groups)),
        history_summary_v3=EncodingAudit(_freeze_groups(v3_groups)),
    )


def format_trace(trace: ActionTrace) -> str:
    """Форматирует trace стабильными action-slot labels для replay."""
    return " -> ".join(ACTION_LABELS[int(slot)] for slot in trace) or "<root>"


def _print_collision_examples(name: str, audit: EncodingAudit) -> None:
    print(f"{name}: distinct_tensors={audit.distinct_tensor_count}, "
          f"collision_buckets={audit.collision_bucket_count}")
    for index, traces in enumerate(audit.collision_buckets[:5], start=1):
        print(f"  collision {index}:")
        for trace in traces[:2]:
            print(f"    {format_trace(trace)}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--players", type=int, default=2)
    parser.add_argument("--stack", type=float, default=30.0)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--max-depth", type=int, default=12)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        audit = audit_information_set_aliasing(
            players=args.players,
            stack=args.stack,
            seed=args.seed,
            max_depth=args.max_depth,
        )
    except ValueError as error:
        raise SystemExit(f"Ошибка параметров аудита: {error}") from error
    print(f"decision_nodes={audit.decision_nodes}")
    _print_collision_examples("legacy_v2", audit.legacy_v2)
    _print_collision_examples("history_summary_v3", audit.history_summary_v3)


if __name__ == "__main__":
    main()
