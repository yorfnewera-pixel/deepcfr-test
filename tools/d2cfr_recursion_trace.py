"""Трассировка рекурсивного D2CFR traversal: parent -> child -> buffer."""

from __future__ import annotations

import argparse
import inspect
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.core.action_space import NUM_ACTIONS


def _float_or_none(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _array(values: Any, shape: tuple[int, ...] | None = None) -> list[float]:
    array = np.asarray(values, dtype=np.float32)
    if shape is not None and array.shape != shape:
        raise ValueError(f"Ожидалась форма {shape}, получена {array.shape}")
    return [float(item) for item in array.reshape(-1).tolist()]


def _legal_slots_from_mask(mask: Any) -> list[int]:
    array = np.asarray(mask, dtype=np.float32)
    if array.shape != (NUM_ACTIONS,):
        return []
    return [int(slot) for slot in np.flatnonzero(array).tolist()]


@dataclass
class RecursionTraceRecorder:
    max_nodes: int = 200
    max_edges: int = 400
    max_buffer_writes: int = 200
    max_depth: int | None = None
    nodes: list[dict[str, Any]] = field(default_factory=list)
    edges: list[dict[str, Any]] = field(default_factory=list)
    buffer_writes: list[dict[str, Any]] = field(default_factory=list)
    _stack: list[int | None] = field(default_factory=list)
    _node_counter: int = 0
    _truncated_nodes: int = 0
    _truncated_edges: int = 0
    _truncated_buffer_writes: int = 0

    def trace_traversal(
        self,
        agent: Any,
        state: Any,
        *,
        iteration: int,
        traversing_player: int,
    ) -> dict[str, Any]:
        original_cfr = agent._cfr_traverse_multi
        original_record = getattr(agent, "_record_d2cfr_advantage_sample", None)

        def traced_cfr(next_state: Any, iter_value: int, player_value: int, depth_value: int) -> float:
            edge_context = self._edge_context_from_caller()
            node_id = self._enter_node(
                agent,
                next_state,
                iteration=int(iter_value),
                traversing_player=int(player_value),
                depth=int(depth_value),
                edge_context=edge_context,
            )
            try:
                returned = float(original_cfr(next_state, iter_value, player_value, depth_value))
            except Exception as error:
                self._leave_node(node_id, error=error)
                raise
            self._leave_node(node_id, return_value=returned)
            return returned

        agent._cfr_traverse_multi = traced_cfr
        if original_record is not None:
            record_fn = original_record

            def traced_record(
                encoded_state: Any,
                action_values: Any,
                state_value: Any,
                regrets: Any,
                mask: Any,
                iter_value: int,
            ) -> Any:
                self.record_buffer_write(
                    action_values=action_values,
                    state_value=state_value,
                    regrets=regrets,
                    mask=mask,
                    iteration=int(iter_value),
                )
                return record_fn(encoded_state, action_values, state_value, regrets, mask, iter_value)

            agent._record_d2cfr_advantage_sample = traced_record
        try:
            result = agent.cfr_traverse_multi(
                state,
                iteration=int(iteration),
                traversing_player=int(traversing_player),
            )
        finally:
            agent._cfr_traverse_multi = original_cfr
            if original_record is not None:
                agent._record_d2cfr_advantage_sample = original_record
        return self.report(root_return=float(result))

    def _edge_context_from_caller(self) -> dict[str, Any]:
        frame = inspect.currentframe()
        caller = frame.f_back.f_back if frame is not None and frame.f_back is not None else None
        if caller is None or not self._stack:
            return {}
        locals_ = caller.f_locals
        context: dict[str, Any] = {
            "parent_id": self._stack[-1],
            "parent_depth": int(locals_.get("depth", -1)),
        }
        if "slot" in locals_:
            context["slot"] = int(locals_["slot"])
        if "action" in locals_:
            context["action"] = repr(locals_["action"])
        if "current_player" in locals_ and locals_["current_player"] is not None:
            context["parent_current_player"] = int(locals_["current_player"])
        if "applied_slots" in locals_:
            context["branch"] = "traverser"
        elif "strategy" in locals_:
            context["branch"] = "opponent"
        elif "external_policy" in locals_:
            context["branch"] = "external_policy"
        elif "_traversal_random_agent" in locals_:
            context["branch"] = "random_agent"
        return context

    def _enter_node(
        self,
        agent: Any,
        state: Any,
        *,
        iteration: int,
        traversing_player: int,
        depth: int,
        edge_context: dict[str, Any],
    ) -> int | None:
        self._node_counter += 1
        node_id = self._node_counter
        should_record = len(self.nodes) < self.max_nodes and (
            self.max_depth is None or depth <= self.max_depth
        )
        if not should_record:
            self._truncated_nodes += 1
            self._stack.append(None)
            return None

        is_terminal = bool(getattr(state, "final_state", False))
        current_player = None if is_terminal else _float_or_none(getattr(state, "current_player", None))
        legal_slots: list[int] = []
        if not is_terminal:
            try:
                legal_slots = _legal_slots_from_mask(agent.get_legal_action_mask(state))
            except Exception:
                legal_slots = []
        node = {
            "id": node_id,
            "parent_id": edge_context.get("parent_id"),
            "depth": depth,
            "iteration": iteration,
            "traversing_player": traversing_player,
            "current_player": None if current_player is None else int(current_player),
            "is_terminal": is_terminal,
            "legal_slots": legal_slots,
            "branching_factor": len(legal_slots),
        }
        self.nodes.append(node)
        self._stack.append(node_id)
        if edge_context.get("parent_id") is not None and len(self.edges) < self.max_edges:
            edge = {
                "parent_id": edge_context.get("parent_id"),
                "child_id": node_id,
                "parent_depth": edge_context.get("parent_depth"),
                "child_depth": depth,
                "slot": edge_context.get("slot"),
                "action": edge_context.get("action"),
                "branch": edge_context.get("branch"),
                "return_value": None,
            }
            self.edges.append(edge)
        elif edge_context.get("parent_id") is not None:
            self._truncated_edges += 1
        return node_id

    def _leave_node(
        self,
        node_id: int | None,
        *,
        return_value: float | None = None,
        error: Exception | None = None,
    ) -> None:
        self._stack.pop()
        if node_id is None:
            return
        for node in reversed(self.nodes):
            if node["id"] == node_id:
                if error is not None:
                    node["error"] = repr(error)
                else:
                    assert return_value is not None
                    node["return_value"] = float(return_value)
                break
        if return_value is not None:
            for edge in reversed(self.edges):
                if edge["child_id"] == node_id:
                    edge["return_value"] = float(return_value)
                    break

    def record_buffer_write(
        self,
        *,
        action_values: Any,
        state_value: Any,
        regrets: Any,
        mask: Any,
        iteration: int,
    ) -> None:
        node_id = self._stack[-1] if self._stack else None
        if node_id is None or len(self.buffer_writes) >= self.max_buffer_writes:
            self._truncated_buffer_writes += 1
            return
        legal_slots = _legal_slots_from_mask(mask)
        action_values_list = _array(action_values, (NUM_ACTIONS,))
        regrets_list = _array(regrets, (NUM_ACTIONS,))
        state_value_float = float(np.asarray(state_value, dtype=np.float32).item())
        self.buffer_writes.append(
            {
                "node_id": node_id,
                "iteration": iteration,
                "legal_slots": legal_slots,
                "action_values": action_values_list,
                "state_value": state_value_float,
                "regrets": regrets_list,
                "legal": [
                    {
                        "slot": slot,
                        "q": action_values_list[slot],
                        "regret": regrets_list[slot],
                    }
                    for slot in legal_slots
                ],
            }
        )

    def report(self, *, root_return: float) -> dict[str, Any]:
        fanouts: dict[int, int] = {}
        for edge in self.edges:
            parent = edge.get("parent_id")
            if parent is not None:
                fanouts[int(parent)] = fanouts.get(int(parent), 0) + 1
        return {
            "summary": {
                "root_return": root_return,
                "nodes_recorded": len(self.nodes),
                "edges_recorded": len(self.edges),
                "buffer_writes_recorded": len(self.buffer_writes),
                "branching_nodes": sum(1 for count in fanouts.values() if count > 1),
                "max_recorded_depth": max((int(node["depth"]) for node in self.nodes), default=0),
                "truncated_nodes": self._truncated_nodes,
                "truncated_edges": self._truncated_edges,
                "truncated_buffer_writes": self._truncated_buffer_writes,
            },
            "nodes": self.nodes,
            "edges": self.edges,
            "buffer_writes": self.buffer_writes,
        }


def run_recursion_trace(
    agent: Any,
    state: Any,
    *,
    iteration: int,
    traversing_player: int,
    max_nodes: int = 200,
    max_edges: int = 400,
    max_buffer_writes: int = 200,
    max_depth: int | None = None,
) -> dict[str, Any]:
    recorder = RecursionTraceRecorder(
        max_nodes=max_nodes,
        max_edges=max_edges,
        max_buffer_writes=max_buffer_writes,
        max_depth=max_depth,
    )
    return recorder.trace_traversal(
        agent,
        state,
        iteration=iteration,
        traversing_player=traversing_player,
    )


def run_live_recursion_trace(
    config_path: Path,
    *,
    seed: int,
    iteration: int,
    traversing_player: int,
    root_index: int,
    num_players: int | None,
    max_nodes: int,
    max_edges: int,
    max_buffer_writes: int,
    max_depth: int | None,
    device: str,
) -> dict[str, Any]:
    from src.core.deep_cfr import DeepCFRAgent
    from src.utils import config as config_mod
    from tools.benchmark_traversal import generate_root_bank, make_state, seed_everything

    config_mod.load_config(config_path)
    configured_players = config_mod.cfg_get("num_players", 6)
    player_count = int(num_players) if num_players is not None else int(configured_players or 6)
    roots = generate_root_bank(root_index + 1, seed, player_count)
    root = roots[root_index]
    seed_everything(seed)
    agent = DeepCFRAgent(
        player_id=int(traversing_player),
        num_players=player_count,
        device=device,
    )
    agent.prepare_iteration(iteration, traversing_player)
    agent.reset_traversal_stats()
    report = run_recursion_trace(
        agent,
        make_state(root, player_count),
        iteration=iteration,
        traversing_player=traversing_player,
        max_nodes=max_nodes,
        max_edges=max_edges,
        max_buffer_writes=max_buffer_writes,
        max_depth=max_depth,
    )
    report["root"] = {
        "button": int(root.button),
        "deal_seed": int(root.deal_seed),
        "root_index": int(root_index),
        "seed": int(seed),
        "num_players": player_count,
        "config": str(config_path),
    }
    report["traversal_stats"] = agent.get_traversal_stats()
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("config.yaml"))
    parser.add_argument("--seed", type=int, default=3)
    parser.add_argument("--iteration", type=int, default=1)
    parser.add_argument("--traversing-player", type=int, default=0)
    parser.add_argument("--root-index", type=int, default=0)
    parser.add_argument("--num-players", type=int, default=None)
    parser.add_argument("--max-nodes", type=int, default=200)
    parser.add_argument("--max-edges", type=int, default=400)
    parser.add_argument("--max-buffer-writes", type=int, default=200)
    parser.add_argument("--max-depth", type=int, default=None)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = run_live_recursion_trace(
        args.config,
        seed=args.seed,
        iteration=args.iteration,
        traversing_player=args.traversing_player,
        root_index=args.root_index,
        num_players=args.num_players,
        max_nodes=args.max_nodes,
        max_edges=args.max_edges,
        max_buffer_writes=args.max_buffer_writes,
        max_depth=args.max_depth,
        device=args.device,
    )
    payload = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload + "\n", encoding="utf-8")
    print(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
