import importlib

import numpy as np

from src.core.action_space import NUM_ACTIONS

recursion_trace = importlib.import_module("tools.d2cfr_recursion_trace")


class FakeRecursiveAgent:
    def __init__(self):
        self.d2cfr_enabled = True
        self.recorded = []

    def cfr_traverse_multi(self, state, iteration, traversing_player):
        return self._cfr_traverse_multi(state, iteration, traversing_player, 0)

    def _cfr_traverse_multi(self, state, iteration, traversing_player, depth):
        if state.startswith("leaf"):
            return float(state.rsplit("_", 1)[-1])

        legal_slots = [0, 2]
        action_values = np.zeros(NUM_ACTIONS, dtype=np.float32)
        applied_slots = []
        for slot in legal_slots:
            action = f"slot-{slot}"
            action_values[slot] = self._cfr_traverse_multi(
                f"leaf_{slot + 1}",
                iteration,
                traversing_player,
                depth + 1,
            )
            applied_slots.append(slot)
        state_value = float(action_values[applied_slots].mean())
        regrets = np.zeros(NUM_ACTIONS, dtype=np.float32)
        for slot in applied_slots:
            regrets[slot] = action_values[slot] - state_value
        mask = np.zeros(NUM_ACTIONS, dtype=np.float32)
        mask[applied_slots] = 1.0
        self._record_d2cfr_advantage_sample(
            np.zeros(3, dtype=np.float32),
            action_values,
            np.float32(state_value),
            regrets,
            mask,
            iteration,
        )
        return state_value

    def _record_d2cfr_advantage_sample(
        self,
        encoded,
        action_values,
        state_value,
        regrets,
        mask,
        iteration,
    ):
        self.recorded.append((encoded, action_values, state_value, regrets, mask, iteration))

    def get_legal_action_mask(self, state):
        mask = np.zeros(NUM_ACTIONS, dtype=np.float32)
        if state == "root":
            mask[[0, 2]] = 1.0
        return mask


def test_recursion_trace_connects_parent_children_and_buffer_write():
    agent = FakeRecursiveAgent()

    report = recursion_trace.run_recursion_trace(
        agent,
        "root",
        iteration=1,
        traversing_player=0,
        max_nodes=10,
        max_edges=10,
    )

    assert report["summary"]["root_return"] == 2.0
    assert report["summary"]["nodes_recorded"] == 3
    assert report["summary"]["edges_recorded"] == 2
    assert report["summary"]["branching_nodes"] == 1
    assert [edge["slot"] for edge in report["edges"]] == [0, 2]
    assert [edge["return_value"] for edge in report["edges"]] == [1.0, 3.0]

    write = report["buffer_writes"][0]
    assert write["node_id"] == 1
    assert write["legal_slots"] == [0, 2]
    assert write["state_value"] == 2.0
    assert write["action_values"][0] == 1.0
    assert write["action_values"][2] == 3.0
    assert write["regrets"][0] == -1.0
    assert write["regrets"][2] == 1.0
