import importlib

import numpy as np
import pytest


probe = importlib.import_module("tools.d2cfr_replay_provenance_probe")


def test_provenance_audit_separates_history_aliasing_from_repeated_same_infoset_targets():
    mask = np.array([0, 0, 0, 0, 0, 1], dtype=np.float32)
    buffer = {
        "states": np.array([[1.0, 2.0], [1.0, 2.0], [1.0, 2.0]], dtype=np.float32),
        "action_values": np.array([[0, 0, 0, 0, 0, 0.2], [0, 0, 0, 0, 0, -0.2], [0, 0, 0, 0, 0, 0.4]], dtype=np.float32),
        "state_values": np.array([0.1, 0.2, 0.4], dtype=np.float32),
        "regrets": np.array([[0, 0, 0, 0, 0, 0.1], [0, 0, 0, 0, 0, -0.4], [0, 0, 0, 0, 0, 0.0]], dtype=np.float32),
        "masks": np.stack((mask, mask, mask)),
        "iterations": np.array([7, 7, 7], dtype=np.float32),
        "provenance_enabled": True,
        "provenances": np.array([
            np.full(16, 1, dtype=np.uint8),
            np.full(16, 2, dtype=np.uint8),
            np.full(16, 2, dtype=np.uint8),
        ]),
    }

    report = probe.audit_leg_buffer(buffer)

    assert report["provenance_available"] is True
    assert report["within_iteration"]["different_history_groups"] == 1
    assert report["within_iteration"]["same_history_duplicate_groups"] == 1
    assert report["within_iteration"]["same_history_state_value_range_mean"] == pytest.approx(0.2)
