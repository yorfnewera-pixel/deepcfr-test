import importlib

import numpy as np


probe = importlib.import_module("tools.d2cfr_replay_collision_probe")


def test_audit_leg_buffer_separates_same_iteration_conflict_from_cross_iteration_drift():
    mask = np.array([0, 0, 0, 0, 0, 1], dtype=np.float32)
    buffer = {
        "states": np.array([[1.0, 2.0], [1.0, 2.0], [1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
        "action_values": np.array([[0, 0, 0, 0, 0, 0.2], [0, 0, 0, 0, 0, -0.3], [0, 0, 0, 0, 0, 0.4], [0, 0, 0, 0, 0, 0.1]], dtype=np.float32),
        "state_values": np.zeros(4, dtype=np.float32),
        "regrets": np.array([[0, 0, 0, 0, 0, 0.2], [0, 0, 0, 0, 0, -0.3], [0, 0, 0, 0, 0, 0.4], [0, 0, 0, 0, 0, 0.1]], dtype=np.float32),
        "masks": np.stack((mask, mask, mask, mask)),
        "iterations": np.array([1, 1, 2, 1], dtype=np.float32),
    }

    report = probe.audit_leg_buffer(buffer)

    assert report["samples"] == 4
    assert report["exact_collision_groups"] == 1
    assert report["collision_samples"] == 3
    assert report["within_iteration"]["duplicate_groups"] == 1
    assert report["within_iteration"]["groups_with_all_in_sign_disagreement"] == 1
    assert report["between_iterations"]["groups_with_multiple_iterations"] == 1
    assert report["between_iterations"]["all_in_mean_sign_disagreement_groups"] == 1
