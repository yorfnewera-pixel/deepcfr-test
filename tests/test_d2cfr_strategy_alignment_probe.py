import importlib

import numpy as np


probe = importlib.import_module("tools.d2cfr_strategy_alignment_probe")


def test_policy_target_ignores_samples_where_action_is_illegal():
    policies = np.array([[0, 0, 0, 0, 0, 1], [0, 0, 0, 0, 0, 0]], dtype=np.float32)
    masks = np.array([[1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 0]], dtype=np.float32)
    iterations = np.array([1, 2], dtype=np.int64)

    target = probe.historical_policy_target(policies, masks, iterations)

    assert target[5] == 1.0


def test_all_in_extreme_summary_reports_inflation_rate():
    historical = np.array([0.05, 0.2, 0.8], dtype=np.float32)
    predicted = np.array([0.8, 0.1, 0.1], dtype=np.float32)

    result = probe.all_in_extreme_summary(historical, predicted)

    assert result["historical_low_predicted_high"] == 1
    assert result["historical_low_count"] == 1
    assert result["historical_low_predicted_high_rate"] == 1.0
