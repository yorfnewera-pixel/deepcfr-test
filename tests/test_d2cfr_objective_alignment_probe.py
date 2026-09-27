import importlib

import numpy as np


probe = importlib.import_module("tools.d2cfr_objective_alignment_probe")


def test_weighted_target_statistics_separates_within_and_between_iteration_variance():
    values = np.array([0.0, 2.0, 4.0, 6.0], dtype=np.float32)
    iterations = np.array([1, 1, 2, 2], dtype=np.int64)
    weights = np.ones(4, dtype=np.float64)

    result = probe.weighted_target_statistics(values, iterations, weights)

    assert result["mean"] == 3.0
    assert result["variance"] == 5.0
    assert result["within_iteration_variance"] == 1.0
    assert result["between_iteration_variance"] == 4.0


def test_action_target_uses_only_samples_where_action_is_legal():
    regrets = np.array([
        [0.0, 0.0, 0.0, 0.0, 0.0, 2.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, -100.0],
    ], dtype=np.float32)
    masks = np.array([
        [1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 0],
    ], dtype=np.float32)
    iterations = np.array([1, 2], dtype=np.int64)
    weights = np.ones(2, dtype=np.float64)

    result = probe.action_target_statistics(regrets, masks, iterations, weights, action_index=5)

    assert result["samples"] == 1
    assert result["mean"] == 2.0


def test_margin_flip_counts_only_strong_opposite_regrets():
    historical = np.array([-0.2, -0.05, 0.2, 0.3], dtype=np.float32)
    predicted = np.array([0.3, 0.2, -0.2, 0.01], dtype=np.float32)

    result = probe.margin_flip_counts(historical, predicted, margin=0.1)

    assert result == {"historical_negative_predicted_positive": 1, "historical_positive_predicted_negative": 1}
