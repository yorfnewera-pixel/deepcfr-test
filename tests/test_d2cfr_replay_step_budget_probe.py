import importlib

import numpy as np


probe = importlib.import_module("tools.d2cfr_replay_step_budget_probe")


def test_group_holdout_split_never_places_equal_encoded_infoset_in_both_arms():
    states = np.array(
        [[1.0, 2.0], [1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [5.0, 6.0]],
        dtype=np.float32,
    )
    masks = np.ones((5, 6), dtype=np.float32)

    train_indices, holdout_indices = probe.group_holdout_split(
        states, masks, holdout_fraction=0.4, seed=7
    )

    assert len(train_indices) + len(holdout_indices) == 5
    for first, second in ((0, 1), (3, 4)):
        assert (first in train_indices) == (second in train_indices)
        assert (first in holdout_indices) == (second in holdout_indices)
