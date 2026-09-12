import src.training as training


def test_training_api_does_not_advertise_removed_opponent_modeling_entrypoints():
    assert "train_deep_cfr_with_opponent_modeling" not in training.__all__
    assert "train_mixed_with_opponent_modeling" not in training.__all__
