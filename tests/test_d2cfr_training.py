import numpy as np
import pytest
import torch

from src.core.action_space import NUM_ACTIONS
from src.core.deep_cfr import DeepCFRAgent
from src.core.traversal_errors import TraversalFailure
from src.utils import config as config_mod


@pytest.fixture
def d2_agent(tmp_path):
    config_path = tmp_path / "d2cfr.yaml"
    config_path.write_text(
        "\n".join(
            (
                "num_actions: 6",
                "num_players: 2",
                "hidden_size: 8",
                "d2cfr_enabled: true",
                "d2cfr_regret_loss_weight: 1.0",
                "d2cfr_state_value_loss_weight: 1.0",
                "d2cfr_action_value_loss_weight: 1.0",
                "d2cfr_reinitialize_each_iteration: true",
                "d2cfr_iteration_weight_power: 1.0",
                "d2cfr_mc_correction_enabled: false",
                "advantage_reward_scale: 200",
                "advantage_train_steps: 1",
                "training_preload_to_device: false",
            )
        )
        + "\n",
        encoding="utf-8",
    )
    config_mod.load_config(config_path)
    try:
        yield DeepCFRAgent(player_id=0, num_players=2, device="cpu")
    finally:
        config_mod.load_config("config.yaml")


def test_d2cfr_targets_share_one_scale_and_preserve_q_minus_v(d2_agent):
    action_values = np.array([10, 4, 0, 0, 0, 0], dtype=np.float32)

    q_targets, state_value_target, regret_targets = d2_agent._normalise_d2cfr_targets(
        action_values,
        state_value=6.0,
        state=object(),
        legal_slots=[0, 1],
    )

    assert q_targets[:2].tolist() == pytest.approx([0.05, 0.02])
    assert state_value_target == pytest.approx(0.03)
    assert regret_targets[:2].tolist() == pytest.approx([0.02, -0.01])
    assert np.allclose(q_targets[:2] - state_value_target, regret_targets[:2])


def test_d2cfr_training_uses_three_masked_counterfactual_losses(d2_agent):
    d2_agent.iteration_count = 2
    for parameter_group in d2_agent.optimizer.param_groups:
        parameter_group["lr"] = 0.0
    mask = np.array([1, 1, 0, 0, 0, 0], dtype=np.float32)
    d2_agent.d2cfr_buffer.add(
        np.zeros(d2_agent.input_size, dtype=np.float32),
        np.array([2, 4, 100, 100, 100, 100], dtype=np.float32),
        3.0,
        np.array([-1, 1, 100, 100, 100, 100], dtype=np.float32),
        mask,
        iteration=2,
    )

    loss = d2_agent.train_d2cfr_advantage_network_multi(batch_size=1)

    assert loss == pytest.approx(20.0)
    assert d2_agent.last_advantage_target_stats == {
        "regret_loss": pytest.approx(1.0),
        "state_value_loss": pytest.approx(9.0),
        "action_value_loss": pytest.approx(10.0),
        "total_loss": pytest.approx(20.0),
    }


def test_d2cfr_empty_buffer_clears_loss_stats_instead_of_reusing_previous_values(d2_agent):
    d2_agent.last_advantage_target_stats = {
        "regret_loss": 1.0,
        "state_value_loss": 2.0,
        "action_value_loss": 3.0,
        "total_loss": 6.0,
    }

    loss = d2_agent.train_d2cfr_advantage_network_multi()

    assert loss == 0.0
    assert d2_agent.last_advantage_train_steps == 0
    assert d2_agent.last_advantage_target_stats is None


def test_d2cfr_reinitializes_network_and_optimizer_without_target_bootstrap(d2_agent):
    d2_agent.iteration_count = 1
    previous_network = d2_agent.advantage_net
    previous_optimizer = d2_agent.optimizer
    d2_agent.d2cfr_buffer.add(
        np.zeros(d2_agent.input_size, dtype=np.float32),
        np.zeros(NUM_ACTIONS, dtype=np.float32),
        0.0,
        np.zeros(NUM_ACTIONS, dtype=np.float32),
        np.array([1, 1, 0, 0, 0, 0], dtype=np.float32),
        iteration=1,
    )

    d2_agent.train_advantage_network_multi(batch_size=1)

    assert d2_agent.advantage_net is not previous_network
    assert d2_agent.optimizer is not previous_optimizer
    assert d2_agent.advantage_target_net is None
    assert len(d2_agent.d2cfr_buffer) == 1


def test_d2cfr_record_keeps_q_v_and_r_atomically_in_traversal_collector(d2_agent):
    state = np.zeros(d2_agent.input_size, dtype=np.float32)
    action_values = np.array([0.05, 0.02, 0, 0, 0, 0], dtype=np.float32)
    regrets = np.array([0.02, -0.01, 0, 0, 0, 0], dtype=np.float32)
    mask = np.array([1, 1, 0, 0, 0, 0], dtype=np.float32)
    d2_agent._active_traversal_collector = d2_agent._new_traversal_sample_collector()

    d2_agent._record_d2cfr_advantage_sample(
        state,
        action_values,
        np.float32(0.03),
        regrets,
        mask,
        iteration=3,
    )

    assert len(d2_agent.d2cfr_buffer) == 0
    d2_agent._commit_traversal_collector(d2_agent._active_traversal_collector)
    _, recorded_q, recorded_v, recorded_r, recorded_mask, recorded_iterations = (
        d2_agent.d2cfr_buffer.sample()
    )
    assert recorded_q[0, :2].tolist() == pytest.approx([0.05, 0.02])
    assert recorded_v.tolist() == pytest.approx([0.03])
    assert recorded_r[0, :2].tolist() == pytest.approx([0.02, -0.01])
    assert recorded_mask.tolist() == [mask.tolist()]
    assert recorded_iterations.tolist() == [3.0]


def test_d2cfr_masked_loss_weights_each_legal_action_not_each_infoset():
    predictions = torch.zeros(2, NUM_ACTIONS)
    targets = torch.tensor(
        [[2, 0, 0, 0, 0, 0], [1, 1, 0, 0, 0, 0]], dtype=torch.float32
    )
    masks = torch.tensor(
        [[1, 0, 0, 0, 0, 0], [1, 1, 0, 0, 0, 0]], dtype=torch.float32
    )
    weights = torch.tensor([0.5, 1.0], dtype=torch.float32)

    loss = DeepCFRAgent._d2cfr_masked_weighted_mse(predictions, targets, masks, weights)

    assert loss.item() == pytest.approx(1.6)


def test_d2cfr_record_rejects_regrets_inconsistent_with_q_minus_v(d2_agent):
    with pytest.raises(TraversalFailure, match="Q - V"):
        d2_agent._record_d2cfr_advantage_sample(
            np.zeros(d2_agent.input_size, dtype=np.float32),
            np.array([0.05, 0.02, 0, 0, 0, 0], dtype=np.float32),
            np.float32(0.03),
            np.zeros(NUM_ACTIONS, dtype=np.float32),
            np.array([1, 1, 0, 0, 0, 0], dtype=np.float32),
            iteration=3,
        )
