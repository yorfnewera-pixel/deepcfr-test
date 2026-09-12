import numpy as np
import pytest
import torch

from src.core.action_space import NUM_ACTIONS
from src.core import buffers as buffers_mod
from src.core.buffers import DuelingAdvantageBuffer
from src.core.model import (
    CARD_CONTEXT_ARCHITECTURE,
    MONOLITHIC_ARCHITECTURE,
    DuelingRegretNetwork,
)
from src.utils import config as config_mod


@pytest.mark.parametrize(
    "architecture",
    (MONOLITHIC_ARCHITECTURE, CARD_CONTEXT_ARCHITECTURE),
)
def test_dueling_network_returns_regret_as_action_value_minus_state_value(architecture):
    network = DuelingRegretNetwork(181, hidden_size=16, architecture=architecture)
    state = torch.randn(3, 181)

    result = network.forward_components(state)

    assert result.state_values.shape == (3, 1)
    assert result.action_values.shape == (3, NUM_ACTIONS)
    assert result.regrets.shape == (3, NUM_ACTIONS)
    assert torch.equal(result.regrets, result.action_values - result.state_values)
    assert torch.equal(network(state), result.regrets)


def test_dueling_card_encoder_ignores_context_features():
    network = DuelingRegretNetwork(181, hidden_size=16, architecture=CARD_CONTEXT_ARCHITECTURE)
    first = torch.zeros(1, 181)
    second = first.clone()
    second[:, 109:] = 1.0

    assert torch.equal(network.encode_cards(first), network.encode_cards(second))


def test_dueling_buffer_stores_all_counterfactual_targets():
    buffer = DuelingAdvantageBuffer(capacity=2, state_dim=3)
    status = buffer.add(
        state=np.array([1.0, 2.0, 3.0], dtype=np.float32),
        action_values=np.arange(NUM_ACTIONS, dtype=np.float32),
        state_value=np.float32(2.5),
        regrets=np.arange(NUM_ACTIONS, dtype=np.float32) - 2.5,
        mask=np.array([1, 1, 0, 0, 0, 0], dtype=np.float32),
        iteration=7,
    )

    states, action_values, state_values, regrets, masks, iterations = buffer.sample()

    assert status == "recorded"
    assert states.tolist() == [[1.0, 2.0, 3.0]]
    assert action_values.shape == regrets.shape == masks.shape == (1, NUM_ACTIONS)
    assert state_values.shape == (1,)
    assert state_values.tolist() == [2.5]
    assert iterations.tolist() == [7.0]


@pytest.mark.parametrize(
    "field,value,message",
    (
        ("state", np.full(3, np.nan, dtype=np.float32), "конеч"),
        ("action_values", np.full(NUM_ACTIONS, np.nan, dtype=np.float32), "конеч"),
        ("state_value", np.float32(np.nan), "конеч"),
        ("regrets", np.full(NUM_ACTIONS, np.nan, dtype=np.float32), "конеч"),
        ("mask", np.full(NUM_ACTIONS, 0.5, dtype=np.float32), "mask"),
        ("iteration", 0, "iteration"),
    ),
)
def test_dueling_buffer_rejects_invalid_sample_before_mutation(field, value, message):
    buffer = DuelingAdvantageBuffer(capacity=2, state_dim=3)
    sample = {
        "state": np.zeros(3, dtype=np.float32),
        "action_values": np.zeros(NUM_ACTIONS, dtype=np.float32),
        "state_value": np.float32(0.0),
        "regrets": np.zeros(NUM_ACTIONS, dtype=np.float32),
        "mask": np.ones(NUM_ACTIONS, dtype=np.float32),
        "iteration": 1,
    }
    sample[field] = value

    with pytest.raises(ValueError, match=message):
        buffer.add(**sample)

    assert len(buffer) == 0
    assert buffer._cur_id == 0
    assert buffer.eviction_count == 0
    assert buffer.skip_count == 0


def test_dueling_buffer_uses_reservoir_replacement(monkeypatch):
    buffer = DuelingAdvantageBuffer(capacity=2, state_dim=1)
    monkeypatch.setattr(np.random, "randint", lambda *_args: 0)
    mask = np.array([1, 1, 0, 0, 0, 0], dtype=np.float32)

    for iteration in (1, 2, 3):
        buffer.add(
            np.array([float(iteration)], dtype=np.float32),
            np.zeros(NUM_ACTIONS, dtype=np.float32),
            0.0,
            np.zeros(NUM_ACTIONS, dtype=np.float32),
            mask,
            iteration,
        )

    states, _, _, _, _, iterations = buffer.sample()
    assert sorted(states.reshape(-1).tolist()) == [2.0, 3.0]
    assert sorted(iterations.tolist()) == [2.0, 3.0]
    assert buffer.eviction_count == 1


def test_dueling_buffer_requires_six_action_slots():
    with pytest.raises(ValueError):
        DuelingAdvantageBuffer(capacity=2, state_dim=3, num_actions=NUM_ACTIONS - 1)


def test_dueling_buffer_is_exported_as_public_buffer_contract():
    assert "DuelingAdvantageBuffer" in buffers_mod.__all__


def test_d2cfr_defaults_to_anchored_loss_contract(tmp_path):
    config_path = tmp_path / "d2cfr-defaults.yaml"
    config_path.write_text("num_actions: 6\nd2cfr_enabled: true\n", encoding="utf-8")

    try:
        config_mod.load_config(config_path)

        assert config_mod.cfg_get("d2cfr_loss_mode") == "anchored"
        assert config_mod.cfg_get("d2cfr_loss_function") == "huber"
        assert config_mod.cfg_get("d2cfr_state_value_loss_weight") == pytest.approx(0.5)
        assert config_mod.cfg_get("d2cfr_iteration_weight_mode") == "batch_mean_1"
    finally:
        config_mod.load_config("config.yaml")


@pytest.mark.parametrize(
    "contents,message",
    (
        ("d2cfr_enabled: true\nd2cfr_mc_correction_enabled: true\n", "MC correction"),
        ("d2cfr_enabled: true\nadvantage_regret_clip: 1.0\n", "clip"),
        (
            "d2cfr_enabled: true\nd2cfr_action_value_loss_weight: 1\n",
            "d2cfr_action_value_loss_weight",
        ),
        ("d2cfr_enabled: true\ndiscount_gamma: 1.0\n", "discount_gamma"),
        (
            "d2cfr_enabled: true\nadvantage_buffer_reservoir: false\n",
            "advantage_buffer_reservoir",
        ),
        (
            "d2cfr_enabled: true\nd2cfr_reinitialize_each_iteration: false\n",
            "reinitialize_each_iteration",
        ),
        ("d2cfr_enabled: true\nstrategy_train_every: 0\n", "strategy_train_every"),
        (
            "d2cfr_enabled: true\nstrategy_final_train_steps: 0\n",
            "strategy_final_train_steps",
        ),
    ),
)
def test_config_rejects_unsupported_d2cfr_combinations(tmp_path, contents, message):
    config_path = tmp_path / "invalid-d2cfr.yaml"
    config_path.write_text(f"num_actions: 6\n{contents}", encoding="utf-8")

    try:
        with pytest.raises(ValueError, match=message):
            config_mod.load_config(config_path)
    finally:
        config_mod.load_config("config.yaml")


@pytest.mark.parametrize(
    "contents,flag_name",
    (
        ("d2cfr_enabled: 'false'\n", "d2cfr_enabled"),
        (
            "d2cfr_enabled: true\nd2cfr_mc_correction_enabled: 'false'\n",
            "d2cfr_mc_correction_enabled",
        ),
        (
            "d2cfr_enabled: true\nd2cfr_reinitialize_each_iteration: 'false'\n",
            "d2cfr_reinitialize_each_iteration",
        ),
    ),
)
def test_config_requires_boolean_d2cfr_flags(tmp_path, contents, flag_name):
    config_path = tmp_path / "non-boolean-d2cfr.yaml"
    config_path.write_text(f"num_actions: 6\n{contents}", encoding="utf-8")

    try:
        with pytest.raises(ValueError, match=flag_name):
            config_mod.load_config(config_path)
    finally:
        config_mod.load_config("config.yaml")


def test_config_keeps_last_valid_values_after_rejected_d2cfr_file(tmp_path):
    config_path = tmp_path / "invalid-d2cfr.yaml"
    config_path.write_text(
        "num_actions: 6\nd2cfr_enabled: true\nd2cfr_mc_correction_enabled: true\n",
        encoding="utf-8",
    )
    config_mod.load_config("config.yaml")
    expected = config_mod.cfg_all()

    try:
        with pytest.raises(ValueError, match="MC correction"):
            config_mod.load_config(config_path)
        assert config_mod.cfg_all() == expected
    finally:
        config_mod.load_config("config.yaml")
