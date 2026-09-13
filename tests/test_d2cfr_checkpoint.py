import random

import numpy as np
import pytest
import torch

from src.core.action_space import NUM_ACTIONS
from src.core.deep_cfr import DeepCFRAgent
from src.core.model import CARD_CONTEXT_ARCHITECTURE, CARD_CONTEXT_V2_ARCHITECTURE
from src.training import train as train_mod
from src.utils import config as config_mod


def _assert_nested_state_equal(actual, expected):
    assert actual.keys() == expected.keys()
    for key, expected_value in expected.items():
        actual_value = actual[key]
        if isinstance(expected_value, dict):
            _assert_nested_state_equal(actual_value, expected_value)
        elif isinstance(expected_value, list):
            assert len(actual_value) == len(expected_value)
            for actual_item, expected_item in zip(actual_value, expected_value, strict=True):
                if isinstance(expected_item, dict):
                    _assert_nested_state_equal(actual_item, expected_item)
                else:
                    assert actual_item == expected_item
        elif torch.is_tensor(expected_value):
            assert torch.equal(actual_value, expected_value)
        else:
            assert actual_value == expected_value


def _step_optimizer(optimizer):
    for parameter in optimizer.param_groups[0]["params"]:
        parameter.grad = torch.ones_like(parameter)
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)


@pytest.fixture
def d2_hu_agent(tmp_path):
    config_path = tmp_path / "hu-d2cfr.yaml"
    config_path.write_text(
        "\n".join(
            (
                "num_actions: 6",
                "num_players: 2",
                "num_trainable_players: 2",
                "hu_current_policy_self_play: true",
                "hidden_size: 8",
                "d2cfr_enabled: true",
                "d2cfr_mc_correction_enabled: false",
                "advantage_memory_size: 8",
                "strategy_memory_size: 8",
            )
        )
        + "\n",
        encoding="utf-8",
    )
    config_mod.load_config(config_path)
    try:
        agent = DeepCFRAgent(player_id=0, num_players=2, device="cpu")
        train_mod._create_hu_current_policy_coordinator(agent)
        agent.iteration_count = 4
        mask = np.array([1, 1, 0, 0, 0, 0], dtype=np.float32)
        for player_id, buffer in enumerate(agent.hu_advantage_buffers):
            buffer.add(
                np.full(agent.input_size, player_id, dtype=np.float32),
                np.array([0.05, 0.02, 0, 0, 0, 0], dtype=np.float32),
                0.03,
                np.array([0.02, -0.01, 0, 0, 0, 0], dtype=np.float32),
                mask,
                iteration=4,
            )
        yield agent
    finally:
        config_mod.load_config("config.yaml")


def test_hu_d2_checkpoint_serializes_two_dueling_legs_without_target_networks(d2_hu_agent):
    checkpoint = train_mod._build_hu_checkpoint(d2_hu_agent, seed=17)

    assert checkpoint["algorithm_variant"] == "d2cfr_dueling_v1"
    assert checkpoint["training_target_semantics"] == "counterfactual_q_v_regret_q_minus_v1"
    assert checkpoint["d2cfr_config"]["target_normalization"] == "shared_advantage_reward_scale"
    assert checkpoint["config"]["d2cfr_historical_advantage_reservoir"] is True
    assert "discount_gamma" not in checkpoint["config"]
    assert "advantage_target" not in checkpoint["architecture"]
    assert all("target_network" not in leg for leg in checkpoint["advantage_legs"])
    assert all({"action_values", "state_values", "regrets"} <= set(leg["buffer"]) for leg in checkpoint["advantage_legs"])


def test_hu_d2_checkpoint_round_trip_restores_two_legs(tmp_path, d2_hu_agent):
    for optimizer in (*d2_hu_agent.hu_advantage_optimizers, d2_hu_agent.strategy_optimizer):
        _step_optimizer(optimizer)
    path = train_mod._save_hu_checkpoint(d2_hu_agent, tmp_path / "hu-d2.pt", seed=17)
    expected_rng = (random.random(), np.random.random(), torch.rand(1))

    config_path = tmp_path / "restore.yaml"
    config_path.write_text(
        "\n".join(
            (
                "num_actions: 6", "num_players: 2", "num_trainable_players: 2",
                "hu_current_policy_self_play: true", "hidden_size: 8", "d2cfr_enabled: true",
                "d2cfr_mc_correction_enabled: false", "advantage_memory_size: 8", "strategy_memory_size: 8",
            )
        ) + "\n",
        encoding="utf-8",
    )
    config_mod.load_config(config_path)
    try:
        restored = DeepCFRAgent(player_id=0, num_players=2, device="cpu")
        train_mod._create_hu_current_policy_coordinator(restored)
        train_mod._load_hu_checkpoint(restored, path)
    finally:
        config_mod.load_config("config.yaml")

    assert restored.iteration_count == 4
    assert [len(buffer) for buffer in restored.hu_advantage_buffers] == [1, 1]
    assert restored.hu_advantage_target_nets is None
    actual_rng = (random.random(), np.random.random(), torch.rand(1))
    assert actual_rng[0] == expected_rng[0]
    assert actual_rng[1] == expected_rng[1]
    assert torch.equal(actual_rng[2], expected_rng[2])
    state = torch.zeros((1, d2_hu_agent.input_size), dtype=torch.float32)
    for source, restored_leg in zip(
        d2_hu_agent.hu_advantage_nets,
        restored.hu_advantage_nets,
        strict=True,
    ):
        source_output = source.forward_components(state)
        restored_output = restored_leg.forward_components(state)
        assert torch.equal(restored_output.state_values, source_output.state_values)
        assert torch.equal(restored_output.action_values, source_output.action_values)
        assert torch.equal(restored_output.regrets, source_output.regrets)
    for source_optimizer, restored_optimizer in zip(
        d2_hu_agent.hu_advantage_optimizers,
        restored.hu_advantage_optimizers,
        strict=True,
    ):
        _assert_nested_state_equal(restored_optimizer.state_dict(), source_optimizer.state_dict())
    _assert_nested_state_equal(
        restored.strategy_optimizer.state_dict(), d2_hu_agent.strategy_optimizer.state_dict()
    )


def test_hu_d2_resume_rejects_v2_weights_forged_as_v1_before_network_load(tmp_path, monkeypatch):
    source_config = tmp_path / "d2-v2-source.yaml"
    target_config = tmp_path / "d2-v1-target.yaml"
    source_config.write_text(
        "\n".join((
            "num_actions: 6", "num_players: 2", "num_trainable_players: 2",
            "hu_current_policy_self_play: true", "hidden_size: 8", "d2cfr_enabled: true",
            "d2cfr_mc_correction_enabled: false", "network_architecture: card_context_v2",
        )) + "\n",
        encoding="utf-8",
    )
    target_config.write_text(
        "\n".join((
            "num_actions: 6", "num_players: 2", "num_trainable_players: 2",
            "hu_current_policy_self_play: true", "hidden_size: 8", "d2cfr_enabled: true",
            "d2cfr_mc_correction_enabled: false", "network_architecture: card_context_v1",
        )) + "\n",
        encoding="utf-8",
    )
    config_mod.load_config(source_config)
    try:
        source = DeepCFRAgent(player_id=0, num_players=2, device="cpu")
        train_mod._create_hu_current_policy_coordinator(source)
        checkpoint = train_mod._build_hu_checkpoint(source)
        for network_schema in (*checkpoint["architecture"]["advantage"], checkpoint["architecture"]["strategy"]):
            network_schema["network_architecture"] = CARD_CONTEXT_ARCHITECTURE
            network_schema.pop("fusion_input_size")
            network_schema.pop("fusion_output_size")
        path = tmp_path / "forged-v1-d2-hu.pt"
        torch.save(checkpoint, path)
    finally:
        config_mod.load_config(target_config)
    try:
        restored = DeepCFRAgent(player_id=0, num_players=2, device="cpu")
        train_mod._create_hu_current_policy_coordinator(restored)
        load_calls = []

        def unexpected_load(*args, **kwargs):
            load_calls.append((args, kwargs))
            raise AssertionError("load_state_dict не должен вызываться для forged architecture")

        monkeypatch.setattr(restored.hu_advantage_nets[0], "load_state_dict", unexpected_load)
        with pytest.raises(ValueError, match="веса.*архитектур"):
            train_mod._load_hu_checkpoint(restored, path)
        assert load_calls == []
    finally:
        config_mod.load_config("config.yaml")


def test_single_agent_d2_checkpoint_round_trip_and_rejects_action_only_variant(tmp_path):
    d2_config = tmp_path / "d2.yaml"
    d2_config.write_text(
        "num_actions: 6\nnum_players: 2\nhidden_size: 8\nd2cfr_enabled: true\n"
        "d2cfr_mc_correction_enabled: false\nsave_replay_buffers_in_checkpoint: true\n",
        encoding="utf-8",
    )
    action_only_config = tmp_path / "action-only.yaml"
    action_only_config.write_text("num_actions: 6\nnum_players: 2\nhidden_size: 8\n", encoding="utf-8")
    config_mod.load_config(d2_config)
    try:
        source = DeepCFRAgent(player_id=0, num_players=2, device="cpu")
        source.iteration_count = 5
        source.d2cfr_buffer.add(
            np.zeros(source.input_size, dtype=np.float32),
            np.array([0.05, 0.02, 0, 0, 0, 0], dtype=np.float32),
            0.03,
            np.array([0.02, -0.01, 0, 0, 0, 0], dtype=np.float32),
            np.array([1, 1, 0, 0, 0, 0], dtype=np.float32),
            5,
        )
        d2_path = source.save_model(tmp_path / "d2.pt")
        replay_payload = torch.load(d2_path, weights_only=False)["advantage_buffer"]
        assert replay_payload["buffer_type"] == "dueling_advantage"
        assert replay_payload["size"] == 1
        assert replay_payload["total_seen"] == 1
        assert "cur_id" not in replay_payload
        assert "count" not in replay_payload
        restored = DeepCFRAgent(player_id=0, num_players=2, device="cpu")
        restored.load_model(d2_path)
        assert restored.iteration_count == 5
        assert len(restored.d2cfr_buffer) == 1
        state = torch.zeros((1, source.input_size), dtype=torch.float32)
        source_output = source.advantage_net.forward_components(state)
        restored_output = restored.advantage_net.forward_components(state)
        assert torch.equal(restored_output.state_values, source_output.state_values)
        assert torch.equal(restored_output.action_values, source_output.action_values)
        assert torch.equal(restored_output.regrets, source_output.regrets)
        assert np.array_equal(
            restored.d2cfr_buffer._action_values[:1],
            source.d2cfr_buffer._action_values[:1],
        )
        assert np.array_equal(
            restored.d2cfr_buffer._state_values[:1],
            source.d2cfr_buffer._state_values[:1],
        )
        assert np.array_equal(
            restored.d2cfr_buffer._regrets[:1],
            source.d2cfr_buffer._regrets[:1],
        )
    finally:
        config_mod.load_config(action_only_config)

    action_only = DeepCFRAgent(player_id=0, num_players=2, device="cpu")
    action_only_path = action_only.save_model(tmp_path / "action-only.pt")
    before = {key: value.detach().clone() for key, value in source.advantage_net.state_dict().items()}
    config_mod.load_config(d2_config)
    try:
        d2_runtime = DeepCFRAgent(player_id=0, num_players=2, device="cpu")
        before = {key: value.detach().clone() for key, value in d2_runtime.advantage_net.state_dict().items()}
        with pytest.raises(ValueError, match="algorithm variant"):
            d2_runtime.load_model(action_only_path)
        assert all(torch.equal(d2_runtime.advantage_net.state_dict()[key], value) for key, value in before.items())
    finally:
        config_mod.load_config("config.yaml")


def test_d2cfr_resume_rejects_different_replay_capacity(tmp_path):
    source_config = tmp_path / "source.yaml"
    source_config.write_text(
        "num_actions: 6\nnum_players: 2\nhidden_size: 8\nd2cfr_enabled: true\n"
        "d2cfr_mc_correction_enabled: false\nsave_replay_buffers_in_checkpoint: true\n"
        "advantage_memory_size: 4\n",
        encoding="utf-8",
    )
    destination_config = tmp_path / "destination.yaml"
    destination_config.write_text(
        "num_actions: 6\nnum_players: 2\nhidden_size: 8\nd2cfr_enabled: true\n"
        "d2cfr_mc_correction_enabled: false\nsave_replay_buffers_in_checkpoint: true\n"
        "advantage_memory_size: 8\n",
        encoding="utf-8",
    )
    config_mod.load_config(source_config)
    try:
        source = DeepCFRAgent(player_id=0, num_players=2, device="cpu")
        path = source.save_model(tmp_path / "d2.pt")
    finally:
        config_mod.load_config(destination_config)
    try:
        destination = DeepCFRAgent(player_id=0, num_players=2, device="cpu")
        with pytest.raises(ValueError, match="Replay capacity differs"):
            destination.load_model(path)
    finally:
        config_mod.load_config("config.yaml")


def test_single_agent_d2_checkpoint_rejects_incompatible_dueling_weights_before_mutation(
    tmp_path,
):
    config_path = tmp_path / "d2.yaml"
    config_path.write_text(
        "num_actions: 6\nnum_players: 2\nhidden_size: 8\nd2cfr_enabled: true\n"
        "d2cfr_mc_correction_enabled: false\n",
        encoding="utf-8",
    )
    config_mod.load_config(config_path)
    try:
        source = DeepCFRAgent(player_id=0, num_players=2, device="cpu")
        path = source.save_model(tmp_path / "d2.pt")
        payload = torch.load(path, weights_only=True)
        payload["advantage_net"].pop("state_value_head.bias")
        torch.save(payload, path)

        runtime = DeepCFRAgent(player_id=0, num_players=2, device="cpu")
        before = {
            key: value.detach().clone()
            for key, value in runtime.advantage_net.state_dict().items()
        }
        with pytest.raises(ValueError, match="advantage_net имеет несовместимый набор параметров"):
            runtime.load_model(path)
        assert all(
            torch.equal(runtime.advantage_net.state_dict()[key], value)
            for key, value in before.items()
        )
    finally:
        config_mod.load_config("config.yaml")
