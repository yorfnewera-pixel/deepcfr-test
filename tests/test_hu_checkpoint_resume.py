"""Контракты полного checkpoint/resume для HU current-policy self-play."""
from __future__ import annotations

import random
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from src.core.buffers import AdvantageBuffer
from src.core.hu_self_play import HuStrategyBuffer
from src.core.model import PokerNetwork
from src.training import train as train_mod


def _network_with_optimizer(input_size: int):
    network = PokerNetwork(input_size, hidden_size=8)
    optimizer = torch.optim.AdamW(network.parameters(), lr=1e-3)
    loss = network(torch.ones((1, input_size))).sum()
    loss.backward()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    return network, optimizer


def _hu_runtime(iteration: int = 7):
    advantage_nets, advantage_targets, advantage_optimizers, advantage_buffers = [], [], [], []
    for player_id in (0, 1):
        network, optimizer = _network_with_optimizer(4)
        target, _ = _network_with_optimizer(4)
        advantage_nets.append(network)
        advantage_targets.append(target)
        advantage_optimizers.append(optimizer)
        buffer = AdvantageBuffer(3, 4)
        buffer.add(np.full(4, player_id, dtype=np.float32), np.full(6, player_id, dtype=np.float32), np.ones(6, dtype=np.float32), iteration)
        advantage_buffers.append(buffer)
    strategy_net, strategy_optimizer = _network_with_optimizer(6)
    strategy_buffer = HuStrategyBuffer(3, 4)
    strategy_buffer.add(1, np.full(4, 3, dtype=np.float32), np.array([0.5, 0.5, 0, 0, 0, 0], dtype=np.float32), np.array([1, 1, 0, 0, 0, 0], dtype=np.float32), iteration)
    agent = SimpleNamespace(
        num_players=2,
        num_trainable_players=2,
        num_actions=6,
        use_multi_agent=False,
        encoding_version="history_summary_v3",
        input_size=4,
        iteration_count=iteration,
        hu_current_policy_self_play=True,
        hu_advantage_nets=tuple(advantage_nets),
        hu_advantage_target_nets=tuple(advantage_targets),
        hu_advantage_optimizers=tuple(advantage_optimizers),
        hu_advantage_buffers=tuple(advantage_buffers),
        strategy_net=strategy_net,
        strategy_optimizer=strategy_optimizer,
        hu_strategy_buffer=strategy_buffer,
    )
    return agent


def _state_dict_copy(network):
    return {key: value.detach().clone() for key, value in network.state_dict().items()}


def test_hu_full_checkpoint_round_trip_restores_all_training_state_and_rng(tmp_path):
    random.seed(91)
    np.random.seed(91)
    torch.manual_seed(91)
    source = _hu_runtime()
    expected_advantage = [_state_dict_copy(network) for network in source.hu_advantage_nets]
    expected_target = [_state_dict_copy(network) for network in source.hu_advantage_target_nets]
    expected_strategy = _state_dict_copy(source.strategy_net)

    path = tmp_path / "hu.pt"
    train_mod._save_hu_checkpoint(source, path, seed=91)
    expected_random = (random.random(), float(np.random.random()), torch.rand(1))

    restored = _hu_runtime(iteration=0)
    for network in (*restored.hu_advantage_nets, *restored.hu_advantage_target_nets, restored.strategy_net):
        for parameter in network.parameters():
            parameter.data.zero_()
    train_mod._load_hu_checkpoint(restored, path)

    assert restored.iteration_count == 7
    checkpoint = train_mod._build_hu_checkpoint(source, seed=91)
    assert checkpoint["config"]["hu_current_policy_self_play"] is True
    assert checkpoint["update_order"] == [
        "traverse_p0", "traverse_p1", "train_advantage_p0", "train_advantage_p1", "train_strategy"
    ]
    for expected, network in zip(expected_advantage, restored.hu_advantage_nets, strict=True):
        assert all(torch.equal(value, network.state_dict()[key]) for key, value in expected.items())
    for expected, network in zip(expected_target, restored.hu_advantage_target_nets, strict=True):
        assert all(torch.equal(value, network.state_dict()[key]) for key, value in expected.items())
    assert all(torch.equal(value, restored.strategy_net.state_dict()[key]) for key, value in expected_strategy.items())
    assert all(optimizer.state for optimizer in restored.hu_advantage_optimizers)
    assert restored.strategy_optimizer.state
    assert np.array_equal(restored.hu_advantage_buffers[1]._regrets[0], np.ones(6, dtype=np.float32))
    assert restored.hu_strategy_buffer.actor_ids().tolist() == [1]
    assert np.array_equal(restored.hu_strategy_buffer._policies[0], np.array([0.5, 0.5, 0, 0, 0, 0], dtype=np.float32))
    assert random.random() == expected_random[0]
    assert float(np.random.random()) == expected_random[1]
    assert torch.equal(torch.rand(1), expected_random[2])


@pytest.mark.parametrize(
    "payload, message",
    [
        ({"checkpoint_format_version": 6}, "HU checkpoint"),
        ({"checkpoint_kind": "hu_current_policy_self_play"}, "версию"),
        ({"checkpoint_kind": "hu_current_policy_self_play", "hu_checkpoint_version": 999}, "версию"),
    ],
)
def test_hu_resume_rejects_legacy_or_incompatible_checkpoint(tmp_path, payload, message):
    path = tmp_path / "incompatible.pt"
    torch.save(payload, path)

    with pytest.raises(ValueError, match=message):
        train_mod._load_hu_checkpoint(_hu_runtime(), path)
