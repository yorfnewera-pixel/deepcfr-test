"""Контракты полного checkpoint/resume для HU current-policy self-play."""
from __future__ import annotations

import random
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from src.core.buffers import AdvantageBuffer
from src.core.hu_self_play import HuStrategyBuffer
from src.core.model import (
    CARD_CONTEXT_ARCHITECTURE,
    CARD_CONTEXT_ARCHITECTURES,
    CARD_CONTEXT_V2_ARCHITECTURE,
    CARD_FEATURE_SIZE,
    MONOLITHIC_ARCHITECTURE,
    PokerNetwork,
)
from src.training import train as train_mod


class _UnsupportedPayload:
    pass


def _network_with_optimizer(input_size: int, architecture: str = MONOLITHIC_ARCHITECTURE):
    network = PokerNetwork(input_size, hidden_size=8, architecture=architecture)
    optimizer = torch.optim.AdamW(network.parameters(), lr=1e-3)
    loss = network(torch.ones((1, input_size))).sum()
    loss.backward()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    return network, optimizer


def _hu_runtime(
    iteration: int = 7,
    network_architecture: str = MONOLITHIC_ARCHITECTURE,
):
    input_size = CARD_FEATURE_SIZE + 1 if network_architecture in CARD_CONTEXT_ARCHITECTURES else 4
    advantage_nets, advantage_targets, advantage_optimizers, advantage_buffers = [], [], [], []
    for player_id in (0, 1):
        network, optimizer = _network_with_optimizer(input_size, network_architecture)
        target, _ = _network_with_optimizer(input_size, network_architecture)
        advantage_nets.append(network)
        advantage_targets.append(target)
        advantage_optimizers.append(optimizer)
        buffer = AdvantageBuffer(3, input_size)
        buffer.add(np.full(input_size, player_id, dtype=np.float32), np.full(6, player_id, dtype=np.float32), np.ones(6, dtype=np.float32), iteration)
        advantage_buffers.append(buffer)
    strategy_net, strategy_optimizer = _network_with_optimizer(
        input_size + 2,
        network_architecture,
    )
    strategy_buffer = HuStrategyBuffer(3, input_size)
    strategy_buffer.add(1, np.full(input_size, 3, dtype=np.float32), np.array([0.5, 0.5, 0, 0, 0, 0], dtype=np.float32), np.array([1, 1, 0, 0, 0, 0], dtype=np.float32), iteration)
    agent = SimpleNamespace(
        num_players=2,
        num_trainable_players=2,
        num_actions=6,
        use_multi_agent=False,
        encoding_version="history_summary_v3",
        input_size=input_size,
        iteration_count=iteration,
        hu_current_policy_self_play=True,
        hu_advantage_nets=tuple(advantage_nets),
        hu_advantage_target_nets=tuple(advantage_targets),
        hu_advantage_optimizers=tuple(advantage_optimizers),
        hu_advantage_buffers=tuple(advantage_buffers),
        strategy_net=strategy_net,
        strategy_optimizer=strategy_optimizer,
        hu_strategy_buffer=strategy_buffer,
        advantage_accumulation="dcfr_plus",
        discount_alpha=2.0,
        discount_gamma=1.0,
        advantage_regret_norm="none",
        advantage_regret_clip=None,
        advantage_reward_scale=1.0,
        advantage_loss="mse",
        advantage_huber_delta=1.0,
        advantage_batch_size=4,
        strategy_batch_size=4,
        advantage_epochs=1,
        strategy_epochs=1,
        advantage_train_steps=None,
        strategy_train_steps=None,
        advantage_buffer_reservoir=False,
        clear_strategy_buffer_each_iteration=False,
        strategy_distillation_lambda=0.0,
        strategy_distillation_temperature=1.0,
        strategy_distillation_anneal_iterations=0,
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
    assert checkpoint["hu_checkpoint_version"] == 3
    assert checkpoint["game_rules_version"] == "holdem_standard_hu_v2"
    assert checkpoint["update_order"] == [
        "traverse_p0", "traverse_p1", "train_advantage_p0", "train_advantage_p1", "train_strategy"
    ]
    for leg in checkpoint["advantage_legs"]:
        assert leg["buffer"]["buffer_type"] == "advantage"
        assert leg["buffer"]["size"] == 1
        assert leg["buffer"]["total_seen"] == 1
    assert checkpoint["strategy"]["buffer"]["buffer_type"] == "hu_strategy"
    assert checkpoint["strategy"]["buffer"]["size"] == 1
    assert checkpoint["strategy"]["buffer"]["total_seen"] == 1
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


def test_hu_resume_accepts_legacy_monolithic_architecture_without_metadata(tmp_path):
    checkpoint = train_mod._build_hu_checkpoint(_hu_runtime())
    for network_schema in (
        *checkpoint["architecture"]["advantage"],
        *checkpoint["architecture"]["advantage_target"],
        checkpoint["architecture"]["strategy"],
    ):
        network_schema.pop("network_architecture")
    path = tmp_path / "legacy-monolithic-hu.pt"
    torch.save(checkpoint, path)

    restored = _hu_runtime(iteration=0)
    assert train_mod._load_hu_checkpoint(restored, path)["iteration"] == 7


def test_hu_resume_rejects_explicit_null_monolithic_architecture_metadata(tmp_path):
    checkpoint = train_mod._build_hu_checkpoint(_hu_runtime())
    checkpoint["architecture"]["strategy"]["network_architecture"] = None
    path = tmp_path / "null-monolithic-hu.pt"
    torch.save(checkpoint, path)

    with pytest.raises(ValueError, match="некорректное значение архитектуры"):
        train_mod._load_hu_checkpoint(_hu_runtime(iteration=0), path)


def test_hu_card_context_checkpoint_round_trip(tmp_path):
    source = _hu_runtime(network_architecture=CARD_CONTEXT_ARCHITECTURE)
    checkpoint = train_mod._build_hu_checkpoint(source)
    path = tmp_path / "card-context-hu.pt"
    train_mod._save_hu_checkpoint(source, path)

    assert checkpoint["architecture"]["strategy"]["network_architecture"] == CARD_CONTEXT_ARCHITECTURE
    assert checkpoint["architecture"]["strategy"]["card_feature_size"] == CARD_FEATURE_SIZE
    restored = _hu_runtime(iteration=0, network_architecture=CARD_CONTEXT_ARCHITECTURE)
    assert train_mod._load_hu_checkpoint(restored, path)["iteration"] == 7


def test_hu_card_context_v2_checkpoint_round_trip_restores_logits_and_fusion_metadata(tmp_path):
    source = _hu_runtime(network_architecture=CARD_CONTEXT_V2_ARCHITECTURE)
    advantage_input = torch.randn(2, source.input_size)
    strategy_input = torch.randn(2, source.input_size + 2)
    expected_advantage = source.hu_advantage_nets[0](advantage_input).detach().clone()
    expected_target = source.hu_advantage_target_nets[1](advantage_input).detach().clone()
    expected_strategy = source.strategy_net(strategy_input).detach().clone()
    path = tmp_path / "card-context-v2-hu.pt"

    checkpoint = train_mod._build_hu_checkpoint(source)
    train_mod._save_hu_checkpoint(source, path)

    expected_strategy_schema = {
        "network_architecture": CARD_CONTEXT_V2_ARCHITECTURE,
        "card_feature_size": CARD_FEATURE_SIZE,
        "input_size": source.input_size + 2,
        "hidden_size": 8,
        "fusion_input_size": 16,
        "fusion_output_size": 8,
        "num_actions": 6,
    }
    assert checkpoint["architecture"]["strategy"] == expected_strategy_schema
    restored = _hu_runtime(iteration=0, network_architecture=CARD_CONTEXT_V2_ARCHITECTURE)
    assert train_mod._load_hu_checkpoint(restored, path)["iteration"] == 7
    assert torch.equal(restored.hu_advantage_nets[0](advantage_input), expected_advantage)
    assert torch.equal(restored.hu_advantage_target_nets[1](advantage_input), expected_target)
    assert torch.equal(restored.strategy_net(strategy_input), expected_strategy)


@pytest.mark.parametrize(
    ("source_architecture", "target_architecture"),
    (
        (CARD_CONTEXT_ARCHITECTURE, CARD_CONTEXT_V2_ARCHITECTURE),
        (CARD_CONTEXT_V2_ARCHITECTURE, CARD_CONTEXT_ARCHITECTURE),
    ),
)
def test_hu_resume_rejects_card_context_version_mismatch(
    tmp_path,
    source_architecture,
    target_architecture,
):
    checkpoint = train_mod._build_hu_checkpoint(
        _hu_runtime(network_architecture=source_architecture)
    )
    path = tmp_path / "version-mismatch.pt"
    torch.save(checkpoint, path)

    with pytest.raises(
        ValueError,
        match=f"{source_architecture}.*{target_architecture}|{target_architecture}.*{source_architecture}",
    ):
        train_mod._load_hu_checkpoint(
            _hu_runtime(iteration=0, network_architecture=target_architecture),
            path,
        )


def test_hu_resume_rejects_card_context_without_architecture_metadata(tmp_path):
    checkpoint = train_mod._build_hu_checkpoint(
        _hu_runtime(network_architecture=CARD_CONTEXT_ARCHITECTURE)
    )
    for network_schema in (
        *checkpoint["architecture"]["advantage"],
        *checkpoint["architecture"]["advantage_target"],
        checkpoint["architecture"]["strategy"],
    ):
        network_schema.pop("network_architecture")
    path = tmp_path / "metadata-less-card-context-hu.pt"
    torch.save(checkpoint, path)

    with pytest.raises(ValueError, match="метаданные архитектуры"):
        train_mod._load_hu_checkpoint(
            _hu_runtime(iteration=0, network_architecture=CARD_CONTEXT_ARCHITECTURE),
            path,
        )


def test_hu_resume_rejects_card_context_with_mismatched_card_feature_size(tmp_path):
    checkpoint = train_mod._build_hu_checkpoint(
        _hu_runtime(network_architecture=CARD_CONTEXT_ARCHITECTURE)
    )
    checkpoint["architecture"]["strategy"]["card_feature_size"] = CARD_FEATURE_SIZE + 1
    path = tmp_path / "mismatched-card-context-hu.pt"
    torch.save(checkpoint, path)

    with pytest.raises(ValueError, match="размер card-признаков"):
        train_mod._load_hu_checkpoint(
            _hu_runtime(iteration=0, network_architecture=CARD_CONTEXT_ARCHITECTURE),
            path,
        )


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


def test_hu_resume_rejects_checkpoint_without_current_game_rules_version(tmp_path):
    checkpoint = train_mod._build_hu_checkpoint(_hu_runtime())
    checkpoint.pop("game_rules_version")
    path = tmp_path / "pre-hu-rules.pt"
    torch.save(checkpoint, path)

    with pytest.raises(ValueError, match="исправления правил"):
        train_mod._load_hu_checkpoint(_hu_runtime(), path)


def test_hu_resume_loads_only_weights_only_checkpoint_and_rejects_unsupported_payload(tmp_path, monkeypatch):
    source = _hu_runtime()
    valid_path = tmp_path / "valid.pt"
    train_mod._save_hu_checkpoint(source, valid_path)
    original_load = train_mod.torch.load
    options = []

    def safe_load(*args, **kwargs):
        options.append(kwargs.get("weights_only"))
        return original_load(*args, **kwargs)

    monkeypatch.setattr(train_mod.torch, "load", safe_load)
    train_mod._load_hu_checkpoint(_hu_runtime(), valid_path)
    assert options == [True]

    unsafe_path = tmp_path / "unsafe.pt"
    torch.save({"checkpoint_kind": "hu_current_policy_self_play", "payload": _UnsupportedPayload()}, unsafe_path)
    with pytest.raises(ValueError, match="безопасно"):
        train_mod._load_hu_checkpoint(_hu_runtime(), unsafe_path)


@pytest.mark.parametrize(
    "mutate, message",
    [
        (lambda checkpoint: checkpoint["config"].__setitem__("discount_alpha", 9.0), "конфигурацию"),
        (lambda checkpoint: checkpoint.pop("iteration"), "iteration"),
        (lambda checkpoint: checkpoint.__setitem__("iteration", -1), "iteration"),
        (lambda checkpoint: checkpoint.__setitem__("iteration", "seven"), "iteration"),
        (lambda checkpoint: checkpoint.__setitem__("action_labels", ["bad"] * 6), "action_labels"),
    ],
)
def test_hu_resume_rejects_config_and_schema_contract_mismatches(tmp_path, mutate, message):
    path = tmp_path / "broken.pt"
    checkpoint = train_mod._build_hu_checkpoint(_hu_runtime())
    mutate(checkpoint)
    torch.save(checkpoint, path)

    with pytest.raises(ValueError, match=message):
        train_mod._load_hu_checkpoint(_hu_runtime(), path)


def test_hu_resume_allows_explicit_runtime_only_configuration(tmp_path):
    path = tmp_path / "runtime-only.pt"
    checkpoint = train_mod._build_hu_checkpoint(_hu_runtime())
    checkpoint["config"]["save_dir"] = "another-machine-output"
    torch.save(checkpoint, path)

    assert train_mod._load_hu_checkpoint(_hu_runtime(), path)["iteration"] == 7
