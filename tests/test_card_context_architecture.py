from pathlib import Path

import pytest
import torch

from src.core.deep_cfr import DeepCFRAgent
from src.core.model import (
    CARD_CONTEXT_ARCHITECTURE,
    CARD_CONTEXT_V2_ARCHITECTURE,
    CARD_FEATURE_SIZE,
    DuelingRegretNetwork,
    MONOLITHIC_ARCHITECTURE,
    PokerNetwork,
)
from policy_runtime.core import PokerNetwork as RuntimePokerNetwork
from src.utils import config as config_mod


def test_hu_smoke_baseline_creates_card_context_agent():
    smoke_config = Path(__file__).parents[1] / "configs" / "hu_smoke_baseline.yaml"

    try:
        config_mod.load_config(smoke_config)
        agent = DeepCFRAgent(player_id=0, num_players=2, device="cpu")

        assert agent.network_architecture == CARD_CONTEXT_ARCHITECTURE
    finally:
        config_mod.load_config("config.yaml")


def test_card_encoder_ignores_context_features():
    network = PokerNetwork(181, hidden_size=16, architecture="card_context_v1")
    first = torch.zeros(1, 181)
    second = first.clone()
    second[:, 109:] = 1.0

    assert torch.equal(network.encode_cards(first), network.encode_cards(second))


def test_card_context_architecture_validates_card_feature_boundary():
    with pytest.raises(ValueError, match="109"):
        PokerNetwork(CARD_FEATURE_SIZE - 1, architecture=CARD_CONTEXT_ARCHITECTURE)


def test_monolithic_architecture_remains_default_and_uses_legacy_keys():
    network = PokerNetwork(181, hidden_size=16)

    assert network.architecture == MONOLITHIC_ARCHITECTURE
    assert set(network.state_dict()) == {
        "base.0.weight",
        "base.0.bias",
        "base.2.weight",
        "base.2.bias",
        "base.4.weight",
        "base.4.bias",
        "action_head.weight",
        "action_head.bias",
    }


@pytest.mark.parametrize("network_class", (PokerNetwork, RuntimePokerNetwork))
def test_card_context_v1_keeps_direct_head_input(network_class):
    network = network_class(181, hidden_size=16, architecture=CARD_CONTEXT_ARCHITECTURE)

    assert network.action_head.in_features == 32
    assert not hasattr(network, "fusion")
    assert set(network.state_dict()) == {
        "card_encoder.0.weight",
        "card_encoder.0.bias",
        "context_encoder.0.weight",
        "context_encoder.0.bias",
        "action_head.weight",
        "action_head.bias",
    }


@pytest.mark.parametrize("network_class", (PokerNetwork, RuntimePokerNetwork))
def test_card_context_v2_fuses_embeddings_before_action_head(network_class):
    network = network_class(181, hidden_size=16, architecture=CARD_CONTEXT_V2_ARCHITECTURE)
    logits = network(torch.randn(3, 181))

    assert network.fusion[0].in_features == 32
    assert network.fusion[0].out_features == 16
    assert network.fusion[2].in_features == 16
    assert network.fusion[2].out_features == 16
    assert network.action_head.in_features == 16
    assert logits.shape == (3, 6)
    assert torch.equal(network.action_head.weight, torch.zeros_like(network.action_head.weight))
    assert torch.equal(network.action_head.bias, torch.zeros_like(network.action_head.bias))


def test_dueling_card_context_v2_preserves_exact_action_value_minus_state_value():
    network = DuelingRegretNetwork(181, hidden_size=16, architecture=CARD_CONTEXT_V2_ARCHITECTURE)
    result = network.forward_components(torch.randn(3, 181))

    assert network.fusion[0].in_features == 32
    assert network.fusion[0].out_features == 16
    assert network.fusion[2].in_features == 16
    assert network.fusion[2].out_features == 16
    assert network.state_value_head.in_features == 16
    assert network.action_value_head.in_features == 16
    assert result.state_values.shape == (3, 1)
    assert result.action_values.shape == (3, 6)
    assert result.regrets.shape == (3, 6)
    assert torch.equal(result.regrets, result.action_values - result.state_values)
    assert torch.equal(network.state_value_head.weight, torch.zeros_like(network.state_value_head.weight))
    assert torch.equal(network.action_value_head.weight, torch.zeros_like(network.action_value_head.weight))


def test_card_context_v2_represents_binary_card_context_interaction():
    inputs = torch.zeros(4, 181)
    inputs[1, 109] = 1.0
    inputs[2, 0] = 1.0
    inputs[3, 0] = 1.0
    inputs[3, 109] = 1.0
    v1 = PokerNetwork(181, hidden_size=1, architecture=CARD_CONTEXT_ARCHITECTURE)
    v2 = PokerNetwork(181, hidden_size=1, architecture=CARD_CONTEXT_V2_ARCHITECTURE)

    with torch.no_grad():
        for network in (v1, v2):
            network.card_encoder[0].weight.zero_()
            network.card_encoder[0].bias.zero_()
            network.card_encoder[0].weight[0, 0] = 1.0
            network.context_encoder[0].weight.zero_()
            network.context_encoder[0].bias.zero_()
            network.context_encoder[0].weight[0, 0] = 1.0

        v1.action_head.weight.zero_()
        v1.action_head.bias.zero_()
        v1.action_head.weight[0, 0] = 1.0
        v1.action_head.weight[0, 1] = 1.0

        v2.fusion[0].weight.fill_(1.0)
        v2.fusion[0].bias.fill_(-1.0)
        v2.fusion[2].weight.fill_(1.0)
        v2.fusion[2].bias.zero_()
        v2.action_head.weight.zero_()
        v2.action_head.bias.zero_()
        v2.action_head.weight[0, 0] = 1.0

    v1_outputs = v1(inputs)[:, 0]
    v2_outputs = v2(inputs)[:, 0]

    assert torch.equal(v1_outputs, torch.tensor((0.0, 1.0, 1.0, 2.0)))
    assert torch.equal(v2_outputs, torch.tensor((0.0, 0.0, 0.0, 1.0)))
    assert v1_outputs[3] - v1_outputs[2] - v1_outputs[1] + v1_outputs[0] == 0.0
    assert v2_outputs[3] - v2_outputs[2] - v2_outputs[1] + v2_outputs[0] == 1.0


@pytest.mark.parametrize("network_class", (PokerNetwork, RuntimePokerNetwork))
@pytest.mark.parametrize("architecture", (CARD_CONTEXT_ARCHITECTURE, CARD_CONTEXT_V2_ARCHITECTURE))
def test_card_context_encoders_are_available_for_both_card_context_architectures(
    network_class, architecture
):
    network = network_class(181, hidden_size=16, architecture=architecture)
    state = torch.randn(3, 181)

    assert network.encode_cards(state).shape == (3, 16)
    assert network.encode_context(state).shape == (3, 16)


@pytest.mark.parametrize("network_class", (PokerNetwork, RuntimePokerNetwork))
def test_monolithic_networks_keep_forward_contract_and_reject_card_context_encoders(network_class):
    network = network_class(181, hidden_size=16, architecture=MONOLITHIC_ARCHITECTURE)

    assert network(torch.randn(3, 181)).shape == (3, 6)
    with pytest.raises(ValueError, match="Кодировщик карт"):
        network.encode_cards(torch.randn(1, 181))
    with pytest.raises(ValueError, match="Контекстный кодировщик"):
        network.encode_context(torch.randn(1, 181))


def test_monolithic_agent_rejects_card_context_checkpoint(tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text("num_actions: 6\nhidden_size: 8\n", encoding="utf-8")
    checkpoint_path = tmp_path / "card-context.pt"

    try:
        config_mod.load_config(config_path)
        card_context_agent = DeepCFRAgent(
            player_id=0,
            num_players=2,
            network_architecture=CARD_CONTEXT_ARCHITECTURE,
        )
        card_context_agent.save_model(str(checkpoint_path))
        checkpoint = torch.load(checkpoint_path, weights_only=False)
        assert checkpoint["network_architecture"] == CARD_CONTEXT_ARCHITECTURE
        assert checkpoint["card_feature_size"] == CARD_FEATURE_SIZE
        light_checkpoint = card_context_agent.build_light_checkpoint()
        assert light_checkpoint["network_architecture"] == CARD_CONTEXT_ARCHITECTURE
        assert light_checkpoint["card_feature_size"] == CARD_FEATURE_SIZE

        monolithic_agent = DeepCFRAgent(
            player_id=0,
            num_players=2,
            network_architecture=MONOLITHIC_ARCHITECTURE,
        )
        with pytest.raises(ValueError, match="архитектур"):
            monolithic_agent.load_model(str(checkpoint_path))
    finally:
        config_mod.load_config("config.yaml")


def test_legacy_checkpoint_without_metadata_is_supported_only_by_monolithic_agent(tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text("num_actions: 6\nhidden_size: 8\n", encoding="utf-8")

    try:
        config_mod.load_config(config_path)
        monolithic_source = DeepCFRAgent(
            player_id=0,
            num_players=2,
            network_architecture=MONOLITHIC_ARCHITECTURE,
        )
        legacy_checkpoint = monolithic_source._build_checkpoint()
        legacy_checkpoint.pop("network_architecture", None)
        legacy_checkpoint.pop("card_feature_size", None)
        legacy_checkpoint["config"].pop("network_architecture", None)
        legacy_checkpoint["config"].pop("card_feature_size", None)
        legacy_path = tmp_path / "legacy-monolithic.pt"
        torch.save(legacy_checkpoint, legacy_path)

        monolithic_target = DeepCFRAgent(
            player_id=0,
            num_players=2,
            network_architecture=MONOLITHIC_ARCHITECTURE,
        )
        assert monolithic_target.load_model(str(legacy_path))["iteration"] == 0

        card_context_source = DeepCFRAgent(
            player_id=0,
            num_players=2,
            network_architecture=CARD_CONTEXT_ARCHITECTURE,
        )
        card_context_checkpoint = card_context_source._build_checkpoint()
        card_context_checkpoint.pop("network_architecture", None)
        card_context_checkpoint.pop("card_feature_size", None)
        card_context_checkpoint["config"].pop("network_architecture", None)
        card_context_checkpoint["config"].pop("card_feature_size", None)
        card_context_path = tmp_path / "legacy-card-context.pt"
        torch.save(card_context_checkpoint, card_context_path)

        card_context_target = DeepCFRAgent(
            player_id=0,
            num_players=2,
            network_architecture=CARD_CONTEXT_ARCHITECTURE,
        )
        with pytest.raises(ValueError, match="метаданн"):
            card_context_target.load_model(str(card_context_path))
    finally:
        config_mod.load_config("config.yaml")


def test_sixmax_rejects_card_context_heads_up_teacher_before_legacy_projection(tmp_path):
    checkpoint_path = tmp_path / "hu-card-context-teacher.pt"
    teacher = DeepCFRAgent(
        player_id=0,
        num_players=2,
        network_architecture=CARD_CONTEXT_ARCHITECTURE,
    )
    torch.save(teacher.build_light_checkpoint(), checkpoint_path)
    student = DeepCFRAgent(player_id=0, num_players=6)

    with pytest.raises(ValueError, match="HU projection.*card_context_v1"):
        student.load_teacher_strategy_checkpoint(checkpoint_path)


def test_monolithic_agent_rejects_explicit_null_architecture_metadata(tmp_path):
    checkpoint_path = tmp_path / "null-architecture.pt"
    source = DeepCFRAgent(
        player_id=0,
        num_players=2,
        network_architecture=MONOLITHIC_ARCHITECTURE,
    )
    payload = source._build_checkpoint()
    payload["network_architecture"] = None
    torch.save(payload, checkpoint_path)
    target = DeepCFRAgent(
        player_id=0,
        num_players=2,
        network_architecture=MONOLITHIC_ARCHITECTURE,
    )

    with pytest.raises(ValueError, match="некорректное значение архитектуры"):
        target.load_model(str(checkpoint_path))


def test_config_rejects_unsupported_network_architecture(tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "num_actions: 6\nnetwork_architecture: unsupported_v1\n",
        encoding="utf-8",
    )

    try:
        with pytest.raises(ValueError, match="network_architecture"):
            config_mod.load_config(config_path)
    finally:
        config_mod.load_config("config.yaml")
