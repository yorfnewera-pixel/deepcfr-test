import pytest
import torch

from src.core.deep_cfr import DeepCFRAgent
from src.core.model import (
    CARD_CONTEXT_ARCHITECTURE,
    CARD_FEATURE_SIZE,
    MONOLITHIC_ARCHITECTURE,
    PokerNetwork,
)
from src.utils import config as config_mod


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
