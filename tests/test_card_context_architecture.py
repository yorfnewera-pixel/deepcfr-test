import pytest
import torch

from src.core.model import (
    CARD_CONTEXT_ARCHITECTURE,
    CARD_FEATURE_SIZE,
    MONOLITHIC_ARCHITECTURE,
    PokerNetwork,
)


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
