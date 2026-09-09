from __future__ import annotations

import hashlib

import pytest
import torch

from src.core.action_space import ACTION_SPACE_VERSION, NUM_ACTIONS
from src.core.deep_cfr import CHECKPOINT_FORMAT_VERSION, DeepCFRAgent
from src.core.model import CARD_CONTEXT_ARCHITECTURE, CARD_FEATURE_SIZE, PokerNetwork


_HU_CHECKPOINT_KIND = "hu_current_policy_self_play"
_HU_CHECKPOINT_VERSION = 2
_HU_ENCODING_VERSION = "history_summary_v3"


def _hu_card_checkpoint(path):
    """Создаёт минимальный безопасно сериализуемый HU full checkpoint для переноса."""
    source_network = PokerNetwork(
        CARD_FEATURE_SIZE + 3,
        hidden_size=8,
        architecture=CARD_CONTEXT_ARCHITECTURE,
    )
    with torch.no_grad():
        source_network.card_encoder[0].weight.fill_(17.0)
        source_network.card_encoder[0].bias.fill_(-3.0)
        source_network.context_encoder[0].weight.fill_(41.0)
        source_network.action_head.weight.fill_(73.0)
        source_network.action_head.bias.fill_(79.0)

    checkpoint = {
        "checkpoint_format_version": CHECKPOINT_FORMAT_VERSION,
        "checkpoint_kind": _HU_CHECKPOINT_KIND,
        "hu_checkpoint_version": _HU_CHECKPOINT_VERSION,
        "action_space_version": ACTION_SPACE_VERSION,
        "num_actions": NUM_ACTIONS,
        "mode": {
            "hu_current_policy_self_play": True,
            "num_players": 2,
            "num_trainable_players": 2,
            "use_multi_agent_advantage": False,
            "encoding_version": _HU_ENCODING_VERSION,
            "encoder_input_size": CARD_FEATURE_SIZE + 1,
        },
        "architecture": {
            "strategy": {
                "network_architecture": CARD_CONTEXT_ARCHITECTURE,
                "card_feature_size": CARD_FEATURE_SIZE,
                "input_size": CARD_FEATURE_SIZE + 3,
                "hidden_size": 8,
                "num_actions": NUM_ACTIONS,
            }
        },
        "advantage_legs": [
            {
                "network": {},
                "target_network": {},
                "optimizer": {},
                "buffer": {},
            },
            {
                "network": {},
                "target_network": {},
                "optimizer": {},
                "buffer": {},
            },
        ],
        "strategy": {"network": source_network.state_dict(), "optimizer": {}, "buffer": {}},
        "rng": {},
    }
    torch.save(checkpoint, path)
    return source_network


def _six_max_student():
    return DeepCFRAgent(
        player_id=0,
        num_players=6,
        device="cpu",
        hidden_size=8,
        network_architecture=CARD_CONTEXT_ARCHITECTURE,
    )


def test_transfer_copies_only_card_encoder_without_aliasing_or_head_mutation(tmp_path):
    """Ломается, если переносит не только card_encoder либо оставляет общие tensor-ы."""
    checkpoint_path = tmp_path / "hu-card.pt"
    source_network = _hu_card_checkpoint(checkpoint_path)
    student = _six_max_student()
    context_before = {
        name: parameter.detach().clone()
        for name, parameter in student.strategy_net.context_encoder.named_parameters()
    }
    head_before = {
        name: parameter.detach().clone()
        for name, parameter in student.strategy_net.action_head.named_parameters()
    }

    provenance = student.load_card_encoder_from_hu_checkpoint(checkpoint_path)

    assert provenance.mode == "card_encoder_warmstart"
    assert provenance.copied_blocks == ("strategy_net.card_encoder",)
    assert provenance.source_path == checkpoint_path.resolve()
    assert provenance.checksum_sha256 == hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()
    assert provenance.source_architecture == CARD_CONTEXT_ARCHITECTURE
    assert provenance.source_encoding_version == _HU_ENCODING_VERSION
    assert provenance.teacher_num_players == 2
    assert provenance.freeze is False
    assert all(parameter.requires_grad for parameter in student.strategy_net.card_encoder.parameters())
    assert torch.equal(
        student.strategy_net.card_encoder[0].weight,
        source_network.card_encoder[0].weight,
    )
    assert torch.equal(
        student.strategy_net.card_encoder[0].bias,
        source_network.card_encoder[0].bias,
    )
    assert all(
        torch.equal(parameter, context_before[name])
        for name, parameter in student.strategy_net.context_encoder.named_parameters()
    )
    assert all(
        torch.equal(parameter, head_before[name])
        for name, parameter in student.strategy_net.action_head.named_parameters()
    )

    with torch.no_grad():
        student.strategy_net.card_encoder[0].weight.add_(1.0)
    assert not torch.equal(
        student.strategy_net.card_encoder[0].weight,
        source_network.card_encoder[0].weight,
    )


def test_transfer_rejects_non_hu_source_before_student_mutation(tmp_path):
    """Ломается, если light/normal checkpoint меняет student до валидации источника."""
    checkpoint_path = tmp_path / "light.pt"
    _hu_card_checkpoint(checkpoint_path)
    payload = torch.load(checkpoint_path, weights_only=True)
    payload["checkpoint_kind"] = "strategy_only"
    torch.save(payload, checkpoint_path)
    student = _six_max_student()
    before = {
        name: parameter.detach().clone()
        for name, parameter in student.strategy_net.card_encoder.named_parameters()
    }

    with pytest.raises(ValueError, match="полный HU checkpoint"):
        student.load_card_encoder_from_hu_checkpoint(checkpoint_path)

    assert all(
        torch.equal(parameter, before[name])
        for name, parameter in student.strategy_net.card_encoder.named_parameters()
    )


def test_transfer_rejects_incomplete_hu_checkpoint_before_student_mutation(tmp_path):
    """Ломается, если усечённый HU checkpoint принимается как full-источник."""
    checkpoint_path = tmp_path / "incomplete-hu.pt"
    _hu_card_checkpoint(checkpoint_path)
    payload = torch.load(checkpoint_path, weights_only=True)
    payload.pop("rng")
    torch.save(payload, checkpoint_path)
    student = _six_max_student()
    before = {
        name: parameter.detach().clone()
        for name, parameter in student.strategy_net.card_encoder.named_parameters()
    }

    with pytest.raises(ValueError, match="полный набор training state"):
        student.load_card_encoder_from_hu_checkpoint(checkpoint_path)

    assert all(
        torch.equal(parameter, before[name])
        for name, parameter in student.strategy_net.card_encoder.named_parameters()
    )


def test_transfer_can_freeze_copied_card_encoder(tmp_path):
    """Ломается, если явный freeze не распространяется на перенесённые параметры."""
    checkpoint_path = tmp_path / "hu-card.pt"
    _hu_card_checkpoint(checkpoint_path)
    student = _six_max_student()

    provenance = student.load_card_encoder_from_hu_checkpoint(checkpoint_path, freeze=True)

    assert provenance.freeze is True
    assert all(not parameter.requires_grad for parameter in student.strategy_net.card_encoder.parameters())
