from __future__ import annotations

import hashlib
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

import src.core.teacher_transfer as transfer_mod
from src.utils import config as config_mod
from src.core.action_space import NUM_ACTIONS
from src.core.buffers import AdvantageBuffer
from src.core.deep_cfr import DeepCFRAgent
from src.core.hu_self_play import HuStrategyBuffer
from src.core.model import (
    CARD_CONTEXT_ARCHITECTURE,
    CARD_CONTEXT_V2_ARCHITECTURE,
    CARD_FEATURE_SIZE,
    PokerNetwork,
)
from src.training import train as train_mod


_HU_INPUT_SIZE = 181
_HU_STRATEGY_INPUT_SIZE = 183


def _network_with_optimizer(
    input_size: int,
    *,
    architecture: str = CARD_CONTEXT_ARCHITECTURE,
    step_optimizer: bool = True,
):
    network = PokerNetwork(input_size, hidden_size=8, architecture=architecture)
    optimizer = torch.optim.AdamW(network.parameters(), lr=1e-3)
    if step_optimizer:
        network(torch.ones((1, input_size))).sum().backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
    return network, optimizer


def _real_hu_runtime(
    *,
    architecture: str = CARD_CONTEXT_ARCHITECTURE,
    step_optimizers: bool = True,
):
    """Строит компактный, но реальный HU runtime для штатного checkpoint builder."""
    advantage_nets, advantage_targets = [], []
    advantage_optimizers, advantage_buffers = [], []
    for player_id in (0, 1):
        network, optimizer = _network_with_optimizer(
            _HU_INPUT_SIZE,
            architecture=architecture,
            step_optimizer=step_optimizers,
        )
        target, _ = _network_with_optimizer(_HU_INPUT_SIZE, architecture=architecture)
        advantage_nets.append(network)
        advantage_targets.append(target)
        advantage_optimizers.append(optimizer)
        buffer = AdvantageBuffer(3, _HU_INPUT_SIZE)
        buffer.add(
            torch.full((_HU_INPUT_SIZE,), float(player_id)).numpy(),
            torch.full((NUM_ACTIONS,), float(player_id)).numpy(),
            torch.ones(NUM_ACTIONS).numpy(),
            7,
        )
        advantage_buffers.append(buffer)

    strategy_net, strategy_optimizer = _network_with_optimizer(
        _HU_STRATEGY_INPUT_SIZE,
        architecture=architecture,
        step_optimizer=step_optimizers,
    )
    strategy_buffer = HuStrategyBuffer(3, _HU_INPUT_SIZE)
    strategy_buffer.add(
        1,
        torch.full((_HU_INPUT_SIZE,), 3.0).numpy(),
        torch.tensor([0.5, 0.5, 0, 0, 0, 0]).numpy(),
        torch.tensor([1, 1, 0, 0, 0, 0]).numpy(),
        7,
    )
    return SimpleNamespace(
        num_players=2,
        num_trainable_players=2,
        num_actions=NUM_ACTIONS,
        use_multi_agent=False,
        encoding_version="history_summary_v3",
        input_size=_HU_INPUT_SIZE,
        iteration_count=7,
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


def _hu_card_checkpoint(
    path: Path,
    *,
    architecture: str = CARD_CONTEXT_ARCHITECTURE,
    step_optimizers: bool = True,
):
    """Сохраняет checkpoint, созданный штатным HU builder, без ручной подделки схемы."""
    source = _real_hu_runtime(architecture=architecture, step_optimizers=step_optimizers)
    with torch.no_grad():
        source.strategy_net.card_encoder[0].weight.fill_(17.0)
        source.strategy_net.card_encoder[0].bias.fill_(-3.0)
        source.strategy_net.context_encoder[0].weight.fill_(41.0)
        if architecture == CARD_CONTEXT_V2_ARCHITECTURE:
            source.strategy_net.fusion[0].weight.fill_(53.0)
        source.strategy_net.action_head.weight.fill_(73.0)
        source.strategy_net.action_head.bias.fill_(79.0)
    torch.save(train_mod._build_hu_checkpoint(source), path)
    return source.strategy_net


def _hu_d2cfr_card_checkpoint(path: Path, config_path: Path):
    """Сохраняет настоящий D2CFR HU teacher для переноса в six-max."""
    config_path.write_text(
        "\n".join((
            "num_actions: 6",
            "num_players: 2",
            "num_trainable_players: 2",
            "hu_current_policy_self_play: true",
            "hidden_size: 8",
            "d2cfr_enabled: true",
            "d2cfr_mc_correction_enabled: false",
            "advantage_memory_size: 3",
            "strategy_memory_size: 3",
        )) + "\n",
        encoding="utf-8",
    )
    config_mod.load_config(config_path)
    try:
        source = DeepCFRAgent(
            player_id=0,
            num_players=2,
            device="cpu",
            network_architecture=CARD_CONTEXT_ARCHITECTURE,
        )
        train_mod._create_hu_current_policy_coordinator(source)
        with torch.no_grad():
            source.strategy_net.card_encoder[0].weight.fill_(23.0)
            source.strategy_net.card_encoder[0].bias.fill_(-7.0)
        torch.save(train_mod._build_hu_checkpoint(source), path)
        return source.strategy_net
    finally:
        config_mod.load_config("config.yaml")


def _six_max_student(architecture: str = CARD_CONTEXT_ARCHITECTURE):
    return DeepCFRAgent(
        player_id=0,
        num_players=6,
        device="cpu",
        hidden_size=8,
        network_architecture=architecture,
    )


def _card_state(network):
    return {name: parameter.detach().clone() for name, parameter in network.card_encoder.named_parameters()}


def _assert_card_state_unchanged(network, before):
    assert all(torch.equal(parameter, before[name]) for name, parameter in network.card_encoder.named_parameters())


def test_transfer_copies_only_card_encoder_without_aliasing_or_head_mutation(tmp_path):
    """Ломается, если переносит не только card_encoder либо оставляет общие tensor-ы."""
    checkpoint_path = tmp_path / "hu-card.pt"
    source_network = _hu_card_checkpoint(checkpoint_path)
    student = _six_max_student(CARD_CONTEXT_V2_ARCHITECTURE)
    context_before = {name: parameter.detach().clone() for name, parameter in student.strategy_net.context_encoder.named_parameters()}
    head_before = {name: parameter.detach().clone() for name, parameter in student.strategy_net.action_head.named_parameters()}
    fusion_before = {name: parameter.detach().clone() for name, parameter in student.strategy_net.fusion.named_parameters()}

    provenance = student.load_card_encoder_from_hu_checkpoint(checkpoint_path)

    assert provenance.mode == "card_encoder_warmstart"
    assert provenance.copied_blocks == ("strategy_net.card_encoder",)
    assert provenance.source_path == checkpoint_path.resolve()
    assert provenance.checksum_sha256 == hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()
    assert provenance.source_architecture == CARD_CONTEXT_ARCHITECTURE
    assert provenance.source_encoding_version == "history_summary_v3"
    assert provenance.teacher_num_players == 2
    assert provenance.freeze is False
    assert all(parameter.requires_grad for parameter in student.strategy_net.card_encoder.parameters())
    assert torch.equal(student.strategy_net.card_encoder[0].weight, source_network.card_encoder[0].weight)
    assert torch.equal(student.strategy_net.card_encoder[0].bias, source_network.card_encoder[0].bias)
    assert all(torch.equal(parameter, context_before[name]) for name, parameter in student.strategy_net.context_encoder.named_parameters())
    assert all(torch.equal(parameter, head_before[name]) for name, parameter in student.strategy_net.action_head.named_parameters())
    assert all(torch.equal(parameter, fusion_before[name]) for name, parameter in student.strategy_net.fusion.named_parameters())

    with torch.no_grad():
        student.strategy_net.card_encoder[0].weight.add_(1.0)
    assert not torch.equal(student.strategy_net.card_encoder[0].weight, source_network.card_encoder[0].weight)


def test_transfer_accepts_v2_hu_card_encoder_without_copying_fusion_or_heads(tmp_path):
    checkpoint_path = tmp_path / "hu-card-v2.pt"
    source_network = _hu_card_checkpoint(
        checkpoint_path,
        architecture=CARD_CONTEXT_V2_ARCHITECTURE,
    )
    student = _six_max_student(CARD_CONTEXT_V2_ARCHITECTURE)
    context_before = {
        name: parameter.detach().clone()
        for name, parameter in student.strategy_net.context_encoder.named_parameters()
    }
    head_before = {
        name: parameter.detach().clone()
        for name, parameter in student.strategy_net.action_head.named_parameters()
    }
    fusion_before = {
        name: parameter.detach().clone()
        for name, parameter in student.strategy_net.fusion.named_parameters()
    }

    provenance = student.load_card_encoder_from_hu_checkpoint(checkpoint_path)

    assert provenance.source_architecture == CARD_CONTEXT_V2_ARCHITECTURE
    assert torch.equal(student.strategy_net.card_encoder[0].weight, source_network.card_encoder[0].weight)
    assert all(
        torch.equal(parameter, context_before[name])
        for name, parameter in student.strategy_net.context_encoder.named_parameters()
    )
    assert all(
        torch.equal(parameter, head_before[name])
        for name, parameter in student.strategy_net.action_head.named_parameters()
    )
    assert all(
        torch.equal(parameter, fusion_before[name])
        for name, parameter in student.strategy_net.fusion.named_parameters()
    )


def test_transfer_accepts_genuine_hu_checkpoint_with_empty_adamw_state(tmp_path):
    """Ломается, если checkpoint границы итерации без AdamW update отвергается."""
    checkpoint_path = tmp_path / "hu-empty-optimizer-state.pt"
    source_network = _hu_card_checkpoint(checkpoint_path, step_optimizers=False)
    student = _six_max_student()

    student.load_card_encoder_from_hu_checkpoint(checkpoint_path)

    assert torch.equal(student.strategy_net.card_encoder[0].weight, source_network.card_encoder[0].weight)


def test_transfer_accepts_genuine_hu_d2cfr_checkpoint(tmp_path):
    """Ломается, если активный D2CFR HU teacher нельзя перенести в six-max."""
    checkpoint_path = tmp_path / "hu-d2cfr-card.pt"
    source_network = _hu_d2cfr_card_checkpoint(checkpoint_path, tmp_path / "hu-d2cfr.yaml")
    student = _six_max_student()
    context_before = {
        name: parameter.detach().clone()
        for name, parameter in student.strategy_net.context_encoder.named_parameters()
    }

    provenance = student.load_card_encoder_from_hu_checkpoint(checkpoint_path)

    assert provenance.teacher_num_players == 2
    assert torch.equal(student.strategy_net.card_encoder[0].weight, source_network.card_encoder[0].weight)
    assert all(
        torch.equal(parameter, context_before[name])
        for name, parameter in student.strategy_net.context_encoder.named_parameters()
    )


def test_transfer_rejects_partial_valid_adamw_state_before_student_mutation(tmp_path):
    """Ломается, если частичный state AdamW принимается как полный training state."""
    checkpoint_path = tmp_path / "partial-adamw-state.pt"
    _hu_card_checkpoint(checkpoint_path)
    payload = torch.load(checkpoint_path, weights_only=True)
    parameter_state = payload["strategy"]["optimizer"]["state"][0]
    payload["strategy"]["optimizer"]["state"] = {0: parameter_state}
    torch.save(payload, checkpoint_path)
    student = _six_max_student()
    before = _card_state(student.strategy_net)

    with pytest.raises(ValueError, match="повреждённый optimizer"):
        student.load_card_encoder_from_hu_checkpoint(checkpoint_path)

    _assert_card_state_unchanged(student.strategy_net, before)


@pytest.mark.parametrize("rng_key", ["torch_cpu", "torch_cuda"])
def test_transfer_rejects_truncated_torch_rng_state_before_student_mutation(tmp_path, rng_key, monkeypatch):
    """Ломается, если torch RNG ненулевой, но не канонической длины, принимается."""
    checkpoint_path = tmp_path / f"truncated-{rng_key}-rng.pt"
    _hu_card_checkpoint(checkpoint_path)
    payload = torch.load(checkpoint_path, weights_only=True)
    payload["rng"][rng_key] = (
        torch.zeros(1, dtype=torch.uint8)
        if rng_key == "torch_cpu"
        else [torch.zeros(1, dtype=torch.uint8)]
    )
    torch.save(payload, checkpoint_path)
    if rng_key == "torch_cuda":
        monkeypatch.setattr(transfer_mod.torch.cuda, "is_available", lambda: False)
    student = _six_max_student()
    before = _card_state(student.strategy_net)

    with pytest.raises(ValueError, match="повреждённое RNG состояние"):
        student.load_card_encoder_from_hu_checkpoint(checkpoint_path)

    _assert_card_state_unchanged(student.strategy_net, before)


def test_transfer_accepts_canonical_cuda_rng_state_without_local_cuda(tmp_path, monkeypatch):
    """Ломается, если CPU student отвергает 16-байтный CUDA Philox state teacher checkpoint."""
    checkpoint_path = tmp_path / "cuda-rng-without-local-cuda.pt"
    source_network = _hu_card_checkpoint(checkpoint_path)
    payload = torch.load(checkpoint_path, weights_only=True)
    payload["rng"]["torch_cuda"] = [torch.zeros(16, dtype=torch.uint8)]
    torch.save(payload, checkpoint_path)
    monkeypatch.setattr(transfer_mod.torch.cuda, "is_available", lambda: False)
    student = _six_max_student()

    student.load_card_encoder_from_hu_checkpoint(checkpoint_path)

    assert torch.equal(student.strategy_net.card_encoder[0].weight, source_network.card_encoder[0].weight)


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda checkpoint: checkpoint.__setitem__("checkpoint_kind", "strategy_only"), "полный HU checkpoint"),
        (lambda checkpoint: checkpoint.pop("action_labels"), "action_labels"),
        (lambda checkpoint: checkpoint.__setitem__("iteration", -1), "iteration"),
        (lambda checkpoint: checkpoint["config"].__setitem__("encoding_version", "legacy_v2"), "конфигурацию"),
        (lambda checkpoint: checkpoint.__setitem__("update_order", []), "порядок обновления"),
        (lambda checkpoint: checkpoint["strategy"].pop("optimizer"), "полный набор training state"),
        (lambda checkpoint: checkpoint["advantage_legs"][0]["network"].clear(), "advantage-сети"),
        (lambda checkpoint: checkpoint["advantage_legs"][1]["target_network"].clear(), "advantage-сети"),
        (lambda checkpoint: checkpoint["advantage_legs"][0]["optimizer"]["param_groups"].clear(), "optimizer"),
        (
            lambda checkpoint: checkpoint["strategy"]["optimizer"].__setitem__(
                "state",
                {
                    99: {
                        "step": torch.tensor(1.0),
                        "exp_avg": torch.zeros(1),
                        "exp_avg_sq": torch.zeros(1),
                    }
                },
            ),
            "optimizer",
        ),
        (lambda checkpoint: checkpoint["advantage_legs"][0]["buffer"].pop("states"), "replay-буфер"),
        (lambda checkpoint: checkpoint["advantage_legs"][1]["buffer"].__setitem__("regrets", torch.empty(0)), "replay-буфер"),
        (lambda checkpoint: checkpoint["rng"]["numpy"].pop("state"), "RNG"),
        (lambda checkpoint: checkpoint["rng"].__setitem__("python", {}), "RNG"),
        (lambda checkpoint: checkpoint["rng"].__setitem__("python", (3, (), None)), "RNG"),
        (lambda checkpoint: checkpoint["rng"]["numpy"].__setitem__("state", torch.empty(0, dtype=torch.uint32)), "RNG"),
        (lambda checkpoint: checkpoint["rng"].__setitem__("torch_cuda", None), "RNG"),
        (lambda checkpoint: checkpoint.pop("rng"), "RNG"),
        (lambda checkpoint: checkpoint["mode"].__setitem__("encoder_input_size", 110), "режим"),
        (lambda checkpoint: checkpoint["architecture"]["strategy"].__setitem__("input_size", 112), "архитектуру"),
    ],
)
def test_transfer_rejects_each_malformed_hu_schema_before_student_mutation(tmp_path, mutate, message):
    """Ломается, если неполная или несовместимая HU-схема меняет student."""
    checkpoint_path = tmp_path / "invalid-hu.pt"
    _hu_card_checkpoint(checkpoint_path)
    payload = torch.load(checkpoint_path, weights_only=True)
    mutate(payload)
    torch.save(payload, checkpoint_path)
    student = _six_max_student()
    before = _card_state(student.strategy_net)

    with pytest.raises(ValueError, match=message):
        student.load_card_encoder_from_hu_checkpoint(checkpoint_path)

    _assert_card_state_unchanged(student.strategy_net, before)


def test_transfer_rejects_sparse_bias_before_any_card_parameter_mutation(tmp_path):
    """Ломается, если weight уже скопирован к моменту ошибки на разреженном bias."""
    checkpoint_path = tmp_path / "sparse-bias.pt"
    _hu_card_checkpoint(checkpoint_path)
    payload = torch.load(checkpoint_path, weights_only=True)
    with torch.sparse.check_sparse_tensor_invariants():
        payload["strategy"]["network"]["card_encoder.0.bias"] = torch.sparse_coo_tensor(
            torch.tensor([[0]]), torch.tensor([-3.0]), size=(8,)
        )
    torch.save(payload, checkpoint_path)
    student = _six_max_student()
    before = _card_state(student.strategy_net)

    with pytest.raises(ValueError, match="недопустимый tensor"):
        student.load_card_encoder_from_hu_checkpoint(checkpoint_path)

    _assert_card_state_unchanged(student.strategy_net, before)


def test_transfer_rejects_invalid_dtype_before_any_card_parameter_mutation(tmp_path):
    """Ломается, если float64 teacher tensor не отклоняется до копирования weight."""
    checkpoint_path = tmp_path / "float64-weight.pt"
    _hu_card_checkpoint(checkpoint_path)
    payload = torch.load(checkpoint_path, weights_only=True)
    payload["strategy"]["network"]["card_encoder.0.weight"] = (
        payload["strategy"]["network"]["card_encoder.0.weight"].to(torch.float64)
    )
    torch.save(payload, checkpoint_path)
    student = _six_max_student()
    before = _card_state(student.strategy_net)

    with pytest.raises(ValueError, match="недопустимый tensor"):
        student.load_card_encoder_from_hu_checkpoint(checkpoint_path)

    _assert_card_state_unchanged(student.strategy_net, before)


def test_transfer_loads_and_hashes_one_immutable_byte_snapshot(tmp_path, monkeypatch):
    """Ломается, если checksum и десериализация читают файл двумя разными снимками."""
    checkpoint_path = tmp_path / "hu-card.pt"
    _hu_card_checkpoint(checkpoint_path)
    original_read_bytes = Path.read_bytes
    read_count = 0

    def read_bytes_once(path):
        nonlocal read_count
        read_count += 1
        return original_read_bytes(path)

    monkeypatch.setattr(transfer_mod.Path, "read_bytes", read_bytes_once)

    _six_max_student().load_card_encoder_from_hu_checkpoint(checkpoint_path)

    assert read_count == 1


def test_transfer_can_freeze_copied_card_encoder(tmp_path):
    """Ломается, если явный freeze не распространяется на перенесённые параметры."""
    checkpoint_path = tmp_path / "hu-card.pt"
    _hu_card_checkpoint(checkpoint_path)
    student = _six_max_student()

    provenance = student.load_card_encoder_from_hu_checkpoint(checkpoint_path, freeze=True)

    assert provenance.freeze is True
    assert all(not parameter.requires_grad for parameter in student.strategy_net.card_encoder.parameters())


@pytest.mark.parametrize(
    ("lines", "message"),
    [
        (
            ["teacher_transfer_enabled: true", "teacher_transfer_mode: unsupported"],
            "teacher_transfer_mode",
        ),
        (
            ["teacher_transfer_enabled: true"],
            "teacher_transfer_checkpoint",
        ),
        (
            ["teacher_hu_aux_distillation_enabled: true"],
            "Stage B",
        ),
        (
            ["teacher_hu_aux_distillation_weight: 0.1"],
            "Stage B",
        ),
        (
            ["teacher_transfer_auxiliary_enabled: true"],
            "teacher_hu_aux_distillation_enabled",
        ),
        (
            [
                "teacher_transfer_enabled: true",
                "teacher_transfer_checkpoint: hu.pt",
                "teacher_strategy_checkpoint: policy.pt",
            ],
            "teacher_strategy_checkpoint",
        ),
        (
            [
                "teacher_transfer_enabled: true",
                "teacher_transfer_checkpoint: hu.pt",
                "num_players: 2",
            ],
            "six-max",
        ),
        (
            [
                "teacher_transfer_enabled: true",
                "teacher_transfer_checkpoint: hu.pt",
                "hu_current_policy_self_play: true",
                "num_players: 2",
                "num_trainable_players: 2",
            ],
            "hu_current_policy_self_play",
        ),
    ],
)
def test_transfer_configuration_rejects_incompatible_or_unimplemented_modes(tmp_path, lines, message):
    config_path = tmp_path / "config.yaml"
    config_path.write_text("\n".join(["num_actions: 6", *lines]), encoding="utf-8")

    try:
        with pytest.raises(ValueError, match=message):
            config_mod.load_config(config_path)
    finally:
        config_mod.load_config("config.yaml")


@pytest.mark.parametrize(
    "kwargs",
    [
        {"teacher_hu_aux_distillation_enabled": True},
        {"teacher_hu_aux_distillation_weight": 0.25},
        {
            "teacher_hu_aux_distillation_enabled": True,
            "teacher_hu_aux_distillation_weight": 0.25,
        },
    ],
)
def test_programmatic_stage_b_options_fail_before_agent_creation(monkeypatch, kwargs):
    monkeypatch.setattr(
        train_mod,
        "DeepCFRAgent",
        lambda **_kwargs: pytest.fail("Stage B должен быть отклонён до создания agent"),
    )

    with pytest.raises(ValueError, match="Stage B"):
        train_mod.train_self_play_multi(num_iterations=0, **kwargs)


def test_programmatic_legacy_auxiliary_option_is_rejected_explicitly():
    with pytest.raises(ValueError, match="teacher_hu_aux_distillation_enabled"):
        train_mod.train_self_play_multi(
            num_iterations=0,
            teacher_transfer_auxiliary_enabled=True,
            teacher_transfer_auxiliary_weight=0.25,
        )
