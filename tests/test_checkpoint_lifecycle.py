from pathlib import Path
from typing import cast

import pytest
import torch

from src.core.deep_cfr import DeepCFRAgent
from src.core.teacher_transfer import CardEncoderWarmstartProvenance
from src.core.model import (
    CARD_CONTEXT_ARCHITECTURE,
    CARD_FEATURE_SIZE,
    MONOLITHIC_ARCHITECTURE,
    PokerNetwork,
)
from src.training import train as train_mod
from src.utils import config as config_mod


def _load_transfer_config(path, extra_lines):
    path.write_text(
        "\n".join(["num_actions: 6", *extra_lines]),
        encoding="utf-8",
    )
    config_mod.load_config(path)


class TinyAgent:
    def __init__(self):
        self.iteration_count = 0

    def _build_checkpoint(self, seed=None):
        return {"iteration": self.iteration_count, "seed": seed}


class LightCheckpointAgent(TinyAgent):
    def build_light_checkpoint(self, seed=None):
        return {"iteration": self.iteration_count, "seed": seed, "strategy_net": {}}


def test_hu_architecture_metadata_describes_card_context_network():
    network = PokerNetwork(
        CARD_FEATURE_SIZE + 2,
        hidden_size=8,
        architecture=CARD_CONTEXT_ARCHITECTURE,
    )

    assert train_mod._network_architecture(network) == {
        "network_architecture": CARD_CONTEXT_ARCHITECTURE,
        "card_feature_size": CARD_FEATURE_SIZE,
        "input_size": CARD_FEATURE_SIZE + 2,
        "hidden_size": 8,
        "num_actions": 6,
    }


def test_opponent_strategy_loader_rejects_incompatible_architecture_before_weights(tmp_path):
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
        monolithic_agent = DeepCFRAgent(
            player_id=0,
            num_players=2,
            network_architecture=MONOLITHIC_ARCHITECTURE,
        )

        with pytest.raises(ValueError, match="архитектур"):
            train_mod._load_full_checkpoint_strategy_state(checkpoint_path, monolithic_agent)
    finally:
        config_mod.load_config("config.yaml")


@pytest.mark.parametrize(
    ("iteration", "expected"),
    [(1, False), (999, False), (1000, True), (1001, False), (2000, True)],
)
def test_checkpoint_save_due_every_thousand_iterations(iteration, expected):
    assert train_mod._checkpoint_save_due(iteration, every=1000) is expected


def test_iteration_checkpoints_are_unique_and_rotated(tmp_path):
    agent = TinyAgent()
    for iteration in (1, 2, 3):
        agent.iteration_count = iteration
        train_mod._save_iteration_checkpoint(agent, tmp_path, iteration, "checkpoint_iter_", seed=11)

    assert torch.load(tmp_path / "checkpoint_iter_1.pt", weights_only=False)["iteration"] == 1
    assert torch.load(tmp_path / "checkpoint_iter_3.pt", weights_only=False)["iteration"] == 3


def test_light_checkpoint_contains_only_strategy_artifact(tmp_path):
    agent = LightCheckpointAgent()
    agent.iteration_count = 4

    train_mod._save_iteration_checkpoint(agent, tmp_path, 4, seed=11)
    path = train_mod._save_iteration_light_checkpoint(agent, tmp_path, 4, seed=11)

    assert path.name == "light_checkpoint_iter_4.pt"
    assert torch.load(path, weights_only=False) == {"iteration": 4, "seed": 11, "strategy_net": {}}


def test_disabled_transfer_preserves_legacy_monolithic_startup(tmp_path):
    config_path = tmp_path / "config.yaml"
    try:
        _load_transfer_config(
            config_path,
            [
                "num_players: 6",
                "teacher_transfer_enabled: false",
                "network_architecture: monolithic_v1",
            ],
        )

        agent = train_mod.train_self_play_multi(num_iterations=0, save_dir=tmp_path)

        assert agent.network_architecture == MONOLITHIC_ARCHITECTURE
        assert agent.teacher_transfer_provenance is None
    finally:
        config_mod.load_config("config.yaml")


def test_enabled_transfer_starts_card_context_and_persists_provenance(tmp_path, monkeypatch):
    config_path = tmp_path / "config.yaml"
    source_path = tmp_path / "teacher.pt"
    source_path.touch()
    calls = []
    expected_provenance = {
        "mode": "card_encoder_warmstart",
        "copied_blocks": ["strategy_net.card_encoder"],
        "source_path": str(source_path.resolve()),
        "checksum_sha256": "a" * 64,
        "source_architecture": CARD_CONTEXT_ARCHITECTURE,
        "source_encoding_version": "history_summary_v3",
        "source_game_rules_version": "holdem_standard_hu_v2",
        "teacher_num_players": 2,
        "freeze": True,
    }

    def transfer(agent, path, freeze=False):
        calls.append((agent.network_architecture, Path(path), freeze))
        return CardEncoderWarmstartProvenance(
            mode="card_encoder_warmstart",
            copied_blocks=("strategy_net.card_encoder",),
            source_path=source_path.resolve(),
            checksum_sha256="a" * 64,
            source_architecture=CARD_CONTEXT_ARCHITECTURE,
            source_encoding_version="history_summary_v3",
            source_game_rules_version="holdem_standard_hu_v2",
            teacher_num_players=2,
            freeze=True,
        )

    try:
        _load_transfer_config(
            config_path,
            [
                "num_players: 6",
                "network_architecture: monolithic_v1",
                "hidden_size: 8",
                "teacher_transfer_enabled: true",
                "teacher_transfer_mode: card_encoder_warmstart",
                f"teacher_transfer_checkpoint: {source_path}",
                "teacher_transfer_freeze_card_encoder: true",
            ],
        )
        monkeypatch.setattr(DeepCFRAgent, "load_card_encoder_from_hu_checkpoint", transfer)

        agent = train_mod.train_self_play_multi(num_iterations=0, save_dir=tmp_path)
        checkpoint = agent._build_checkpoint()

        assert calls == [(CARD_CONTEXT_ARCHITECTURE, source_path, True)]
        assert agent.network_architecture == CARD_CONTEXT_ARCHITECTURE
        assert checkpoint["teacher_transfer_provenance"] == expected_provenance
    finally:
        config_mod.load_config("config.yaml")


def test_enabled_transfer_resume_restores_provenance_without_retransfer(tmp_path, monkeypatch):
    config_path = tmp_path / "config.yaml"
    checkpoint_path = tmp_path / "six-max-full.pt"
    source_path = tmp_path / "teacher.pt"
    source_path.touch()
    provenance = {
        "mode": "card_encoder_warmstart",
        "copied_blocks": ["strategy_net.card_encoder"],
        "source_path": str(source_path.resolve()),
        "checksum_sha256": "b" * 64,
        "source_architecture": CARD_CONTEXT_ARCHITECTURE,
        "source_encoding_version": "history_summary_v3",
        "teacher_num_players": 2,
        "freeze": False,
    }
    try:
        _load_transfer_config(config_path, ["num_players: 6", "hidden_size: 8"])
        source = DeepCFRAgent(
            player_id=0,
            num_players=6,
            hidden_size=8,
            network_architecture=CARD_CONTEXT_ARCHITECTURE,
        )
        source.teacher_transfer_provenance = provenance
        torch.save(source._build_checkpoint(), checkpoint_path)
        _load_transfer_config(
            config_path,
            [
                "num_players: 6",
                "network_architecture: monolithic_v1",
                "hidden_size: 8",
                "teacher_transfer_enabled: true",
                "teacher_transfer_mode: card_encoder_warmstart",
                f"teacher_transfer_checkpoint: {source_path}",
            ],
        )
        monkeypatch.setattr(
            DeepCFRAgent,
            "load_card_encoder_from_hu_checkpoint",
            lambda *_args, **_kwargs: pytest.fail("resume не должен повторно переносить веса"),
        )

        restored = train_mod.train_self_play_multi(
            num_iterations=0,
            save_dir=tmp_path,
            initial_checkpoint=str(checkpoint_path),
        )

        assert restored.teacher_transfer_provenance == provenance
        assert restored._build_checkpoint()["teacher_transfer_provenance"] == provenance
    finally:
        config_mod.load_config("config.yaml")


def test_enabled_transfer_resume_restores_frozen_card_encoder(tmp_path, monkeypatch):
    """Ломается, если resume теряет freeze, записанный в provenance warm-start."""
    config_path = tmp_path / "config.yaml"
    checkpoint_path = tmp_path / "six-max-frozen-full.pt"
    source_path = tmp_path / "teacher.pt"
    source_path.touch()
    provenance = {
        "mode": "card_encoder_warmstart",
        "copied_blocks": ["strategy_net.card_encoder"],
        "source_path": str(source_path.resolve()),
        "checksum_sha256": "c" * 64,
        "source_architecture": CARD_CONTEXT_ARCHITECTURE,
        "source_encoding_version": "history_summary_v3",
        "teacher_num_players": 2,
        "freeze": True,
    }
    try:
        _load_transfer_config(config_path, ["num_players: 6", "hidden_size: 8"])
        source = DeepCFRAgent(
            player_id=0,
            num_players=6,
            hidden_size=8,
            network_architecture=CARD_CONTEXT_ARCHITECTURE,
        )
        source.teacher_transfer_provenance = provenance
        torch.save(source._build_checkpoint(), checkpoint_path)
        _load_transfer_config(
            config_path,
            [
                "num_players: 6",
                "network_architecture: monolithic_v1",
                "hidden_size: 8",
                "teacher_transfer_enabled: true",
                "teacher_transfer_mode: card_encoder_warmstart",
                f"teacher_transfer_checkpoint: {source_path}",
                "teacher_transfer_freeze_card_encoder: true",
            ],
        )
        monkeypatch.setattr(
            DeepCFRAgent,
            "load_card_encoder_from_hu_checkpoint",
            lambda *_args, **_kwargs: pytest.fail("resume не должен повторно переносить веса"),
        )

        restored = train_mod.train_self_play_multi(
            num_iterations=0,
            save_dir=tmp_path,
            initial_checkpoint=str(checkpoint_path),
        )

        assert all(
            not parameter.requires_grad
            for parameter in restored.strategy_net.card_encoder.parameters()
        )
    finally:
        config_mod.load_config("config.yaml")


def test_resume_rejects_non_boolean_transfer_freeze(tmp_path):
    """Ломается, если повреждённый provenance молча меняет freeze card_encoder."""
    config_path = tmp_path / "config.yaml"
    checkpoint_path = tmp_path / "six-max-invalid-freeze.pt"
    try:
        _load_transfer_config(config_path, ["num_players: 6", "hidden_size: 8"])
        source = DeepCFRAgent(
            player_id=0,
            num_players=6,
            hidden_size=8,
            network_architecture=CARD_CONTEXT_ARCHITECTURE,
        )
        source.teacher_transfer_provenance = {"freeze": 1}
        for parameter in source.strategy_net.parameters():
            parameter.grad = torch.ones_like(parameter)
        source.strategy_optimizer.step()
        torch.save(source._build_checkpoint(), checkpoint_path)
        target = DeepCFRAgent(
            player_id=0,
            num_players=6,
            hidden_size=8,
            network_architecture=CARD_CONTEXT_ARCHITECTURE,
        )
        before_strategy = {
            name: value.detach().clone()
            for name, value in target.strategy_net.state_dict().items()
        }
        before_provenance = {"freeze": False}
        target.teacher_transfer_provenance = before_provenance

        with pytest.raises(ValueError, match="freeze"):
            target.load_model(checkpoint_path)

        assert all(
            torch.equal(value, before_strategy[name])
            for name, value in target.strategy_net.state_dict().items()
        )
        assert target.strategy_optimizer.state_dict()["state"] == {}
        assert target.teacher_transfer_provenance == before_provenance
    finally:
        config_mod.load_config("config.yaml")


def test_atomic_save_keeps_previous_file_after_failure(tmp_path, monkeypatch):
    target = tmp_path / "checkpoint.pt"
    torch.save({"old": True}, target)

    def failing_save(_payload, path):
        path.write_bytes(b"partial")
        raise OSError("disk full")

    monkeypatch.setattr(train_mod.torch, "save", failing_save)
    with pytest.raises(OSError, match="disk full"):
        train_mod._atomic_torch_save({"new": True}, target)

    assert torch.load(target, weights_only=False) == {"old": True}


@pytest.mark.parametrize(
    ("iteration", "expected"),
    [
        (1001, [1000, 1000, 1000, 1000, 1000]),
        (2001, [1000, 1000, 1000, 1000, 2000]),
        (3001, [1000, 1000, 1000, 2000, 3000]),
        (6001, [2000, 3000, 4000, 5000, 6000]),
    ],
)
def test_opponent_pool_uses_thousand_step_warmup(tmp_path, iteration, expected):
    for checkpoint_iteration in range(1000, 7001, 1000):
        (tmp_path / f"multi_checkpoint_iter_{checkpoint_iteration}.pt").touch()

    paths = train_mod._opponent_checkpoint_paths(
        iteration, tmp_path, 5, checkpoint_every=1000, historical_every=50000
    )

    assert sorted(train_mod._checkpoint_iteration(path) for path in paths) == expected


def test_opponent_pool_uses_last_slot_and_previous_ten_slots(tmp_path, monkeypatch):
    for checkpoint_iteration in range(1000, 17001, 1000):
        (tmp_path / f"multi_checkpoint_iter_{checkpoint_iteration}.pt").touch()
    monkeypatch.setattr(train_mod.random, "sample", lambda candidates, count: candidates[-count:])

    paths = train_mod._opponent_checkpoint_paths(17001, tmp_path, 5, 1000, 50000)

    assert sorted(train_mod._checkpoint_iteration(path) for path in paths) == [13000, 14000, 15000, 16000, 17000]


def test_opponent_pool_uses_mature_historical_checkpoint(tmp_path, monkeypatch):
    for checkpoint_iteration in range(1000, 100001, 1000):
        (tmp_path / f"multi_checkpoint_iter_{checkpoint_iteration}.pt").touch()
    monkeypatch.setattr(train_mod.random, "sample", lambda candidates, count: candidates[-count:])

    paths = train_mod._opponent_checkpoint_paths(100001, tmp_path, 5, 1000, 50000)

    assert sorted(train_mod._checkpoint_iteration(path) for path in paths) == [50000, 97000, 98000, 99000, 100000]


def test_permanent_legacy_pool_keeps_historical_checkpoint_with_four_own_slots(tmp_path, monkeypatch):
    for checkpoint_iteration in range(1000, 100001, 1000):
        (tmp_path / f"multi_checkpoint_iter_{checkpoint_iteration}.pt").touch()
    monkeypatch.setattr(train_mod.random, "sample", lambda candidates, count: candidates[-count:])

    paths = train_mod._opponent_checkpoint_paths(100001, tmp_path, 4, 1000, 50000)

    assert sorted(train_mod._checkpoint_iteration(path) for path in paths) == [50000, 98000, 99000, 100000]


def test_first_historical_checkpoint_is_fixed_in_pool_before_next_milestone(tmp_path, monkeypatch):
    for checkpoint_iteration in range(1000, 51001, 1000):
        (tmp_path / f"multi_checkpoint_iter_{checkpoint_iteration}.pt").touch()
    monkeypatch.setattr(train_mod.random, "sample", lambda candidates, count: candidates[:count])

    paths = train_mod._opponent_checkpoint_paths(51001, tmp_path, 4, 1000, 50000)

    assert sorted(train_mod._checkpoint_iteration(path) for path in paths) == [41000, 42000, 50000, 51000]


def test_opponent_pool_duplicates_available_full_checkpoint_after_partial_resume(tmp_path):
    full_path = tmp_path / "multi_checkpoint_iter_1000.pt"
    full_path.touch()

    paths = train_mod._opponent_checkpoint_paths(3001, tmp_path, 5, 1000, 50000)

    assert paths == [full_path] * 5


def test_opponent_pool_uses_full_checkpoint_and_ignores_old_advantage_file(tmp_path):
    full_path = tmp_path / "multi_checkpoint_iter_1000.pt"
    old_advantage_path = tmp_path / "opponent_advantage_iter_1000.pt"
    full_path.touch()
    old_advantage_path.touch()

    paths = train_mod._opponent_checkpoint_paths(1001, tmp_path, 5, 1000, 50000)

    assert paths == [full_path] * 5


def test_opponent_pool_schedule_keeps_composition_until_next_checkpoint_slot(monkeypatch, tmp_path):
    selected_iterations = []

    def select(iteration, *_args, **_kwargs):
        selected_iterations.append(iteration)
        return [tmp_path / f"multi_checkpoint_iter_{number}.pt" for number in (7000, 6000, 5000, 4000, 1000)]

    monkeypatch.setattr(train_mod, "_opponent_checkpoint_paths", select)
    schedule = train_mod.OpponentPoolSchedule(tmp_path, checkpoint_every=1000, historical_every=50000)

    first = schedule.paths_for_iteration(7750, num_opponents=5)
    middle = schedule.paths_for_iteration(7751, num_opponents=5)
    next_slot = schedule.paths_for_iteration(8001, num_opponents=5)

    assert selected_iterations == [7750, 8001]
    assert first == middle
    assert next_slot == first


def test_full_checkpoint_retention_keeps_eleven_recent_and_two_historical(tmp_path):
    for iteration in range(1000, 100001, 1000):
        (tmp_path / f"multi_checkpoint_iter_{iteration}.pt").touch()
    hu_checkpoint = tmp_path / "hu_checkpoint_iter_5000.pt"
    hu_checkpoint.touch()

    train_mod._prune_full_checkpoints(tmp_path, historical_every=50000)

    assert sorted(train_mod._heavy_checkpoints(tmp_path)) == [50000, *range(89000, 101000, 1000)]
    assert hu_checkpoint.exists()


def test_checkpoint_pipeline_keeps_full_and_light_history_without_advantage_files(tmp_path):
    agent = LightCheckpointAgent()
    for iteration in (1000, 2000):
        agent.iteration_count = iteration
        train_mod._save_iteration_checkpoint(agent, tmp_path, iteration)
        train_mod._save_iteration_light_checkpoint(agent, tmp_path, iteration)

    assert sorted(train_mod._heavy_checkpoints(tmp_path)) == [1000, 2000]
    assert sorted(train_mod._checkpoint_paths(tmp_path, train_mod._LIGHT_CHECKPOINT_PREFIX)) == [1000, 2000]
    assert not list(tmp_path.glob("opponent_advantage_iter_*.pt"))


def test_hu_light_checkpoint_is_retained_next_to_hu_full_checkpoint(tmp_path):
    """Ломается, если pruning HU удаляет light checkpoint без six-max full файла."""
    agent = LightCheckpointAgent()
    agent.iteration_count = 5000
    (tmp_path / "hu_checkpoint_iter_5000.pt").touch()

    path = train_mod._save_iteration_light_checkpoint(
        agent,
        tmp_path,
        5000,
        prefix="hu_light_checkpoint_iter_",
        full_checkpoint_prefix="hu_checkpoint_iter_",
    )

    assert path.name == "hu_light_checkpoint_iter_5000.pt"
    assert path.exists()


def test_hu_iteration_light_uses_own_namespace_without_overwriting_generic_light(
    tmp_path,
    monkeypatch,
):
    """Ломается, если HU save перезаписывает generic light той же итерации."""
    generic_light = tmp_path / "light_checkpoint_iter_5000.pt"
    generic_light.write_bytes(b"generic-light")

    def save_full(_agent, path, seed=None):
        Path(path).touch()
        return Path(path)

    monkeypatch.setattr(train_mod, "_save_hu_checkpoint", save_full)

    _, hu_light = train_mod._save_hu_iteration_checkpoints(
        cast(DeepCFRAgent, LightCheckpointAgent()),
        tmp_path,
        5000,
    )

    assert hu_light.name == "hu_light_checkpoint_iter_5000.pt"
    assert generic_light.read_bytes() == b"generic-light"
    assert hu_light.exists()


def test_hu_retention_protects_current_checkpoint_from_older_series(tmp_path, monkeypatch):
    """Текущий HU checkpoint не должен удаляться из-за больших номеров старого запуска."""
    for iteration in (50000, 100000, 140000, 145000):
        (tmp_path / f"hu_checkpoint_iter_{iteration}.pt").touch()

    def save_full(_agent, path, seed=None):
        Path(path).touch()
        return Path(path)

    monkeypatch.setattr(train_mod, "_save_hu_checkpoint", save_full)
    full_path, light_path = train_mod._save_hu_iteration_checkpoints(
        cast(DeepCFRAgent, LightCheckpointAgent()),
        tmp_path,
        5000,
    )

    assert full_path.is_file()
    assert light_path.is_file()


def test_hu_and_generic_light_pruning_are_namespace_isolated(tmp_path):
    """Ломается, если pruning одного режима удаляет light checkpoint другого режима."""
    (tmp_path / "multi_checkpoint_iter_5000.pt").touch()
    generic_matching = tmp_path / "light_checkpoint_iter_5000.pt"
    generic_matching.touch()
    generic_orphan = tmp_path / "light_checkpoint_iter_10000.pt"
    generic_orphan.touch()
    (tmp_path / "hu_checkpoint_iter_5000.pt").touch()
    hu_matching = tmp_path / "hu_light_checkpoint_iter_5000.pt"
    hu_matching.touch()
    hu_orphan = tmp_path / "hu_light_checkpoint_iter_10000.pt"
    hu_orphan.touch()

    train_mod._prune_light_checkpoints(
        tmp_path,
        full_checkpoint_prefix="hu_checkpoint_iter_",
        light_checkpoint_prefix="hu_light_checkpoint_iter_",
    )

    assert hu_matching.exists()
    assert not hu_orphan.exists()
    assert generic_matching.exists()
    assert generic_orphan.exists()

    train_mod._prune_light_checkpoints(tmp_path)

    assert generic_matching.exists()
    assert not generic_orphan.exists()
    assert hu_matching.exists()


def test_hu_full_checkpoint_retention_keeps_recent_and_milestones(tmp_path):
    """Ломается, если HU retention смешивает namespace или удаляет resume checkpoint."""
    hu_iterations = (5000, 10000, 15000, 50000, 55000, 100000, 105000)
    for iteration in hu_iterations:
        (tmp_path / f"hu_checkpoint_iter_{iteration}.pt").touch()
    (tmp_path / "multi_checkpoint_iter_5000.pt").touch()
    (tmp_path / "hu_checkpoint_final.pt").touch()
    (tmp_path / "checkpoint-notes.pt").touch()

    retention = train_mod._prune_hu_full_checkpoints(
        tmp_path,
        milestone_every=50000,
        keep_recent=2,
        keep_milestones=2,
    )

    assert sorted(train_mod._checkpoint_paths(tmp_path, "hu_checkpoint_iter_")) == [
        50000,
        55000,
        100000,
        105000,
    ]
    assert retention["recent"] == [55000, 105000]
    assert retention["milestones"] == [50000, 100000]
    assert retention["deleted"] == [5000, 10000, 15000]
    assert (tmp_path / "multi_checkpoint_iter_5000.pt").exists()
    assert (tmp_path / "hu_checkpoint_final.pt").exists()
    assert (tmp_path / "checkpoint-notes.pt").exists()

    assert train_mod._prune_hu_full_checkpoints(
        tmp_path,
        milestone_every=50000,
        keep_recent=2,
        keep_milestones=2,
    )["deleted"] == []


def test_hu_light_pruning_removes_only_orphans_for_retained_full_checkpoints(tmp_path):
    """Ломается, если light checkpoint удаляется без проверки HU full checkpoint."""
    for iteration in (55000, 105000):
        (tmp_path / f"hu_checkpoint_iter_{iteration}.pt").touch()
        (tmp_path / f"hu_light_checkpoint_iter_{iteration}.pt").touch()
    orphan = tmp_path / "hu_light_checkpoint_iter_15000.pt"
    orphan.touch()
    final_light = tmp_path / "hu_light_checkpoint_final.pt"
    final_light.touch()

    train_mod._prune_light_checkpoints(
        tmp_path,
        full_checkpoint_prefix="hu_checkpoint_iter_",
        light_checkpoint_prefix="hu_light_checkpoint_iter_",
    )

    assert (tmp_path / "hu_light_checkpoint_iter_55000.pt").exists()
    assert (tmp_path / "hu_light_checkpoint_iter_105000.pt").exists()
    assert not orphan.exists()
    assert final_light.exists()


def test_hu_retention_keeps_latest_resumable_checkpoint_when_limits_are_zero(tmp_path):
    """Ломается, если некорректный лимит retention удаляет последний HU checkpoint."""
    for iteration in (5000, 10000, 15000):
        (tmp_path / f"hu_checkpoint_iter_{iteration}.pt").touch()

    train_mod._prune_hu_full_checkpoints(
        tmp_path,
        milestone_every=50000,
        keep_recent=0,
        keep_milestones=0,
    )

    assert sorted(train_mod._checkpoint_paths(tmp_path, "hu_checkpoint_iter_")) == [15000]


def test_hu_iteration_checkpoint_saves_full_before_pruning_and_light(tmp_path, monkeypatch):
    """Ломается, если HU pruning может выполниться до успешного полного сохранения."""
    events = []

    def save_full(_agent, path, seed=None):
        events.append(("full", Path(path).name, seed))
        Path(path).touch()
        return Path(path)

    def prune_full(directory, **kwargs):
        assert kwargs["protected_iterations"] == {5000}
        events.append(("prune_full", Path(directory).name))
        return {"recent": [], "milestones": [], "deleted": []}

    def save_light(_agent, directory, iteration, **kwargs):
        events.append(
            ("light", Path(directory).name, iteration, kwargs["prefix"], kwargs["full_checkpoint_prefix"])
        )
        path = Path(directory) / f"{kwargs['prefix']}{iteration}.pt"
        path.touch()
        return path

    monkeypatch.setattr(train_mod, "_save_hu_checkpoint", save_full)
    monkeypatch.setattr(train_mod, "_prune_hu_full_checkpoints", prune_full)
    monkeypatch.setattr(train_mod, "_save_iteration_light_checkpoint", save_light)

    full_path, light_path = train_mod._save_hu_iteration_checkpoints(
        cast(DeepCFRAgent, TinyAgent()),
        tmp_path,
        5000,
        seed=17,
    )

    assert full_path.name == "hu_checkpoint_iter_5000.pt"
    assert light_path.name == "hu_light_checkpoint_iter_5000.pt"
    assert events == [
        ("full", "hu_checkpoint_iter_5000.pt", 17),
        ("prune_full", tmp_path.name),
        ("light", tmp_path.name, 5000, "hu_light_checkpoint_iter_", "hu_checkpoint_iter_"),
    ]


def test_hu_iteration_checkpoint_does_not_prune_after_full_save_failure(tmp_path, monkeypatch):
    """Ломается, если ошибка атомарного HU save удаляет предыдущий resume checkpoint."""
    previous = tmp_path / "hu_checkpoint_iter_5000.pt"
    previous.touch()

    def failing_save(_agent, _path, seed=None):
        raise OSError("disk full")

    monkeypatch.setattr(train_mod, "_save_hu_checkpoint", failing_save)
    monkeypatch.setattr(
        train_mod,
        "_prune_hu_full_checkpoints",
        lambda _directory: pytest.fail("pruning не должен запускаться после ошибки save"),
    )

    with pytest.raises(OSError, match="disk full"):
        train_mod._save_hu_iteration_checkpoints(
            cast(DeepCFRAgent, TinyAgent()),
            tmp_path,
            10000,
        )

    assert previous.exists()


def test_opponent_pool_caches_full_checkpoint_strategy_state(tmp_path, monkeypatch):
    full_path = tmp_path / "multi_checkpoint_iter_1000.pt"
    torch.save(
        {
            "action_space_version": "six_fixed_v2",
            "strategy_net": {"weight": torch.tensor([1.0])},
            "advantage_net": {"weight": torch.tensor([2.0])},
            "advantage_buffer": ["large buffer"],
        },
        full_path,
    )
    loaded_paths = []
    original_load = train_mod.torch.load

    def tracked_load(path, *args, **kwargs):
        loaded_paths.append(Path(path))
        return original_load(path, *args, **kwargs)

    class OpponentPoolAgent:
        num_players = 6

        def clear_opponent_advantage_states(self):
            raise AssertionError("Старый advantage API больше не должен использоваться")

        def set_opponent_strategy_states(self, states, traversing_player):
            assert len(states) == 5
            assert traversing_player == 0
            assert all(state["weight"].item() == 1.0 for state in states)

    monkeypatch.setattr(train_mod.torch, "load", tracked_load)
    state_cache = {}

    train_mod._configure_opponent_pool(
        OpponentPoolAgent(), 1001, tmp_path, traversing_player=0, state_cache=state_cache
    )
    train_mod._configure_opponent_pool(
        OpponentPoolAgent(), 1001, tmp_path, traversing_player=0, state_cache=state_cache
    )

    assert loaded_paths == [full_path]


def test_full_checkpoint_strategy_loader_requires_strategy_net(tmp_path):
    full_path = tmp_path / "multi_checkpoint_iter_1000.pt"
    torch.save(
        {
            "action_space_version": "six_fixed_v2",
            "advantage_net": {"weight": torch.tensor([1.0])},
        },
        full_path,
    )

    with pytest.raises(ValueError, match="strategy_net"):
        train_mod._load_full_checkpoint_strategy_state(full_path)
