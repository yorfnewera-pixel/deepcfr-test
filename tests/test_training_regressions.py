import pokers as pkrs
import pytest
import sys
import torch
import numpy as np
from types import SimpleNamespace
from pathlib import Path

from src.core.action_space import ACTION_SPACE_VERSION, NUM_ACTIONS
from src.core import deep_cfr as deep_cfr_mod
from src.core.deep_cfr import CHECKPOINT_FORMAT_VERSION, DeepCFRAgent
from src.core.model import MONOLITHIC_ARCHITECTURE
from src.core.traversal_errors import TraversalFailure, TraversalFailureContext
from src.agents.random_agent import RandomAgent
from src.training.train import (
    _assign_checkpoint_opponents,
    _apply_process_priority,
    _current_strategy_opponent_count,
    _latest_strategy_checkpoint_path,
    _log_multi_cfr_diagnostics,
    _transition_checkpoint_iterations,
    _training_thread_limit,
    _print_opponent_checkpoints,
    _traversal_thread_limit,
)
from src.training import train as train_mod


@pytest.fixture(autouse=True)
def _isolate_action_only_training_mode(monkeypatch):
    """Изолирует unit-тесты обычного пути от выбранного runtime режима config.yaml."""
    original_cfg_get = train_mod.cfg_get

    def cfg_get_for_action_only_tests(key, default=None):
        if key in {"hu_current_policy_self_play", "d2cfr_enabled"}:
            return False
        return original_cfg_get(key, default)

    monkeypatch.setattr(train_mod, "cfg_get", cfg_get_for_action_only_tests)
    monkeypatch.setattr(deep_cfr_mod, "cfg_get", cfg_get_for_action_only_tests)


def _traversal_failure_context(reason="Превышена допустимая глубина обхода"):
    return TraversalFailureContext(
        iteration=1,
        traversal_index=None,
        traversing_player=0,
        acting_player=0,
        depth=201,
        reason=reason,
        action_trace=(),
    )


def _fake_agent_with_traversal_outcomes(outcomes):
    agent = SimpleNamespace(
        iteration_count=0,
        num_players=2,
        optimizer=SimpleNamespace(param_groups=[{"lr": 1e-4}]),
        advantage_buffer=[],
        strategy_buffer=[],
        strategy_net=SimpleNamespace(state_dict=lambda: {"weight": torch.tensor([1.0])}),
        traversal_attempts=0,
        traversal_successes=0,
        traversal_failures=0,
        train_advantage_calls=0,
        prepare_iteration=lambda *_args, **_kwargs: None,
        reset_traversal_stats=lambda: None,
    )
    pending_outcomes = iter(outcomes)

    def cfr_traverse_multi(*_args, **_kwargs):
        outcome = next(pending_outcomes)
        if outcome is not None:
            raise outcome

    agent.cfr_traverse_multi = cfr_traverse_multi
    agent.record_traversal_attempt = lambda: setattr(
        agent, "traversal_attempts", agent.traversal_attempts + 1
    )
    agent.record_traversal_success = lambda: setattr(
        agent, "traversal_successes", agent.traversal_successes + 1
    )
    agent.record_traversal_failure = lambda _error: setattr(
        agent, "traversal_failures", agent.traversal_failures + 1
    )
    agent.train_advantage_network_multi = lambda: setattr(
        agent, "train_advantage_calls", agent.train_advantage_calls + 1
    ) or 0.0
    agent.train_strategy_network = lambda: 0.0
    return agent


def _configure_training_error_mode(monkeypatch, mode, limit):
    monkeypatch.setattr(train_mod, "cfg_training_error_mode", lambda: mode, raising=False)
    monkeypatch.setattr(
        train_mod,
        "cfg_training_max_failed_traversals_per_iteration",
        lambda: limit,
        raising=False,
    )


def _configure_fake_training_loop(monkeypatch, agent):
    monkeypatch.setattr(train_mod, "DeepCFRAgent", lambda **_kwargs: agent)
    monkeypatch.setattr(train_mod, "_new_hand", lambda *_args: None)
    monkeypatch.setattr(train_mod, "_create_writer", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(train_mod, "_log_multi_cfr_diagnostics", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(train_mod, "_configure_strategy_opponent_pool", lambda *_args, **_kwargs: [])
    monkeypatch.setattr(train_mod, "_print_current_strategy_opponents", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(train_mod, "_print_opponent_checkpoints", lambda *_args, **_kwargs: None)


def _add_traversal_recorders(agent):
    agent.record_traversal_attempt = lambda: None
    agent.record_traversal_success = lambda: None
    agent.record_traversal_failure = lambda _error: None
    return agent


def test_cli_forwards_teacher_transfer_and_stage_b_contract_flags(monkeypatch):
    received = {}
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train.py",
            "--self-play-multi",
            "--iterations", "1",
            "--traversals", "1",
            "--teacher-transfer-enabled",
            "--teacher-transfer-mode", "card_encoder_warmstart",
            "--teacher-transfer-checkpoint", "hu.pt",
            "--teacher-transfer-freeze-card-encoder",
            "--teacher-hu-aux-distillation-enabled",
            "--teacher-hu-aux-distillation-weight", "0.25",
        ],
    )
    monkeypatch.setattr(train_mod, "_apply_process_priority", lambda *_args: None)
    monkeypatch.setattr(
        train_mod,
        "train_self_play_multi",
        lambda **kwargs: received.update(kwargs),
    )

    train_mod.main()

    assert received["teacher_transfer_enabled"] is True
    assert received["teacher_transfer_mode"] == "card_encoder_warmstart"
    assert received["teacher_transfer_checkpoint"] == "hu.pt"
    assert received["teacher_transfer_freeze_card_encoder"] is True
    assert received["teacher_hu_aux_distillation_enabled"] is True
    assert received["teacher_hu_aux_distillation_weight"] == 0.25


def test_cli_teacher_transfer_booleans_do_not_require_boolean_optional_action(monkeypatch):
    received = {}
    monkeypatch.delattr(train_mod.argparse, "BooleanOptionalAction", raising=False)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train.py",
            "--self-play-multi",
            "--iterations", "1",
            "--traversals", "1",
            "--teacher-transfer-enabled",
            "--no-teacher-transfer-freeze-card-encoder",
            "--no-teacher-hu-aux-distillation-enabled",
        ],
    )
    monkeypatch.setattr(train_mod, "_apply_process_priority", lambda *_args: None)
    monkeypatch.setattr(
        train_mod,
        "train_self_play_multi",
        lambda **kwargs: received.update(kwargs),
    )

    train_mod.main()

    assert received["teacher_transfer_enabled"] is True
    assert received["teacher_transfer_freeze_card_encoder"] is False
    assert received["teacher_hu_aux_distillation_enabled"] is False


def test_skip_traversal_continues_after_one_failure(monkeypatch, tmp_path):
    agent = _fake_agent_with_traversal_outcomes([TraversalFailure(_traversal_failure_context()), None])
    _configure_training_error_mode(monkeypatch, "skip_traversal", limit=2)
    _configure_fake_training_loop(monkeypatch, agent)

    train_mod.train_self_play_multi(
        num_iterations=1,
        traversals_per_iteration=2,
        evaluate_every=0,
        save_dir=tmp_path,
        num_players=2,
    )

    assert agent.traversal_attempts == 2
    assert agent.traversal_failures == 1
    assert agent.train_advantage_calls == 1


def test_skip_traversal_stops_when_iteration_limit_is_reached(monkeypatch, tmp_path):
    agent = _fake_agent_with_traversal_outcomes(
        [TraversalFailure(_traversal_failure_context()), TraversalFailure(_traversal_failure_context())]
    )
    _configure_training_error_mode(monkeypatch, "skip_traversal", limit=2)
    _configure_fake_training_loop(monkeypatch, agent)

    with pytest.raises(RuntimeError, match="training_max_failed_traversals_per_iteration"):
        train_mod.train_self_play_multi(
            num_iterations=1,
            traversals_per_iteration=2,
            evaluate_every=0,
            save_dir=tmp_path,
            num_players=2,
        )

    assert agent.train_advantage_calls == 0


def test_skip_traversal_limit_is_global_across_heads_up_players(monkeypatch, tmp_path):
    agent = _fake_agent_with_traversal_outcomes(
        [TraversalFailure(_traversal_failure_context()), TraversalFailure(_traversal_failure_context())]
    )
    _configure_training_error_mode(monkeypatch, "skip_traversal", limit=2)
    _configure_fake_training_loop(monkeypatch, agent)

    with pytest.raises(RuntimeError, match="training_max_failed_traversals_per_iteration"):
        train_mod.train_self_play_multi(
            num_iterations=1,
            traversals_per_iteration=1,
            evaluate_every=0,
            save_dir=tmp_path,
            num_players=2,
            trainable_players=2,
        )

    assert agent.traversal_failures == 2
    assert agent.train_advantage_calls == 0


def test_strict_mode_propagates_first_traversal_failure_without_training(monkeypatch, tmp_path):
    failure = TraversalFailure(_traversal_failure_context())
    agent = _fake_agent_with_traversal_outcomes([failure])
    _configure_training_error_mode(monkeypatch, "strict", limit=2)
    _configure_fake_training_loop(monkeypatch, agent)

    with pytest.raises(TraversalFailure) as caught:
        train_mod.train_self_play_multi(
            num_iterations=1,
            traversals_per_iteration=1,
            evaluate_every=0,
            save_dir=tmp_path,
            num_players=2,
        )

    assert caught.value is failure
    assert agent.train_advantage_calls == 0


def test_skip_traversal_propagates_unrelated_runtime_error(monkeypatch, tmp_path):
    failure = RuntimeError("не относится к обходу")
    agent = _fake_agent_with_traversal_outcomes([failure])
    _configure_training_error_mode(monkeypatch, "skip_traversal", limit=2)
    _configure_fake_training_loop(monkeypatch, agent)

    with pytest.raises(RuntimeError, match="не относится к обходу"):
        train_mod.train_self_play_multi(
            num_iterations=1,
            traversals_per_iteration=1,
            evaluate_every=0,
            save_dir=tmp_path,
            num_players=2,
        )

    assert agent.train_advantage_calls == 0


def test_traversal_failure_stats_track_reason_depth_and_success():
    agent = DeepCFRAgent(player_id=0, num_players=2, device="cpu")

    agent.record_traversal_attempt()
    diagnostic = agent.record_traversal_failure(TraversalFailure(_traversal_failure_context()))
    agent.record_traversal_attempt()
    agent.record_traversal_success()

    stats = agent.get_traversal_stats()
    assert diagnostic == {
        "iteration": 1,
        "traversal_index": None,
        "traversing_player": 0,
        "acting_player": 0,
        "reason": "Превышена допустимая глубина обхода",
        "depth": 201,
        "action_trace": (),
    }
    assert stats["attempted"] == 2
    assert stats["successful"] == 1
    assert stats["failed"] == 1
    assert stats["failure_reasons"] == {"Превышена допустимая глубина обхода": 1}
    assert stats["depth_limit_hits"] == 1


def test_traversal_failure_diagnostic_retains_local_context_fields():
    agent = DeepCFRAgent(player_id=0, num_players=2, device="cpu")
    context = TraversalFailureContext(
        iteration=7,
        traversal_index=3,
        traversing_player=0,
        acting_player=1,
        depth=4,
        reason="Некорректная policy оппонента",
        action_trace=("slot 1: call",),
        details={
            "action": "slot 1: call",
            "status": "StateStatus.IllegalAction",
            "mask": [1.0, 1.0, 0.0, 0.0, 0.0, 0.0],
            "policy": [0.5, 0.5, 0.0, 0.0, 0.0, 0.0],
        },
    )

    diagnostic = agent.record_traversal_failure(TraversalFailure(context))

    assert diagnostic == {
        "iteration": 7,
        "traversal_index": 3,
        "traversing_player": 0,
        "acting_player": 1,
        "depth": 4,
        "reason": "Некорректная policy оппонента",
        "action_trace": ("slot 1: call",),
        "action": "slot 1: call",
        "status": "StateStatus.IllegalAction",
        "mask": [1.0, 1.0, 0.0, 0.0, 0.0, 0.0],
        "policy": [0.5, 0.5, 0.0, 0.0, 0.0, 0.0],
    }
    assert agent.get_traversal_stats()["failure_diagnostics"] == [diagnostic]


def test_iteration_summary_contains_all_requested_metrics():
    summary = train_mod._format_iteration_summary(4.5, 3.2, 0.1, 0.2)

    assert "Time/Iteration=4.5s" in summary
    assert "Time/Traversal=3.2s" in summary
    assert "Loss/Advantage=0.100000" in summary
    assert "Loss/Strategy=0.200000" in summary


def test_iteration_summary_includes_opponent_pool_setup_time():
    summary = train_mod._format_iteration_summary(
        iteration_elapsed=10.0,
        traversal_elapsed=4.0,
        advantage_loss=0.1,
        strategy_loss=0.2,
        opponent_setup_elapsed=0.3,
    )

    assert "Time/OpponentPoolSetup=0.3s" in summary


def test_iteration_summary_separates_advantage_and_strategy_training_time():
    summary = train_mod._format_iteration_summary(
        iteration_elapsed=10.0,
        traversal_elapsed=4.0,
        advantage_loss=0.1,
        strategy_loss=0.2,
        advantage_training_elapsed=2.5,
        strategy_training_elapsed=1.5,
        advantage_player_elapsed=(1.0, 1.5),
    )

    assert "Time/Advantage=2.5s" in summary
    assert "Time/Advantage/P0=1.0s" in summary
    assert "Time/Advantage/P1=1.5s" in summary
    assert "Time/Strategy=1.5s" in summary
    assert "Time/Training=4.0s" in summary


def test_training_timing_metrics_are_written_to_tensorboard_per_hu_leg():
    class Writer:
        def __init__(self):
            self.scalars = []

        def add_scalar(self, name, value, iteration):
            self.scalars.append((name, value, iteration))

    writer = Writer()

    train_mod._log_training_time_metrics(
        writer,
        iteration=7,
        advantage_training_elapsed=2.5,
        strategy_training_elapsed=1.5,
        advantage_player_elapsed=(1.0, 1.5),
    )

    assert writer.scalars == [
        ("Time/Advantage", 2.5, 7),
        ("Time/Strategy", 1.5, 7),
        ("Time/Training", 4.0, 7),
        ("Time/Advantage/P0", 1.0, 7),
        ("Time/Advantage/P1", 1.5, 7),
    ]


def test_hu_traversal_time_excludes_advantage_and_strategy_training():
    traversal_elapsed = train_mod._hu_traversal_phase_elapsed(
        iteration_phase_elapsed=10.0,
        advantage_player_elapsed=(2.5, 1.5),
        strategy_training_elapsed=2.0,
    )

    assert traversal_elapsed == 4.0


def test_training_uses_configured_advantage_learning_rate_from_first_iteration(monkeypatch, tmp_path):
    observed_learning_rates = []
    agent = SimpleNamespace(
        iteration_count=0,
        num_players=6,
        optimizer=SimpleNamespace(param_groups=[{"lr": 1e-4}]),
        advantage_buffer=[],
        strategy_buffer=[],
        strategy_net=SimpleNamespace(state_dict=lambda: {"weight": torch.tensor([1.0])}),
        prepare_iteration=lambda *_args, **_kwargs: None,
        reset_traversal_stats=lambda: None,
        cfr_traverse_multi=lambda *_args, **_kwargs: None,
        train_advantage_network_multi=lambda: observed_learning_rates.append(agent.optimizer.param_groups[0]["lr"]) or 0.0,
        train_strategy_network=lambda: 0.0,
    )
    _add_traversal_recorders(agent)
    monkeypatch.setattr(train_mod, "DeepCFRAgent", lambda **_kwargs: agent)
    monkeypatch.setattr(train_mod, "_new_hand", lambda *_args: None)
    monkeypatch.setattr(train_mod, "_log_multi_cfr_diagnostics", lambda *_args: {})
    monkeypatch.setattr(train_mod, "_print_opponent_checkpoints", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(train_mod, "_configure_strategy_opponent_pool", lambda *_args, **_kwargs: [])
    monkeypatch.setattr(train_mod, "_latest_strategy_checkpoint_path", lambda *_args: tmp_path / "current.pt")

    train_mod.train_self_play_multi(
        num_iterations=1,
        traversals_per_iteration=0,
        evaluate_every=0,
        save_dir=tmp_path,
    )

    assert observed_learning_rates == [1e-4]


def test_strategy_training_preload_uses_device_sampling(monkeypatch):
    agent = DeepCFRAgent(player_id=0, num_players=6, device="cpu")
    agent.training_preload_to_device = True
    agent.strategy_train_steps = 2
    agent.strategy_batch_size = 2
    policy = np.full(NUM_ACTIONS, 1.0 / NUM_ACTIONS, dtype=np.float32)
    mask = np.ones(NUM_ACTIONS, dtype=np.float32)
    for idx in range(4):
        agent.strategy_buffer.add(
            np.full(agent.input_size, float(idx), dtype=np.float32),
            policy,
            mask,
            idx + 1,
        )

    original_choice = np.random.choice

    def fail_step_numpy_choice(*args, **kwargs):
        if kwargs.get("replace") is False:
            return original_choice(*args, **kwargs)
        raise AssertionError("preload path должен сэмплировать индексы на torch device")

    monkeypatch.setattr(np.random, "choice", fail_step_numpy_choice)

    loss = agent.train_strategy_network()

    assert isinstance(loss, float)
    assert agent.last_strategy_profile["actual_steps"] == 2


def test_advantage_training_preload_uses_device_sampling(monkeypatch):
    agent = DeepCFRAgent(player_id=0, num_players=6, device="cpu")
    agent.training_preload_to_device = True
    agent.advantage_train_steps = 2
    agent.advantage_batch_size = 2
    regrets = np.arange(NUM_ACTIONS, dtype=np.float32)
    mask = np.ones(NUM_ACTIONS, dtype=np.float32)
    for idx in range(4):
        agent.advantage_buffer.add(
            np.full(agent.input_size, float(idx), dtype=np.float32),
            regrets,
            mask,
            idx + 1,
        )

    original_choice = np.random.choice

    def fail_step_numpy_choice(*args, **kwargs):
        if kwargs.get("replace") is False:
            return original_choice(*args, **kwargs)
        raise AssertionError("preload path должен сэмплировать индексы на torch device")

    monkeypatch.setattr(np.random, "choice", fail_step_numpy_choice)

    loss = agent.train_advantage_network_multi()

    assert isinstance(loss, float)
    assert agent.last_advantage_profile["actual_steps"] == 2


def test_training_can_traverse_all_heads_up_players(monkeypatch, tmp_path):
    traversing_players = []
    agent = SimpleNamespace(
        iteration_count=0,
        num_players=2,
        num_trainable_players=2,
        optimizer=SimpleNamespace(param_groups=[{"lr": 1e-4}]),
        advantage_buffer=[],
        strategy_buffer=[],
        strategy_net=SimpleNamespace(state_dict=lambda: {"weight": torch.tensor([1.0])}),
        set_opponent_strategy_states_by_player=lambda *_args, **_kwargs: None,
        prepare_iteration=lambda *_args, **_kwargs: None,
        reset_traversal_stats=lambda: None,
        cfr_traverse_multi=lambda _state, _iteration, traversing_player, **_kwargs: traversing_players.append(traversing_player),
        train_advantage_network_multi=lambda: 0.0,
        train_strategy_network=lambda: 0.0,
    )
    _add_traversal_recorders(agent)
    monkeypatch.setattr(train_mod, "DeepCFRAgent", lambda **_kwargs: agent)
    monkeypatch.setattr(train_mod, "_new_hand", lambda *_args: None)
    monkeypatch.setattr(train_mod, "_create_writer", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(train_mod, "_log_multi_cfr_diagnostics", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(train_mod, "_configure_strategy_opponent_pool", lambda *_args, **_kwargs: [])
    monkeypatch.setattr(train_mod, "_print_current_strategy_opponents", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(train_mod, "_print_opponent_checkpoints", lambda *_args, **_kwargs: None)

    train_mod.train_self_play_multi(
        num_iterations=1,
        traversals_per_iteration=2,
        evaluate_every=0,
        save_dir=tmp_path,
        num_players=2,
        trainable_players=2,
    )

    assert traversing_players == [0, 0, 1, 1]


def test_strategy_training_adds_teacher_policy_distillation_loss():
    agent = DeepCFRAgent(
        player_id=0,
        num_players=2,
        device="cpu",
        network_architecture=MONOLITHIC_ARCHITECTURE,
    )
    agent.strategy_train_steps = 1
    agent.strategy_batch_size = 2
    agent.strategy_distillation_lambda = 0.5
    agent.strategy_distillation_temperature = 1.0
    teacher = DeepCFRAgent(
        player_id=0,
        num_players=2,
        device="cpu",
        network_architecture=MONOLITHIC_ARCHITECTURE,
    )
    with torch.no_grad():
        teacher.strategy_net.action_head.bias.copy_(
            torch.tensor([4.0, -1.0, -1.0, -1.0, -1.0, -1.0])
        )
    agent.set_teacher_strategy_network(teacher.strategy_net.state_dict())

    policy = np.full(NUM_ACTIONS, 1.0 / NUM_ACTIONS, dtype=np.float32)
    mask = np.ones(NUM_ACTIONS, dtype=np.float32)
    for idx in range(4):
        agent.strategy_buffer.add(
            np.full(agent.input_size, float(idx), dtype=np.float32),
            policy,
            mask,
            idx + 1,
        )

    loss = agent.train_strategy_network()

    assert isinstance(loss, float)
    assert agent.last_strategy_profile["distillation_lambda"] == 0.5
    assert agent.last_strategy_profile["distillation_loss"] > 0.0


def test_sixmax_strategy_training_can_distill_from_heads_up_teacher(tmp_path):
    teacher = DeepCFRAgent(
        player_id=0,
        num_players=2,
        device="cpu",
        network_architecture=MONOLITHIC_ARCHITECTURE,
    )
    teacher.iteration_count = 10
    with torch.no_grad():
        teacher.strategy_net.action_head.bias.copy_(
            torch.tensor([3.0, -1.0, -1.0, -1.0, -1.0, -1.0])
        )
    checkpoint_path = tmp_path / "hu_teacher.pt"
    torch.save(teacher.build_light_checkpoint(), checkpoint_path)

    student = DeepCFRAgent(player_id=0, num_players=6, device="cpu")
    student.strategy_train_steps = 1
    student.strategy_batch_size = 2
    student.strategy_distillation_lambda = 0.25
    student.load_teacher_strategy_checkpoint(checkpoint_path)

    policy = np.full(NUM_ACTIONS, 1.0 / NUM_ACTIONS, dtype=np.float32)
    mask = np.ones(NUM_ACTIONS, dtype=np.float32)
    for idx in range(4):
        student.strategy_buffer.add(
            np.full(student.input_size, float(idx), dtype=np.float32),
            policy,
            mask,
            idx + 1,
        )

    loss = student.train_strategy_network()

    assert isinstance(loss, float)
    assert student.last_strategy_profile["distillation_loss"] > 0.0


def test_self_play_cli_evaluates_against_random_every_ten_iterations(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["train.py", "--self-play-multi"])

    arguments = train_mod._parse_args()

    assert arguments.evaluate_every == 10


def test_training_resumes_from_full_checkpoint_before_next_iteration(monkeypatch, tmp_path):
    full_path = tmp_path / "multi_checkpoint_iter_4000.pt"
    full_path.touch()
    loaded_paths = []
    prepared_iterations = []
    agent = SimpleNamespace(
        iteration_count=0,
        num_players=6,
        optimizer=SimpleNamespace(param_groups=[{"lr": 1e-4}]),
        advantage_buffer=[],
        strategy_buffer=[],
        load_model=lambda path: (
            loaded_paths.append(path),
            setattr(agent, "iteration_count", 4000),
        ),
        prepare_iteration=lambda iteration, **_kwargs: prepared_iterations.append(iteration),
        reset_traversal_stats=lambda: None,
        cfr_traverse_multi=lambda *_args, **_kwargs: None,
        train_advantage_network_multi=lambda: 0.0,
        train_strategy_network=lambda: 0.0,
        set_opponent_strategy_states_by_player=lambda *_args, **_kwargs: None,
        strategy_net=SimpleNamespace(state_dict=lambda: {"weight": torch.tensor([1.0])}),
    )
    _add_traversal_recorders(agent)
    monkeypatch.setattr(train_mod, "DeepCFRAgent", lambda **_kwargs: agent)
    monkeypatch.setattr(train_mod, "_configure_strategy_opponent_pool", lambda *_args, **_kwargs: [])
    monkeypatch.setattr(train_mod, "_new_hand", lambda *_args: None)
    monkeypatch.setattr(train_mod, "_log_multi_cfr_diagnostics", lambda *_args: {})
    monkeypatch.setattr(train_mod, "_print_opponent_checkpoints", lambda *_args: None)

    train_mod.train_self_play_multi(
        num_iterations=1,
        traversals_per_iteration=1,
        evaluate_every=0,
        save_dir=tmp_path,
        initial_checkpoint=str(full_path),
    )

    assert loaded_paths == [str(full_path)]
    assert prepared_iterations == [4001]


def test_resume_can_save_to_clean_dir_while_reading_opponents_from_source_dir(monkeypatch, tmp_path):
    source_dir = tmp_path / "source"
    output_dir = tmp_path / "output"
    source_dir.mkdir()
    output_dir.mkdir()
    checkpoint_path = source_dir / "multi_checkpoint_iter_5000.pt"
    torch.save(
        {
            "iteration": 5000,
            "action_space_version": ACTION_SPACE_VERSION,
            "strategy_net": {"weight": torch.tensor([1.0])},
        },
        checkpoint_path,
    )
    checkpoint_dirs = []
    saved_dirs = []

    class ResumeAgent:
        iteration_count = 0
        num_players = 6
        optimizer = SimpleNamespace(param_groups=[{"lr": 1e-4}])
        advantage_buffer = []
        strategy_buffer = []
        strategy_net = SimpleNamespace(state_dict=lambda: {"weight": torch.tensor([0.0])})

        def load_model(self, path):
            self.iteration_count = 5000
            return {"strategy_net": {"weight": torch.tensor([1.0])}}

        def set_opponent_strategy_states_by_player(self, *_args, **_kwargs):
            pass

        def prepare_iteration(self, *_args, **_kwargs):
            pass

        def reset_traversal_stats(self):
            pass

        def cfr_traverse_multi(self, *_args, **_kwargs):
            pass

        def train_advantage_network_multi(self):
            return 0.0

        def train_strategy_network(self):
            return 0.0

        def _build_checkpoint(self, **_kwargs):
            return {"iteration": 6000}

        def build_light_checkpoint(self, **_kwargs):
            return {"iteration": 6000}

    agent = ResumeAgent()
    _add_traversal_recorders(agent)
    monkeypatch.setattr(train_mod, "DeepCFRAgent", lambda **_kwargs: agent)
    monkeypatch.setattr(train_mod, "_create_writer", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(train_mod, "_new_hand", lambda *_args: None)
    monkeypatch.setattr(train_mod, "_log_multi_cfr_diagnostics", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(train_mod, "_configure_strategy_opponent_pool", lambda *_args, **_kwargs: [])
    monkeypatch.setattr(train_mod, "_checkpoint_save_due", lambda *_args, **_kwargs: True)

    class TrackingSchedule:
        def __init__(self, checkpoint_dir, **_kwargs):
            checkpoint_dirs.append(Path(checkpoint_dir))

        def paths_for_iteration(self, *_args, **_kwargs):
            return [source_dir / f"multi_checkpoint_iter_{slot}.pt" for slot in range(1, 6)]

    monkeypatch.setattr(train_mod, "OpponentPoolSchedule", TrackingSchedule)

    def record_save(agent_arg, save_dir, iteration, **_kwargs):
        del agent_arg, iteration
        saved_dirs.append(Path(save_dir))
        return Path(save_dir) / "checkpoint.pt"

    monkeypatch.setattr(train_mod, "_save_iteration_checkpoint", record_save)
    monkeypatch.setattr(train_mod, "_save_iteration_light_checkpoint", record_save)

    train_mod.train_self_play_multi(
        num_iterations=1,
        traversals_per_iteration=0,
        evaluate_every=0,
        save_dir=output_dir,
        initial_checkpoint=str(checkpoint_path),
        opponent_checkpoint_dir=source_dir,
    )

    assert checkpoint_dirs == [source_dir]
    assert saved_dirs == [output_dir, output_dir]


def test_training_resume_reuses_initial_checkpoint_strategy_state(monkeypatch, tmp_path):
    checkpoint_path = tmp_path / "multi_checkpoint_iter_1000.pt"
    source_agent = DeepCFRAgent(player_id=0, num_players=6)
    torch.save({
        "iteration": 1000,
        "strategy_net": source_agent.strategy_net.state_dict(),
    }, checkpoint_path)
    loaded_paths = []
    received_states = []
    original_load = train_mod.torch.load
    original_configure_pool = train_mod._configure_strategy_opponent_pool

    class ResumeAgent:
        iteration_count = 0
        num_players = 6
        optimizer = SimpleNamespace(param_groups=[{"lr": 1e-4}])
        advantage_buffer = []
        strategy_buffer = []
        strategy_net = source_agent.strategy_net

        def load_model(self, path):
            checkpoint = train_mod.torch.load(path, map_location="cpu", weights_only=False)
            self.iteration_count = int(checkpoint["iteration"])
            return checkpoint

        def set_opponent_strategy_states_by_player(self, states, traversing_player):
            received_states.extend(states.values())
            assert traversing_player == 0

        def prepare_iteration(self, *_args, **_kwargs):
            pass

        def reset_traversal_stats(self):
            pass

        def cfr_traverse_multi(self, *_args, **_kwargs):
            pass

        def train_advantage_network_multi(self):
            return 0.0

        def train_strategy_network(self):
            return 0.0

        def get_traversal_stats(self):
            return {"nodes": 0, "max_depth": 0}

    agent = ResumeAgent()
    _add_traversal_recorders(agent)

    def tracked_load(path, *args, **kwargs):
        loaded_paths.append(Path(path))
        return original_load(path, *args, **kwargs)

    def configure_pool(*args, **kwargs):
        return original_configure_pool(*args, **kwargs)

    monkeypatch.setattr(train_mod, "DeepCFRAgent", lambda **_kwargs: agent)
    monkeypatch.setattr(train_mod.torch, "load", tracked_load)
    monkeypatch.setattr(train_mod, "_new_hand", lambda *_args: None)
    monkeypatch.setattr(train_mod, "_log_multi_cfr_diagnostics", lambda *_args: {})
    monkeypatch.setattr(train_mod, "_print_opponent_checkpoints", lambda *_args: None)
    monkeypatch.setattr(train_mod, "_configure_strategy_opponent_pool", configure_pool)

    train_mod.train_self_play_multi(
        num_iterations=1,
        traversals_per_iteration=1,
        evaluate_every=0,
        save_dir=tmp_path,
        initial_checkpoint=str(checkpoint_path),
    )

    assert loaded_paths == [checkpoint_path]
    assert len(received_states) == 5
    assert all(state["action_head.bias"].shape == torch.Size([6]) for state in received_states)


def test_training_configures_current_strategy_pool_during_warmup(monkeypatch, tmp_path):
    calls = []

    class DummyWriter:
        def add_scalar(self, *_args, **_kwargs):
            pass

        def flush(self):
            pass

        def close(self):
            pass

    agent = SimpleNamespace(
        iteration_count=0,
        num_players=6,
        optimizer=SimpleNamespace(param_groups=[{"lr": 1e-4}]),
        advantage_buffer=[],
        strategy_buffer=[],
        strategy_net=SimpleNamespace(state_dict=lambda: {"weight": torch.tensor([1.0])}),
        prepare_iteration=lambda *_args, **_kwargs: None,
        reset_traversal_stats=lambda: None,
        get_traversal_stats=lambda: {"advantage_loss": 0.0, "strategy_loss": 0.0},
        cfr_traverse_multi=lambda *_args, **_kwargs: None,
        train_advantage_network_multi=lambda: 0.0,
        train_strategy_network=lambda: 0.0,
    )

    def fake_cfg_get(key, default=None):
        if key == "checkpoint_save_every":
            return 1000
        return default

    monkeypatch.setattr(train_mod, "DeepCFRAgent", lambda **_kwargs: agent)
    monkeypatch.setattr(train_mod, "_new_hand", lambda *_args: None)
    monkeypatch.setattr(train_mod, "_create_writer", lambda *_args, **_kwargs: DummyWriter())
    monkeypatch.setattr(
        train_mod,
        "_configure_strategy_opponent_pool",
        lambda *_args, **_kwargs: calls.append("configure") or [],
    )
    monkeypatch.setattr(train_mod, "_print_opponent_checkpoints", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(train_mod, "cfg_get", fake_cfg_get)
    monkeypatch.setattr(train_mod, "_log_multi_cfr_diagnostics", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(train_mod, "_latest_strategy_checkpoint_path", lambda *_args: tmp_path / "current.pt")

    train_mod.train_deep_cfr(
        num_iterations=1,
        traversals_per_iteration=0,
        save_dir=tmp_path,
        log_dir=tmp_path / "logs",
        seed=1,
    )

    assert calls == ["configure"]


def test_strategy_opponent_pool_uses_live_strategy_when_checkpoint_is_absent():
    strategy_state = {"weight": torch.tensor([1.0])}
    received_states = {}
    agent = SimpleNamespace(
        strategy_net=SimpleNamespace(state_dict=lambda: strategy_state),
        set_opponent_strategy_states_by_player=lambda states, _traversing_player: received_states.update(states),
    )

    selected_paths = train_mod._configure_strategy_opponent_pool(
        agent,
        checkpoint_by_player={},
        strategy_positions=(1, 2, 3, 4, 5),
        current_strategy_path=None,
        traversing_player=0,
        state_cache={},
    )

    assert selected_paths == []
    assert set(received_states) == {1, 2, 3, 4, 5}
    assert all(state is strategy_state for state in received_states.values())


def test_training_uses_current_strategy_without_per_traversal_progress(monkeypatch, capsys, tmp_path):
    opponents = []
    agent = SimpleNamespace(
        iteration_count=0,
        num_players=6,
        optimizer=SimpleNamespace(param_groups=[{"lr": 1e-4}]),
        advantage_buffer=[],
        strategy_buffer=[],
        strategy_net=SimpleNamespace(state_dict=lambda: {"weight": torch.tensor([1.0])}),
        prepare_iteration=lambda *_args, **_kwargs: None,
        reset_traversal_stats=lambda: None,
        cfr_traverse_multi=lambda *_args, **kwargs: opponents.append(kwargs.get("random_agent")),
        train_advantage_network_multi=lambda: 0.0,
        train_strategy_network=lambda: 0.0,
    )
    _add_traversal_recorders(agent)
    monkeypatch.setattr(train_mod, "DeepCFRAgent", lambda **_kwargs: agent)
    monkeypatch.setattr(train_mod, "_configure_strategy_opponent_pool", lambda *_args, **_kwargs: [])
    monkeypatch.setattr(train_mod, "_new_hand", lambda *_args: None)
    monkeypatch.setattr(train_mod, "_log_multi_cfr_diagnostics", lambda *_args: {})
    monkeypatch.setattr(
        train_mod,
        "_latest_strategy_checkpoint_path",
        lambda *_args: pytest.fail("На старте без checkpoint нужно использовать живую strategy_net"),
    )

    train_mod.train_self_play_multi(
        num_iterations=1,
        traversals_per_iteration=5,
        evaluate_every=0,
        save_dir=tmp_path,
    )

    output = capsys.readouterr().out
    assert "Запускаю 5 обходов" in output
    assert "Обходы:" not in output
    assert len(opponents) == 5
    assert opponents == [None] * 5


@pytest.mark.parametrize(
    ("iteration", "expected"),
    [(1, 5), (1000, 5), (1001, 4), (2001, 3), (3001, 2), (4001, 1), (5001, 0), (9001, 0)],
)
def test_current_strategy_opponent_count_fades_out_after_five_checkpoints(iteration, expected):
    assert _current_strategy_opponent_count(iteration, checkpoint_every=1000) == expected


def test_training_rotates_five_checkpoint_opponents_after_strategy_fades_out(monkeypatch, tmp_path):
    configured_states = {}
    for iteration in range(1000, 5001, 1000):
        torch.save(
            {
                "action_space_version": ACTION_SPACE_VERSION,
                "strategy_net": {"weight": torch.tensor([float(iteration)])},
            },
            tmp_path / f"multi_checkpoint_iter_{iteration}.pt",
        )

    agent = SimpleNamespace(
        iteration_count=5000,
        num_players=6,
        optimizer=SimpleNamespace(param_groups=[{"lr": 1e-4}]),
        advantage_buffer=[],
        strategy_buffer=[],
        strategy_net=SimpleNamespace(state_dict=lambda: {"weight": torch.tensor([0.0])}),
        set_opponent_strategy_states_by_player=lambda states, _traversing_player: configured_states.update(states),
        prepare_iteration=lambda *_args, **_kwargs: None,
        reset_traversal_stats=lambda: None,
        cfr_traverse_multi=lambda *_args, **_kwargs: None,
        train_advantage_network_multi=lambda: 0.0,
        train_strategy_network=lambda: 0.0,
    )
    _add_traversal_recorders(agent)
    monkeypatch.setattr(train_mod, "DeepCFRAgent", lambda **_kwargs: agent)
    monkeypatch.setattr(train_mod, "_create_writer", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(train_mod, "_log_multi_cfr_diagnostics", lambda *_args, **_kwargs: {})

    train_mod.train_self_play_multi(
        num_iterations=1,
        traversals_per_iteration=0,
        evaluate_every=0,
        save_dir=tmp_path,
    )

    assert set(configured_states) == {1, 2, 3, 4, 5}
    assert len(configured_states) == 5


def test_current_strategy_uses_latest_prior_heavy_checkpoint(tmp_path):
    for iteration in (50_000, 100_000, 150_000):
        (tmp_path / f"multi_checkpoint_iter_{iteration}.pt").touch()

    path = _latest_strategy_checkpoint_path(iteration=150_001, checkpoint_dir=tmp_path)

    assert path.name == "multi_checkpoint_iter_150000.pt"


@pytest.mark.parametrize(
    ("iteration", "expected"),
    [(1001, [1000]), (2001, [2000, 1000]), (3001, [3000, 2000, 1000]),
     (4001, [4000, 3000, 2000, 1000]), (5001, [])],
)
def test_transition_uses_newest_own_checkpoints_before_permanent_legacy_pool(iteration, expected):
    assert _transition_checkpoint_iterations(iteration, checkpoint_every=1000) == expected


def test_checkpoint_assignment_shuffles_current_strategy_and_checkpoint_positions():
    checkpoint_paths = [Path("our_2000.pt"), Path("our_1000.pt")]

    strategy_positions, checkpoint_by_player = _assign_checkpoint_opponents(
        traversing_player=0,
        num_players=6,
        strategy_count=3,
        checkpoint_paths=checkpoint_paths,
    )

    assert len(strategy_positions) == 3
    assert set(strategy_positions).isdisjoint(checkpoint_by_player)
    assert set(strategy_positions) | set(checkpoint_by_player) == {1, 2, 3, 4, 5}
    assert set(checkpoint_by_player.values()) == set(checkpoint_paths)


@pytest.mark.parametrize(
    ("current_player", "expected"),
    [(0, False), (1, True), (2, True), (3, True), (4, True), (5, True)],
)
def test_random_opponent_policy_covers_all_nontraversing_positions(current_player, expected):
    assert DeepCFRAgent._is_random_opponent_turn(
        random_agent=object(), current_player=current_player, traversing_player=0
    ) is expected


def test_training_stops_passing_random_agent_after_thousandth_checkpoint(monkeypatch, tmp_path):
    random_agents = []
    agent = SimpleNamespace(
        iteration_count=1000,
        num_players=6,
        optimizer=SimpleNamespace(param_groups=[{"lr": 1e-4}]),
        advantage_buffer=[],
        strategy_buffer=[],
        prepare_iteration=lambda *_args, **_kwargs: None,
        reset_traversal_stats=lambda: None,
        cfr_traverse_multi=lambda *_args, **kwargs: random_agents.append(kwargs.get("random_agent")),
        train_advantage_network_multi=lambda: 0.0,
        train_strategy_network=lambda: 0.0,
    )
    _add_traversal_recorders(agent)
    monkeypatch.setattr(train_mod, "DeepCFRAgent", lambda **_kwargs: agent)
    checkpoint_path = tmp_path / "multi_checkpoint_iter_1000.pt"
    monkeypatch.setattr(train_mod, "_heavy_checkpoints", lambda *_args, **_kwargs: {1000: checkpoint_path})
    monkeypatch.setattr(train_mod, "_configure_strategy_opponent_pool", lambda *_args, **_kwargs: [checkpoint_path])
    monkeypatch.setattr(train_mod, "_new_hand", lambda *_args: None)
    monkeypatch.setattr(train_mod, "_log_multi_cfr_diagnostics", lambda *_args: {})
    monkeypatch.setattr(train_mod, "_print_opponent_checkpoints", lambda *_args: None)

    train_mod.train_self_play_multi(
        num_iterations=1,
        traversals_per_iteration=1,
        evaluate_every=0,
        save_dir=tmp_path,
    )

    assert random_agents == [None]


def test_traversal_thread_limit_restores_torch_threads_after_error(monkeypatch):
    calls = []
    monkeypatch.setattr("src.training.train.torch.get_num_threads", lambda: 6)
    monkeypatch.setattr("src.training.train.torch.set_num_threads", calls.append)

    with pytest.raises(RuntimeError, match="проверка"):
        with _traversal_thread_limit(True):
            raise RuntimeError("проверка")

    assert calls == [1, 6]


def test_training_thread_limit_restores_torch_threads_after_error(monkeypatch):
    calls = []
    monkeypatch.setattr("src.training.train.torch.get_num_threads", lambda: 6)
    monkeypatch.setattr("src.training.train.torch.set_num_threads", calls.append)

    with pytest.raises(RuntimeError, match="проверка"):
        with _training_thread_limit(12):
            raise RuntimeError("проверка")

    assert calls == [12, 6]


def test_above_normal_process_priority_is_applied_on_windows(monkeypatch, capsys):
    calls = []
    process = SimpleNamespace(nice=calls.append)
    psutil_stub = SimpleNamespace(
        ABOVE_NORMAL_PRIORITY_CLASS=32768,
        Error=RuntimeError,
        Process=lambda: process,
    )
    monkeypatch.setattr("src.training.train.os.name", "nt")
    monkeypatch.setitem(sys.modules, "psutil", psutil_stub)

    _apply_process_priority("above_normal")

    assert calls == [32768]
    assert "ПРИМЕНЁН (above_normal)" in capsys.readouterr().out


def test_process_priority_rejects_unknown_value():
    with pytest.raises(ValueError, match="process_priority"):
        _apply_process_priority("high")


def test_training_logs_nodes_and_max_depth_to_console(capsys):
    agent = SimpleNamespace(
        get_traversal_stats=lambda: {"nodes": 1234, "max_depth": 17},
    )

    result = _log_multi_cfr_diagnostics(agent, None, 1, 0, 10)

    assert result == {"nodes": 1234, "max_depth": 17}
    assert "Обходы: узлы=1234, макс_глубина=17" in capsys.readouterr().out


def test_training_logs_loaded_opponent_checkpoints_to_console(capsys, tmp_path):
    paths = [
        tmp_path / "multi_checkpoint_iter_150.pt",
        tmp_path / "multi_checkpoint_iter_163.pt",
    ]

    _print_opponent_checkpoints(paths)

    output = capsys.readouterr().out
    assert "Играет против:" in output
    assert "multi_checkpoint_iter_150.pt" in output
    assert "multi_checkpoint_iter_163.pt" in output


def test_training_logs_cold_start_when_opponent_checkpoints_are_missing(capsys):
    _print_opponent_checkpoints([])

    assert "холодный старт" in capsys.readouterr().out


def test_training_logs_random_opponents_during_cold_start(capsys):
    _print_opponent_checkpoints([], random_opponents=5)

    assert "Играет против: 5 RandomAgent" in capsys.readouterr().out


def test_agent_policy_and_checkpoint_use_six_fixed_actions(tmp_path):
    state = pkrs.State.from_seed(
        n_players=2, button=0, sb=1.0, bb=2.0, stake=200.0, seed=11
    )
    agent = DeepCFRAgent(player_id=0, num_players=2)
    probabilities = agent.get_policy_distribution(state, player_id=int(state.current_player))
    assert probabilities.shape == (NUM_ACTIONS,)
    assert probabilities.sum() == 1.0

    checkpoint = tmp_path / "six_actions.pt"
    agent.save_model(str(checkpoint))
    payload = agent._load_checkpoint(checkpoint)
    assert payload["action_space_version"] == ACTION_SPACE_VERSION
    assert payload["num_actions"] == NUM_ACTIONS
    assert "advantage_optimizer" in payload
    assert "strategy_optimizer" in payload


def test_history_summary_v3_agent_persists_encoder_contract(tmp_path, monkeypatch):
    import src.core.deep_cfr as deep_cfr_module

    original_cfg_get = deep_cfr_module.cfg_get

    def cfg_get_v3(key, default=None):
        if key == "encoding_version":
            return "history_summary_v3"
        return original_cfg_get(key, default)

    monkeypatch.setattr(deep_cfr_module, "cfg_get", cfg_get_v3)
    agent = DeepCFRAgent(player_id=0, num_players=2, device="cpu")
    state = pkrs.State.from_seed(
        n_players=2, button=0, sb=1.0, bb=2.0, stake=200.0, seed=11
    )

    assert agent.input_size == 181
    assert agent._encode_state(state, int(state.current_player)).shape == (181,)

    checkpoint = agent._build_checkpoint()

    assert checkpoint["checkpoint_format_version"] == CHECKPOINT_FORMAT_VERSION
    assert checkpoint["encoding_version"] == "history_summary_v3"
    assert checkpoint["encoder_input_size"] == 181
    assert checkpoint["config"]["encoding_version"] == "history_summary_v3"


def test_checkpoint_rejects_different_player_count_before_loading_weights(tmp_path):
    agent = DeepCFRAgent(player_id=0, num_players=2, device="cpu")
    payload = agent._build_checkpoint()
    payload["num_players"] = 6
    checkpoint = tmp_path / "wrong_player_count.pt"
    torch.save(payload, checkpoint)

    with pytest.raises(ValueError, match="число игроков"):
        agent._load_checkpoint(checkpoint)


def test_teacher_checkpoint_rejects_wrong_declared_encoder_input_size(tmp_path):
    teacher = DeepCFRAgent(
        player_id=0,
        num_players=2,
        device="cpu",
        network_architecture=MONOLITHIC_ARCHITECTURE,
    )
    payload = teacher.build_light_checkpoint()
    payload["encoder_input_size"] = 0
    checkpoint = tmp_path / "wrong_teacher_input.pt"
    torch.save(payload, checkpoint)
    student = DeepCFRAgent(
        player_id=0,
        num_players=6,
        device="cpu",
        network_architecture=MONOLITHIC_ARCHITECTURE,
    )

    with pytest.raises(ValueError, match="размер входа encoder"):
        student.load_teacher_strategy_checkpoint(checkpoint)


def test_advantage_training_steps_override_epoch_budget():
    agent = DeepCFRAgent(player_id=0, num_players=2)
    agent.advantage_train_steps = 5
    agent.advantage_epochs = 1
    agent.iteration_count = 2
    for _ in range(3):
        agent.advantage_buffer.add(
            np.zeros(agent.input_size, dtype=np.float32),
            np.zeros(NUM_ACTIONS, dtype=np.float32),
            np.ones(NUM_ACTIONS, dtype=np.float32),
            2,
        )

    agent.train_advantage_network_multi(batch_size=2)

    assert agent.last_advantage_profile["actual_steps"] == 5
    assert agent.last_advantage_profile["expected_steps"] == 5


def test_strategy_training_steps_override_epoch_budget():
    agent = DeepCFRAgent(player_id=0, num_players=2)
    agent.strategy_train_steps = 7
    agent.strategy_epochs = 1
    agent.iteration_count = 2
    for _ in range(3):
        agent.strategy_buffer.add(
            np.zeros(agent.input_size, dtype=np.float32),
            np.full(NUM_ACTIONS, 1.0 / NUM_ACTIONS, dtype=np.float32),
            np.ones(NUM_ACTIONS, dtype=np.float32),
            2,
        )

    agent.train_strategy_network(batch_size=2)

    assert agent.last_strategy_profile["actual_steps"] == 7
    assert agent.last_strategy_profile["expected_steps"] == 7


def test_strategy_discount_weights_are_normalized_in_batch():
    agent = DeepCFRAgent(player_id=0, num_players=2)
    agent.strategy_train_steps = None
    agent.strategy_epochs = 1
    agent.discount_gamma = 1.0
    agent.iteration_count = 2
    for group in agent.strategy_optimizer.param_groups:
        group["lr"] = 0.0
    with torch.no_grad():
        for parameter in agent.strategy_net.parameters():
            parameter.zero_()
    mask = np.ones(NUM_ACTIONS, dtype=np.float32)
    agent.strategy_buffer.add(
        np.zeros(agent.input_size, dtype=np.float32),
        np.full(NUM_ACTIONS, 1.0 / NUM_ACTIONS, dtype=np.float32),
        mask,
        1,
    )
    policy = np.zeros(NUM_ACTIONS, dtype=np.float32)
    policy[0] = 1.0
    agent.strategy_buffer.add(
        np.zeros(agent.input_size, dtype=np.float32),
        policy,
        mask,
        2,
    )

    loss = agent.train_strategy_network(batch_size=2)

    assert loss == pytest.approx(5.0 / 9.0)


def test_dcfr_plus_advantage_discount_uses_unit_denominator():
    agent = DeepCFRAgent(player_id=0, num_players=2)
    agent.advantage_train_steps = 1
    agent.discount_alpha = 2.0
    agent.iteration_count = 2
    for group in agent.optimizer.param_groups:
        group["lr"] = 0.0
    with torch.no_grad():
        for network in (agent.advantage_net, agent.advantage_target_net):
            for parameter in network.parameters():
                parameter.zero_()
        agent.advantage_target_net.action_head.bias[0] = 1.0
    mask = np.zeros(NUM_ACTIONS, dtype=np.float32)
    mask[0] = 1.0
    agent.advantage_buffer.add(
        np.zeros(agent.input_size, dtype=np.float32),
        np.zeros(NUM_ACTIONS, dtype=np.float32),
        mask,
        2,
    )

    loss = agent.train_advantage_network_multi(batch_size=1)

    assert loss == pytest.approx((0.5 ** 2) / NUM_ACTIONS)


def test_advantage_training_stops_before_optimizer_step_on_nonfinite_loss(monkeypatch):
    agent = DeepCFRAgent(player_id=0, num_players=2, device="cpu")
    agent.advantage_train_steps = 1
    agent.advantage_buffer.add(
        np.zeros(agent.input_size, dtype=np.float32),
        np.full(NUM_ACTIONS, np.nan, dtype=np.float32),
        np.ones(NUM_ACTIONS, dtype=np.float32),
        1,
    )

    monkeypatch.setattr(
        agent.optimizer,
        "step",
        lambda: pytest.fail("optimizer.step не должен вызываться при NaN loss"),
    )

    with pytest.raises(FloatingPointError, match="сети преимуществ"):
        agent.train_advantage_network_multi(batch_size=1)

    assert all(parameter.grad is None for parameter in agent.advantage_net.parameters())


def test_strategy_training_stops_before_optimizer_step_on_nonfinite_loss(monkeypatch):
    agent = DeepCFRAgent(player_id=0, num_players=2, device="cpu")
    agent.strategy_train_steps = 1
    agent.strategy_buffer.add(
        np.zeros(agent.input_size, dtype=np.float32),
        np.full(NUM_ACTIONS, np.nan, dtype=np.float32),
        np.ones(NUM_ACTIONS, dtype=np.float32),
        1,
    )

    monkeypatch.setattr(
        agent.strategy_optimizer,
        "step",
        lambda: pytest.fail("optimizer.step не должен вызываться при NaN loss"),
    )

    with pytest.raises(FloatingPointError, match="сети стратегии"):
        agent.train_strategy_network(batch_size=1)

    assert all(parameter.grad is None for parameter in agent.strategy_net.parameters())


def test_advantage_training_stops_before_optimizer_step_on_nonfinite_gradient(monkeypatch):
    agent = DeepCFRAgent(player_id=0, num_players=2, device="cpu")
    agent.advantage_train_steps = 1
    agent.advantage_buffer.add(
        np.zeros(agent.input_size, dtype=np.float32),
        np.zeros(NUM_ACTIONS, dtype=np.float32),
        np.ones(NUM_ACTIONS, dtype=np.float32),
        1,
    )
    hook = next(agent.advantage_net.parameters()).register_hook(
        lambda gradient: torch.full_like(gradient, torch.nan)
    )
    monkeypatch.setattr(
        agent.optimizer,
        "step",
        lambda: pytest.fail("optimizer.step не должен вызываться при NaN градиенте"),
    )

    try:
        with pytest.raises(FloatingPointError, match="сети преимуществ"):
            agent.train_advantage_network_multi(batch_size=1)
    finally:
        hook.remove()

    assert all(parameter.grad is None for parameter in agent.advantage_net.parameters())


def test_checkpoint_opponent_strategy_uses_strategy_net_not_advantage_net(monkeypatch):
    state = pkrs.State.from_seed(
        n_players=2, button=0, sb=1.0, bb=2.0, stake=200.0, seed=11
    )
    agent = DeepCFRAgent(player_id=0, num_players=2, device="cpu")
    agent.get_legal_action_mask = lambda _state: torch.tensor(
        [1, 1, 0, 0, 0, 0], dtype=torch.float32
    ).numpy()
    chosen_slots = []
    agent.action_type_to_pokers_action = lambda slot, _state: (
        chosen_slots.append(int(slot)),
        pkrs.Action(pkrs.ActionEnum.Fold),
    )[1]
    with torch.no_grad():
        for network in (agent.advantage_net, agent.strategy_net):
            for parameter in network.parameters():
                parameter.zero_()
        agent.strategy_net.action_head.bias[1] = 10.0

    agent.set_opponent_strategy_states(
        [agent.strategy_net.state_dict()], traversing_player=0
    )
    monkeypatch.setattr(
        np.random,
        "choice",
        lambda choices, p=None: choices[int(np.argmax(p))],
    )

    agent.cfr_traverse_multi(state, iteration=1, traversing_player=0)

    assert chosen_slots == [1]
    assert agent.action_decision_count == 1
