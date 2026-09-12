import numpy as np
import pokers as pkrs
import pytest
import subprocess
import sys
from pathlib import Path

from src.core.action_space import ActionSlot, legal_action_mask, resolve_action
from src.core.model import (
    encoder_input_size,
    encode_state,
    encode_state_history_summary_v3,
    history_summary_size,
    legacy_base_input_size,
)
from tools.audit_information_set_aliasing import audit_information_set_aliasing


def _initial_hu_state() -> pkrs.State:
    return pkrs.State.from_seed(
        n_players=2,
        button=0,
        sb=1.0,
        bb=2.0,
        stake=30.0,
        seed=17,
    )


def _apply_slot(state: pkrs.State, slot: ActionSlot) -> pkrs.State:
    resolved_action = resolve_action(slot, state)
    next_state = state.apply_action(resolved_action.action)
    assert next_state.status == pkrs.StateStatus.Ok
    return next_state


def _flop_raise_vs_turn_raise() -> tuple[pkrs.State, pkrs.State]:
    turn_raise = _initial_hu_state()
    for slot in (
        ActionSlot.CALL,
        ActionSlot.CHECK,
        ActionSlot.CHECK,
        ActionSlot.CHECK,
        ActionSlot.CHECK,
        ActionSlot.RAISE_HALF_POT,
        ActionSlot.CALL,
        ActionSlot.CHECK,
    ):
        turn_raise = _apply_slot(turn_raise, slot)

    flop_raise = _initial_hu_state()
    for slot in (
        ActionSlot.CALL,
        ActionSlot.CHECK,
        ActionSlot.CHECK,
        ActionSlot.RAISE_HALF_POT,
        ActionSlot.CALL,
        ActionSlot.CHECK,
        ActionSlot.CHECK,
        ActionSlot.CHECK,
    ):
        flop_raise = _apply_slot(flop_raise, slot)

    return flop_raise, turn_raise


def _turn_aggressor_player_zero_vs_one() -> tuple[pkrs.State, pkrs.State]:
    player_zero_aggressor = _initial_hu_state()
    for slot in (
        ActionSlot.CALL,
        ActionSlot.CHECK,
        ActionSlot.CHECK,
        ActionSlot.CHECK,
        ActionSlot.CHECK,
        ActionSlot.RAISE_HALF_POT,
        ActionSlot.CALL,
    ):
        player_zero_aggressor = _apply_slot(player_zero_aggressor, slot)

    player_one_aggressor = _initial_hu_state()
    for slot in (
        ActionSlot.CALL,
        ActionSlot.CHECK,
        ActionSlot.CHECK,
        ActionSlot.CHECK,
        ActionSlot.RAISE_HALF_POT,
        ActionSlot.CALL,
    ):
        player_one_aggressor = _apply_slot(player_one_aggressor, slot)

    return player_zero_aggressor, player_one_aggressor


def _assert_current_alias(first: pkrs.State, second: pkrs.State) -> None:
    assert first.stage == pkrs.Stage.River
    assert second.stage == pkrs.Stage.River
    assert np.array_equal(legal_action_mask(first), legal_action_mask(second))

    first_encoding = encode_state(first, player_id=0)
    second_encoding = encode_state(second, player_id=0)
    assert first_encoding.shape == (133,)
    assert second_encoding.shape == (133,)
    assert np.array_equal(first_encoding, second_encoding)


@pytest.mark.parametrize(
    "state_factory",
    (_flop_raise_vs_turn_raise, _turn_aggressor_player_zero_vs_one),
    ids=("street-of-aggression", "turn-aggressor"),
)
def test_current_encoder_reproduces_documented_hu_aliases(state_factory) -> None:
    first, second = state_factory()

    _assert_current_alias(first, second)


@pytest.mark.parametrize(
    "state_factory",
    (_flop_raise_vs_turn_raise, _turn_aggressor_player_zero_vs_one),
    ids=("street-of-aggression", "turn-aggressor"),
)
def test_history_summary_must_separate_documented_hu_aliases(state_factory) -> None:
    first, second = state_factory()

    first_encoding = encode_state_history_summary_v3(first, player_id=0)
    second_encoding = encode_state_history_summary_v3(second, player_id=0)

    assert first_encoding.shape == (181,)
    assert second_encoding.shape == (181,)
    assert not np.array_equal(first_encoding, second_encoding)


def test_history_summary_rejects_incomplete_public_history() -> None:
    state = pkrs.State.from_mid_hand(
        n_players=2,
        button=0,
        sb=1.0,
        bb=2.0,
        stake=20.0,
        deck=[],
        hole_cards=[(pkrs.Card.from_string("cA"), pkrs.Card.from_string("dK")),
                    (pkrs.Card.from_string("hQ"), pkrs.Card.from_string("sJ"))],
        public_cards=[],
        stage=pkrs.Stage.Preflop,
        pot=3.0,
        bet_chips=[1.0, 2.0],
        pot_chips=[0.0, 0.0],
        active=[True, True],
        last_stage_action=[None, None],
        current_player=0,
    )

    with pytest.raises(ValueError, match="полной публичной истории"):
        encode_state_history_summary_v3(state, player_id=0)


def test_encoder_input_size_contracts_are_explicit() -> None:
    assert legacy_base_input_size(2) == 133
    assert history_summary_size(2) == 48
    assert encoder_input_size(2, "legacy_v2") == 133
    assert encoder_input_size(2, "history_summary_v3") == 181
    assert encoder_input_size(6, "history_summary_v3", use_multi_agent=True) == 243


def test_information_set_aliasing_audit_reproduces_v2_collisions_and_v3_regressions() -> None:
    audit = audit_information_set_aliasing(
        players=2,
        stack=30.0,
        seed=17,
        max_depth=12,
    )

    assert audit.decision_nodes > 0
    assert audit.legacy_v2.collision_bucket_count >= 1

    regression_pairs = (
        (
            (
                    ActionSlot.CALL,
                    ActionSlot.CHECK,
                ActionSlot.CHECK,
                ActionSlot.RAISE_HALF_POT,
                ActionSlot.CALL,
                ActionSlot.CHECK,
                ActionSlot.CHECK,
                ActionSlot.CHECK,
            ),
            (
                    ActionSlot.CALL,
                    ActionSlot.CHECK,
                ActionSlot.CHECK,
                ActionSlot.CHECK,
                ActionSlot.CHECK,
                ActionSlot.RAISE_HALF_POT,
                ActionSlot.CALL,
                ActionSlot.CHECK,
            ),
        ),
        (
            (
                    ActionSlot.CALL,
                    ActionSlot.CHECK,
                ActionSlot.CHECK,
                ActionSlot.CHECK,
                ActionSlot.CHECK,
                ActionSlot.RAISE_HALF_POT,
                ActionSlot.CALL,
            ),
            (
                    ActionSlot.CALL,
                    ActionSlot.CHECK,
                ActionSlot.CHECK,
                ActionSlot.CHECK,
                ActionSlot.RAISE_HALF_POT,
                ActionSlot.CALL,
            ),
        ),
    )
    for first_trace, second_trace in regression_pairs:
        first = audit.encoding_for_trace(first_trace)
        second = audit.encoding_for_trace(second_trace)

        assert first.legacy_v2 == second.legacy_v2
        assert first.history_summary_v3 != second.history_summary_v3


def test_information_set_aliasing_audit_excludes_folded_terminal_state() -> None:
    audit = audit_information_set_aliasing(
        players=2,
        stack=30.0,
        seed=17,
        max_depth=1,
    )

    assert (ActionSlot.FOLD,) not in audit.encodings_by_trace


@pytest.mark.parametrize("stack", (float("nan"), float("inf")))
def test_information_set_aliasing_audit_rejects_non_finite_stack(stack: float) -> None:
    with pytest.raises(ValueError):
        audit_information_set_aliasing(
            players=2,
            stack=stack,
            seed=17,
            max_depth=1,
        )


def test_information_set_aliasing_audit_cli_prints_reproducible_summary() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "tools/audit_information_set_aliasing.py",
            "--players", "2",
            "--stack", "30",
            "--seed", "17",
            "--max-depth", "12",
        ],
        check=False,
        capture_output=True,
        text=True,
        cwd=Path(__file__).parents[1],
    )

    assert result.returncode == 0, result.stderr
    assert "decision_nodes=" in result.stdout
    assert "legacy_v2: distinct_tensors=" in result.stdout
    assert "collision 1:" in result.stdout


def test_information_set_aliasing_audit_cli_rejects_non_finite_stack_without_traceback() -> None:
    result = subprocess.run(
        [sys.executable, "tools/audit_information_set_aliasing.py", "--stack", "nan"],
        check=False,
        capture_output=True,
        text=True,
        cwd=Path(__file__).parents[1],
    )

    assert result.returncode != 0
    assert "Traceback" not in result.stderr
    assert "конечным" in result.stderr
