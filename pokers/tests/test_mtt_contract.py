import pytest

import pokers as pkrs


def test_from_seed_accepts_per_seat_stacks_in_chip_units():
    state = pkrs.State.from_seed(
        n_players=3,
        button=0,
        sb=1.0,
        bb=2.0,
        stake=999.0,
        stakes=[31.0, 17.0, 5.0],
        chip_unit=1.0,
        seed=7,
    )

    assert [player.stake for player in state.players_state] == [31.0, 16.0, 3.0]
    assert [player.bet_chips for player in state.players_state] == [0.0, 1.0, 2.0]
    assert state.pot == 3.0
    assert state.current_player == 0


def test_short_blinds_keep_nominal_big_blind_in_multiway_hand():
    state = pkrs.State.from_seed(
        n_players=3,
        button=2,
        sb=1.0,
        bb=2.0,
        stake=999.0,
        stakes=[70.0, 1.0, 186.0],
        chip_unit=1.0,
        seed=11,
    )

    assert [player.stake for player in state.players_state] == [69.0, 0.0, 186.0]
    assert [player.bet_chips for player in state.players_state] == [1.0, 1.0, 0.0]
    assert state.pot == 2.0
    assert state.min_bet == 2.0
    assert state.min_raise == 2.0
    assert state.current_player == 2


@pytest.mark.parametrize("value", [0.3, float("inf"), -1.0])
def test_constructor_rejects_amounts_outside_chip_unit_contract(value):
    with pytest.raises((TypeError, ValueError, OSError)):
        pkrs.State.from_seed(
            n_players=2,
            button=0,
            sb=1.0,
            bb=2.0,
            stake=value,
            chip_unit=1.0,
            seed=3,
        )


def test_mtt_contract_rejects_more_than_eight_seats():
    with pytest.raises((TypeError, ValueError, OSError)):
        pkrs.State.from_seed(
            n_players=9,
            button=0,
            sb=1.0,
            bb=2.0,
            stake=20.0,
            chip_unit=1.0,
            seed=3,
        )


def test_ante_is_paid_before_blinds_without_becoming_a_street_bet():
    state = pkrs.State.from_seed(
        n_players=3,
        button=0,
        sb=1.0,
        bb=2.0,
        stake=99.0,
        stakes=[3.0, 2.0, 2.0],
        ante=1.0,
        chip_unit=1.0,
        seed=17,
    )

    assert [player.stake for player in state.players_state] == [2.0, 0.0, 0.0]
    assert [player.pot_chips for player in state.players_state] == [1.0, 1.0, 1.0]
    assert [player.bet_chips for player in state.players_state] == [0.0, 1.0, 1.0]
    assert state.pot == 5.0
    assert state.min_bet == 1.0


def test_public_history_contains_only_successful_public_actions():
    state = pkrs.State.from_seed(
        n_players=2,
        button=0,
        sb=1.0,
        bb=2.0,
        stake=20.0,
        chip_unit=1.0,
        seed=19,
    )

    illegal = state.apply_action(pkrs.Action(pkrs.ActionEnum.Check))
    assert illegal.status == pkrs.StateStatus.IllegalAction
    assert illegal.action_history == []

    called = state.apply_action(pkrs.Action(pkrs.ActionEnum.Call))
    assert len(called.action_history) == 1
    record = called.action_history[0]
    assert record.actor_id == state.current_player
    assert record.street == pkrs.Stage.Preflop
    assert record.requested_action.action == pkrs.ActionEnum.Call
    assert record.paid_amount == 1.0
    assert record.applied_raise_increment == 0.0
    assert record.is_effective_raise is False
