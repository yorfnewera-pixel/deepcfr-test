"""Интеграционные проверки передачи стеков между MTT-раздачами."""

import pokers as pkrs


def _play_calls_and_checks(state):
    while not state.final_state:
        action = (
            pkrs.ActionEnum.Call
            if pkrs.ActionEnum.Call in state.legal_actions
            else pkrs.ActionEnum.Check
        )
        state = state.apply_action(pkrs.Action(action))
    return state


def test_settled_stacks_start_next_hand_without_eliminated_player():
    first_hand = _play_calls_and_checks(
        pkrs.State.from_seed(
            n_players=3,
            button=0,
            sb=1.0,
            bb=2.0,
            stake=20.0,
            stakes=[1.0, 20.0, 20.0],
            ante=1.0,
            chip_unit=1.0,
            seed=1,
        )
    )

    settled_stacks = [player.stake for player in first_hand.players_state]
    assert settled_stacks == [0.0, 24.0, 17.0]
    assert sum(settled_stacks) == 41.0

    next_hand = pkrs.State.from_seed(
        n_players=2,
        button=0,
        sb=1.0,
        bb=2.0,
        stake=0.0,
        stakes=[stack for stack in settled_stacks if stack > 0.0],
        ante=1.0,
        chip_unit=1.0,
        seed=2,
    )

    assert len(next_hand.players_state) == 2
    assert {player.player for player in next_hand.players_state} == {0, 1}
    assert sum(
        player.stake + player.bet_chips + player.pot_chips
        for player in next_hand.players_state
    ) == 41.0
