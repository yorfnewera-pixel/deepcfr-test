from pathlib import Path
from typing import Any, cast

import numpy as np
import pokers as pkrs
import torch

from src.core.action_space import ActionSlot
from src.agents.legacy_starting_opponent import LegacyStartingOpponent, _encode_legacy_state


CHECKPOINT = (
    Path(__file__).parents[1]
    / "models"
    / "starting_opponent"
    / "mixed_checkpoint_iter_11200.pt"
)


def test_legacy_starting_opponent_loads_four_action_checkpoint():
    opponent = LegacyStartingOpponent(CHECKPOINT)

    assert opponent.input_size == 156
    assert opponent.num_actions == 4


def test_legacy_starting_opponent_exposes_two_bet_action_contract():
    state = pkrs.State.from_seed(
        n_players=6, button=0, sb=1.0, bb=2.0, stake=200.0, seed=7
    )
    state = state.apply_action(cast(Any, pkrs.Action(pkrs.ActionEnum.Call)))
    opponent = LegacyStartingOpponent(CHECKPOINT)

    choices = opponent.legal_choices(state)

    assert choices[0] is ActionSlot.FOLD
    assert choices[1] is ActionSlot.CALL
    assert choices[2] is ActionSlot.RAISE_HALF_POT
    assert choices[3] is ActionSlot.RAISE_POT


def test_legacy_state_encoding_uses_old_156_feature_contract():
    state = pkrs.State.from_seed(
        n_players=6, button=3, sb=1.0, bb=2.0, stake=200.0, seed=107
    )
    state = state.apply_action(cast(Any, pkrs.Action(pkrs.ActionEnum.Call)))
    encoded = _encode_legacy_state(state, player_id=2)
    player_state = state.players_state[2]
    first_card = player_state.hand[0]
    second_card = player_state.hand[1]
    second_suit = 0 if int(second_card.suit) == int(first_card.suit) else 1
    norm_unit = 200.0

    assert encoded.shape == (156,)
    assert encoded[int(first_card.rank)] == 1.0
    assert encoded[second_suit * 13 + int(second_card.rank)] == 1.0
    assert encoded[109] == np.float32(float(state.pot) / norm_unit)
    assert encoded[110 + ((int(state.button) - 2) % 6)] == 1.0
    assert encoded[116 + ((int(state.current_player) - 2) % 6)] == 1.0
    assert encoded[146] == np.float32(float(state.min_bet) / norm_unit)


def test_legacy_action_one_is_call_when_facing_bet(monkeypatch):
    state = pkrs.State.from_seed(
        n_players=6, button=0, sb=1.0, bb=2.0, stake=200.0, seed=7
    )
    opponent = LegacyStartingOpponent(CHECKPOINT)

    def choose_high_call(_state):
        return torch.tensor([[0.0, 20.0, 0.0, 0.0]], dtype=torch.float32)

    monkeypatch.setattr(opponent.network, "forward", choose_high_call)

    action = opponent.choose_action(state)

    assert action.action == pkrs.ActionEnum.Call


def test_legacy_action_two_is_half_pot_raise_when_available(monkeypatch):
    state = pkrs.State.from_seed(
        n_players=6, button=0, sb=1.0, bb=2.0, stake=200.0, seed=7
    )
    state = state.apply_action(cast(Any, pkrs.Action(pkrs.ActionEnum.Call)))
    opponent = LegacyStartingOpponent(CHECKPOINT)

    def choose_high_half_pot(_state):
        return torch.tensor([[0.0, 0.0, 20.0, 0.0]], dtype=torch.float32)

    monkeypatch.setattr(opponent.network, "forward", choose_high_half_pot)

    action = opponent.choose_action(state)

    assert action.action == pkrs.ActionEnum.Raise
    assert action.amount == float(state.pot) * 0.5


def test_legacy_raise_buttons_become_all_in_only_when_stack_is_too_short():
    state = pkrs.State.from_seed(
        n_players=2, button=0, sb=1.0, bb=2.0, stake=3.0, seed=7
    )
    opponent = LegacyStartingOpponent(CHECKPOINT)

    choices = opponent.legal_choices(state)

    assert choices[2] is ActionSlot.ALL_IN
    assert choices[3] is ActionSlot.ALL_IN
