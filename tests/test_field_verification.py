import numpy as np
import pokers as pkrs

from src.core.action_space import ActionSlot, legal_action_mask, resolve_action
from src.core.model import encode_state


def test_state_encoding_keeps_expected_shape_and_normalization():
    state = pkrs.State.from_seed(n_players=6, button=0, sb=1.0, bb=2.0, stake=200.0, seed=42)
    encoded = encode_state(state, player_id=0)
    assert encoded.shape == (157,)
    assert encoded[109] == np.float32(state.pot / (100.0 * state.bb))


def test_raise_slots_use_increment_over_call():
    state = pkrs.State.from_seed(n_players=2, button=0, sb=1.0, bb=2.0, stake=200.0, seed=42)
    assert legal_action_mask(state)[ActionSlot.RAISE_POT] == 1.0
    action = resolve_action(ActionSlot.RAISE_POT, state).action
    assert action.action == pkrs.ActionEnum.Raise
    assert action.amount == state.pot
