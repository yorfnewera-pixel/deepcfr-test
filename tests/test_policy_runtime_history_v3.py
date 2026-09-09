import numpy as np
import pokers as pkrs

from policy_runtime.adapters.pokers import wrap_state
from policy_runtime.core import encode_state_history_summary_v3 as runtime_encode_state
from src.core.model import encode_state_history_summary_v3 as training_encode_state


def test_runtime_history_v3_encoder_matches_training_for_small_blind() -> None:
    state = pkrs.State.from_seed(
        n_players=2,
        button=0,
        sb=0.0025,
        bb=0.005,
        stake=2.0,
        seed=107,
    )

    training_vector = training_encode_state(state, player_id=int(state.current_player))
    runtime_vector = runtime_encode_state(
        wrap_state(state),
        player_id=int(state.current_player),
    )

    assert np.allclose(runtime_vector, training_vector)
