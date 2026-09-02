import pytest
import pokers as pkrs
import torch

from src.core.action_space import ActionSlot, NUM_ACTIONS, legal_action_mask, resolve_action
from src.core.deep_cfr import DeepCFRAgent
from policy_runtime.core import PolicyRuntimeAgent


def _state(stake=200.0):
    return pkrs.State.from_seed(
        n_players=2, button=0, sb=1.0, bb=2.0, stake=stake, seed=7
    )


def test_fixed_slots_have_exact_preflop_semantics():
    state = _state()
    mask = legal_action_mask(state)
    assert mask.shape == (NUM_ACTIONS,)
    assert mask[ActionSlot.RAISE_HALF_POT] == 0.0  # 1.5 < minraise 2
    assert mask[ActionSlot.RAISE_POT] == 1.0
    assert mask[ActionSlot.ALL_IN] == 1.0

    pot_raise = resolve_action(ActionSlot.RAISE_POT, state).action
    all_in = resolve_action(ActionSlot.ALL_IN, state).action
    assert pot_raise.action == pkrs.ActionEnum.Raise
    assert pot_raise.amount == pytest.approx(float(state.pot))
    assert all_in.amount == pytest.approx(198.0)


def test_invalid_fixed_size_is_masked_instead_of_clamped():
    state = _state(stake=3.0)
    mask = legal_action_mask(state)
    assert mask[ActionSlot.RAISE_HALF_POT] == 0.0
    assert mask[ActionSlot.RAISE_POT] == 0.0
    assert mask[ActionSlot.ALL_IN] == 1.0
    with pytest.raises(ValueError):
        resolve_action(ActionSlot.RAISE_POT, state)


def test_half_pot_raise_is_illegal_after_raise_on_current_street():
    state = _state().apply_action(pkrs.Action(pkrs.ActionEnum.Raise, 2.0))

    mask = legal_action_mask(state)

    assert mask[ActionSlot.RAISE_HALF_POT] == 0.0
    assert mask[ActionSlot.RAISE_POT] == 1.0
    assert mask[ActionSlot.ALL_IN] == 1.0


def test_policy_runtime_uses_the_same_slot_mask_as_training(tmp_path):
    checkpoint = tmp_path / "six_fixed.pt"
    DeepCFRAgent(player_id=0, num_players=2).save_model(str(checkpoint))
    state = _state()
    assert (PolicyRuntimeAgent(str(checkpoint)).get_legal_action_mask(state) == legal_action_mask(state)).all()


def test_policy_runtime_loads_v2_light_checkpoint(tmp_path):
    checkpoint = tmp_path / "light_six_fixed.pt"
    agent = DeepCFRAgent(player_id=0, num_players=6)
    torch.save(agent.build_light_checkpoint(), checkpoint)

    loaded = PolicyRuntimeAgent(str(checkpoint))

    assert loaded.iteration == 0
    assert loaded.validate() == ["OK: чекпоинт совместим"]
