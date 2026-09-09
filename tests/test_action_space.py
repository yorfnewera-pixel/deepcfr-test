import pytest
import pokers as pkrs
import torch

from src.core.action_space import ActionSlot, NUM_ACTIONS, legal_action_mask, resolve_action
from src.core.deep_cfr import DeepCFRAgent
from src.core.model import CARD_CONTEXT_ARCHITECTURE, encode_state_for_version as encode_training_state
from policy_runtime.core import PolicyRuntimeAgent, encode_state_for_version as encode_runtime_state


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


def test_policy_runtime_loads_card_context_light_checkpoint(tmp_path):
    checkpoint = tmp_path / "card_context.pt"
    agent = DeepCFRAgent(
        player_id=0,
        num_players=2,
        network_architecture=CARD_CONTEXT_ARCHITECTURE,
    )
    torch.save(agent.build_light_checkpoint(), checkpoint)

    loaded = PolicyRuntimeAgent(str(checkpoint))
    state = _state()
    player_id = int(state.current_player)
    training_encoded = torch.from_numpy(
        encode_training_state(state, player_id, agent.encoding_version)
    ).unsqueeze(0)
    runtime_encoded = torch.from_numpy(
        encode_runtime_state(state, player_id, loaded.encoding_version)
    ).unsqueeze(0)

    assert loaded.strategy_net.architecture == CARD_CONTEXT_ARCHITECTURE
    torch.testing.assert_close(training_encoded, runtime_encoded)
    with torch.inference_mode():
        torch.testing.assert_close(
            agent.strategy_net(training_encoded),
            loaded.strategy_net(runtime_encoded),
        )
    assert loaded.choose_action(state, player_id=player_id, deterministic=True) in range(NUM_ACTIONS)


def test_policy_runtime_loads_top_level_metadata_when_checkpoint_config_is_invalid(tmp_path):
    checkpoint = tmp_path / "invalid_config.pt"
    payload = DeepCFRAgent(player_id=0, num_players=2).build_light_checkpoint()
    payload["config"] = None
    torch.save(payload, checkpoint)

    loaded = PolicyRuntimeAgent(str(checkpoint))

    assert loaded.encoding_version == "history_summary_v3"
