"""Контракт фиксированной HU-игры 1/2/200."""
from __future__ import annotations

import json

import pytest
import torch

from src.core.deep_cfr import DeepCFRAgent
from src.evaluation.blueprint_policy import FrozenBlueprintPolicy
from src.training import train as train_mod
from src.utils import config as config_mod


_EXPECTED_GAME_CONTRACT = {
    "small_blind": 1.0,
    "big_blind": 2.0,
    "starting_stack": 200.0,
    "depth_big_blinds": 100.0,
}


def _hu_agent() -> DeepCFRAgent:
    agent = DeepCFRAgent(player_id=0, num_players=2, device="cpu")
    train_mod._create_hu_current_policy_coordinator(agent)
    return agent


def test_new_hu_hand_preserves_fixed_blinds_stack_depth_and_button():
    state = train_mod._new_hand(2, seed=41)

    assert state.button == 1
    assert state.sb == 1.0
    assert state.bb == 2.0
    assert state.bb / state.sb == 2.0
    assert state.players_state[0].stake + state.players_state[0].bet_chips == 200.0
    assert state.players_state[1].stake + state.players_state[1].bet_chips == 200.0
    assert 200.0 / state.bb == 100.0


def test_new_hu_hand_allows_explicit_button_without_changing_fixed_contract():
    state = train_mod._new_hand(2, seed=41, button=0)

    assert state.button == 0
    assert state.sb == 1.0
    assert state.bb == 2.0
    assert state.players_state[0].stake + state.players_state[0].bet_chips == 200.0


def test_config_yaml_does_not_expose_big_blind_as_training_setting():
    config_mod.load_config("config.yaml")

    assert not config_mod.cfg_has_raw("big_blind")


@pytest.mark.parametrize(
    "legacy_key",
    ("small_blind", "big_blind", "starting_stack", "stake", "stack_size"),
)
def test_config_loader_rejects_legacy_stake_settings(tmp_path, legacy_key):
    config_path = tmp_path / "legacy-stakes.yaml"
    config_path.write_text(f"{legacy_key}: 999\n", encoding="utf-8")

    try:
        with pytest.raises(ValueError, match=r"SB=1.*BB=2.*stack=200"):
            config_mod.load_config(config_path)
    finally:
        config_mod.load_config("config.yaml")


def test_game_contract_metadata_is_identical_in_hu_artifacts(tmp_path):
    agent = _hu_agent()
    full_checkpoint = train_mod._build_hu_checkpoint(agent, seed=41)
    light_checkpoint = agent.build_light_checkpoint(seed=41)
    run_directory, manifest = train_mod._prepare_hu_run_directory(
        tmp_path,
        agent=agent,
        seed=41,
        initial_checkpoint=None,
    )

    persisted_manifest = json.loads(
        (run_directory / "run_manifest.json").read_text(encoding="utf-8")
    )
    assert full_checkpoint["game_contract"] == _EXPECTED_GAME_CONTRACT
    assert light_checkpoint["game_contract"] == _EXPECTED_GAME_CONTRACT
    assert manifest["game_contract"] == _EXPECTED_GAME_CONTRACT
    assert persisted_manifest["game_contract"] == _EXPECTED_GAME_CONTRACT


def test_advantage_reward_scale_does_not_change_hu_game_contract(tmp_path):
    config_path = tmp_path / "reward-scale.yaml"
    config_path.write_text("advantage_reward_scale: 17\n", encoding="utf-8")

    try:
        config_mod.load_config(config_path)
        state = train_mod._new_hand(2, seed=41)
        assert config_mod.cfg_get("advantage_reward_scale") == 17
        assert state.sb == 1.0
        assert state.bb == 2.0
        assert state.players_state[0].stake + state.players_state[0].bet_chips == 200.0
    finally:
        config_mod.load_config("config.yaml")


def test_hu_resume_rejects_incompatible_game_contract(tmp_path):
    agent = _hu_agent()
    checkpoint = train_mod._build_hu_checkpoint(agent, seed=41)
    checkpoint["game_contract"] = {**_EXPECTED_GAME_CONTRACT, "big_blind": 4.0}
    checkpoint_path = tmp_path / "wrong-contract.pt"
    torch.save(checkpoint, checkpoint_path)

    with pytest.raises(ValueError, match="игровой контракт"):
        train_mod._load_hu_checkpoint(_hu_agent(), checkpoint_path)


def test_hu_resume_rejects_missing_game_contract(tmp_path):
    checkpoint = train_mod._build_hu_checkpoint(_hu_agent(), seed=41)
    checkpoint.pop("game_contract")
    checkpoint_path = tmp_path / "missing-contract.pt"
    torch.save(checkpoint, checkpoint_path)

    with pytest.raises(ValueError, match="игровой контракт"):
        train_mod._load_hu_checkpoint(_hu_agent(), checkpoint_path)


@pytest.mark.parametrize("checkpoint_kind", ("light", "full"))
@pytest.mark.parametrize("contract_mutation", ("incompatible", "missing"))
def test_frozen_evaluation_rejects_invalid_hu_game_contract(
    tmp_path, checkpoint_kind, contract_mutation
):
    agent = _hu_agent()
    if checkpoint_kind == "light":
        checkpoint = agent.build_light_checkpoint(seed=41)
    else:
        checkpoint = train_mod._build_hu_checkpoint(agent, seed=41)
    if contract_mutation == "incompatible":
        checkpoint["game_contract"] = {**_EXPECTED_GAME_CONTRACT, "starting_stack": 100.0}
    else:
        checkpoint.pop("game_contract")
    checkpoint_path = tmp_path / f"{contract_mutation}-contract-{checkpoint_kind}.pt"
    torch.save(checkpoint, checkpoint_path)

    with pytest.raises(ValueError, match="игровой контракт"):
        FrozenBlueprintPolicy.from_checkpoint(checkpoint_path, num_players=2)
