import importlib
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from src.core.deep_cfr import DeepCFRAgent
from src.training import train as train_mod
from src.utils import config as config_mod


probe = importlib.import_module("tools.d2cfr_allin_bias_probe")


def test_summarise_action_comparison_separates_all_in_bias_and_regret_signs():
    report = probe.summarise_action_comparison([
        {
            "action_label": "all_in",
            "target_q": 0.10,
            "predicted_q": 0.16,
            "target_regret": 0.02,
            "predicted_regret": 0.01,
        },
        {
            "action_label": "all_in",
            "target_q": 0.20,
            "predicted_q": 0.18,
            "target_regret": -0.03,
            "predicted_regret": 0.02,
        },
        {
            "action_label": "raise_0.5pot",
            "target_q": 0.12,
            "predicted_q": 0.06,
            "target_regret": 0.01,
            "predicted_regret": -0.02,
        },
    ])

    all_in = report["by_action"]["all_in"]
    half_pot = report["by_action"]["raise_0.5pot"]
    assert all_in["count"] == 2
    assert all_in["q_bias_mean"] == pytest.approx(0.02)
    assert all_in["q_mae"] == pytest.approx(0.04)
    assert all_in["regret_sign_flip_rate"] == pytest.approx(0.5)
    assert half_pot["q_bias_mean"] == pytest.approx(-0.06)
    assert half_pot["regret_sign_flip_rate"] == pytest.approx(1.0)
    assert report["all_in_minus_half_pot_q_bias"] == pytest.approx(0.08)


def test_summarise_action_comparison_rejects_unknown_action_label():
    with pytest.raises(ValueError, match="неизвестный слот"):
        probe.summarise_action_comparison([
            {
                "action_label": "unknown",
                "target_q": 0.0,
                "predicted_q": 0.0,
                "target_regret": 0.0,
                "predicted_regret": 0.0,
            },
        ])


def test_describe_resampled_targets_compares_network_error_to_target_noise_and_rm_policy():
    """Ловит регрессию, где отчёт смешивает ошибку сети с разбросом target."""
    legal_mask = np.array([0, 0, 0, 1, 1, 1], dtype=np.float32)
    q_samples = np.array([
        [0, 0, 0, 0.20, 0.45, 0.70],
        [0, 0, 0, 0.30, 0.55, 0.80],
    ], dtype=np.float32)
    v_samples = np.array([0.40, 0.50], dtype=np.float32)
    regret_samples = np.array([
        [0, 0, 0, -0.20, 0.05, 0.30],
        [0, 0, 0, -0.20, 0.05, 0.30],
    ], dtype=np.float32)
    predicted_q = np.array([0, 0, 0, 0.25, 0.50, 0.20], dtype=np.float32)
    predicted_v = 0.45
    predicted_regret = np.array([0, 0, 0, -0.20, 0.05, 0.30], dtype=np.float32)

    report = probe.describe_resampled_targets(
        q_samples=q_samples,
        v_samples=v_samples,
        regret_samples=regret_samples,
        predicted_q=predicted_q,
        predicted_v=predicted_v,
        predicted_regret=predicted_regret,
        legal_mask=legal_mask,
        noise_floor=0.01,
    )

    all_in = report["by_action"]["all_in"]
    assert all_in["target_q_mean"] == pytest.approx(0.75)
    assert all_in["target_q_std"] == pytest.approx(0.05)
    assert all_in["q_abs_error"] == pytest.approx(0.55)
    assert all_in["q_error_to_noise_ratio"] == pytest.approx(11.0)
    assert report["target_rm_policy"][3:] == pytest.approx([0.0, 1 / 7, 6 / 7])
    assert report["predicted_rm_policy"][3:] == pytest.approx([0.0, 1 / 7, 6 / 7])
    assert report["rm_l1"] == pytest.approx(0.0)
    assert report["target_q_covariance"][4][5] == pytest.approx(0.0025)
    assert report["target_regret_covariance"][4][5] == pytest.approx(0.0)


def test_summarise_all_in_traces_counts_opponent_responses_and_rewards():
    report = probe.summarise_all_in_traces([
        {"opponent_response": "fold", "terminal_reward": 0.04},
        {"opponent_response": "call", "terminal_reward": 1.0},
        {"opponent_response": "call", "terminal_reward": -1.0},
    ])

    assert report["samples"] == 3
    assert report["opponent_response_counts"] == {"call": 2, "fold": 1}
    assert report["terminal_reward_mean"] == pytest.approx(0.013333333333333334)
    assert report["terminal_reward_std"] == pytest.approx(0.8167142843260563)


def test_card_code_uses_rank_and_suit_instead_of_python_object_address():
    class Card:
        rank = "RA"
        suit = "Spades"

    assert probe.card_code(Card()) == "As"


def test_deal_key_uses_both_players_private_cards_not_street_or_board():
    class Card:
        def __init__(self, rank, suit):
            self.rank = rank
            self.suit = suit

    class Player:
        def __init__(self, hand):
            self.hand = hand

    class State:
        players_state = (
            Player((Card("RA", "Spades"), Card("RK", "Spades"))),
            Player((Card("RQ", "Hearts"), Card("RJ", "Hearts"))),
        )

    assert probe.deal_key(State()) == (("As", "Ks"), ("Qh", "Jh"))


def test_run_probe_writes_postflop_action_report(tmp_path):
    config_path = tmp_path / "probe.yaml"
    config_path.write_text(
        "\n".join((
            "num_actions: 6",
            "num_players: 2",
            "num_trainable_players: 2",
            "hu_current_policy_self_play: true",
            "hidden_size: 8",
            "d2cfr_enabled: true",
            "d2cfr_mc_correction_enabled: false",
            "advantage_memory_size: 32",
            "strategy_memory_size: 32",
        )) + "\n",
        encoding="utf-8",
    )
    config_mod.load_config(config_path)
    try:
        source = DeepCFRAgent(player_id=0, num_players=2, device="cpu")
        train_mod._create_hu_current_policy_coordinator(source)
        checkpoint_path = train_mod._save_hu_checkpoint(source, tmp_path / "hu.pt")
        output_path = tmp_path / "report.json"
        report = probe.run_probe(
            checkpoint_path,
            config_path=config_path,
            output_path=output_path,
            states=1,
            repeats=2,
            roots=64,
            seed=17,
            trace_all_in=True,
        )
    finally:
        config_mod.load_config("config.yaml")

    assert output_path.is_file()
    assert report["states_collected"] == 1
    assert report["repeats_per_state"] == 2
    assert {"all_in", "raise_0.5pot", "raise_1pot"} <= set(report["summary"]["by_action"])
    assert report["summary"]["rm_l1_mean"] >= 0.0
    state_resampling = report["states"][0]["resampling"]
    assert state_resampling["sample_count"] == 2
    assert state_resampling["rm_l1"] >= 0.0
    assert len(state_resampling["target_rm_policy"]) == 6
    raw_targets = report["states"][0]["target_samples"]
    assert len(raw_targets["q"]) == 2
    assert len(raw_targets["q"][0]) == 6
    assert len(raw_targets["v"]) == 2
    assert len(raw_targets["regret"]) == 2
    trace = report["states"][0]["all_in_trace"]
    assert trace["summary"]["samples"] == 2
    assert len(trace["samples"]) == 2


def test_probe_uses_provenance_mode_recorded_in_checkpoint(tmp_path):
    checkpoint_path = tmp_path / "checkpoint.pt"
    torch.save({
        "advantage_legs": [
            {"buffer": {"provenance_enabled": False}},
            {"buffer": {"provenance_enabled": False}},
        ],
    }, checkpoint_path)

    assert probe._checkpoint_provenance_enabled(checkpoint_path) is False


def test_probe_applies_checkpoint_step_budget_before_validating_runtime():
    agent = SimpleNamespace(advantage_train_steps=3000, strategy_train_steps=5000)

    probe._apply_checkpoint_training_contract(agent, {
        "config": {"advantage_train_steps": 750, "strategy_train_steps": 5000}
    })

    assert agent.advantage_train_steps == 750
