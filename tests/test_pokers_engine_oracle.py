import importlib
import subprocess
import sys
from pathlib import Path

import pytest

engine_oracle = importlib.import_module("tools.pokers_engine_oracle")


def test_preflop_fold_engine_oracle_matches_blind_accounting_and_d2cfr_target():
    report = engine_oracle.run_preflop_fold_oracle(
        config_path=Path("config.yaml"),
        seed=123,
        button=0,
        sb=1.0,
        bb=2.0,
        stake=200.0,
        traversing_player=0,
        iteration=11,
        device="cpu",
    )

    assert report["passed"] is True
    assert report["engine"]["actual_rewards"] == pytest.approx([-1.0, 1.0])
    assert report["engine"]["expected_rewards"] == pytest.approx([-1.0, 1.0])
    assert report["target"]["root_return"] == pytest.approx(-1.0)
    assert report["target"]["recorded"]["action_values"][0] == pytest.approx(-0.005)
    assert report["target"]["recorded"]["state_value"] == pytest.approx(-0.005)
    assert report["target"]["recorded"]["regrets"][0] == pytest.approx(0.0)
    assert report["target"]["max_abs_errors"]["regret"] <= 1e-6


def test_river_showdown_engine_oracle_matches_known_winner_and_d2cfr_target():
    report = engine_oracle.run_river_showdown_oracle(
        config_path=Path("config.yaml"),
        traversing_player=0,
        iteration=12,
        device="cpu",
    )

    assert report["passed"] is True
    assert report["engine"]["actual_rewards"] == pytest.approx([10.0, -10.0])
    assert report["engine"]["expected_rewards"] == pytest.approx([10.0, -10.0])
    assert report["target"]["root_return"] == pytest.approx(10.0)
    assert report["target"]["recorded"]["action_values"][1] == pytest.approx(0.05)
    assert report["target"]["recorded"]["state_value"] == pytest.approx(0.05)
    assert report["target"]["recorded"]["regrets"][1] == pytest.approx(0.0)
    assert report["target"]["max_abs_errors"]["regret"] <= 1e-6


def test_river_allin_response_oracle_matches_exact_multibranch_q_v_r():
    report = engine_oracle.run_river_allin_response_oracle(
        config_path=Path("config.yaml"),
        traversing_player=1,
        iteration=13,
        device="cpu",
    )

    assert report["passed"] is True
    assert report["expected"]["raw_action_values"][0] == pytest.approx(0.0)
    assert report["expected"]["raw_action_values"][2] == pytest.approx(-200.0)
    assert report["expected"]["raw_state_value"] == pytest.approx(-150.0)
    assert report["expected"]["raw_regrets"][0] == pytest.approx(150.0)
    assert report["expected"]["raw_regrets"][2] == pytest.approx(-50.0)
    assert report["recorded"]["action_values"][0] == pytest.approx(0.0)
    assert report["recorded"]["action_values"][2] == pytest.approx(-1.0)
    assert report["recorded"]["state_value"] == pytest.approx(-0.75)
    assert report["recorded"]["regrets"][0] == pytest.approx(0.75)
    assert report["recorded"]["regrets"][2] == pytest.approx(-0.25)
    assert report["max_abs_errors"]["regret"] <= 1e-6


def test_pokers_engine_oracle_script_can_run_from_tools_directory():
    script = Path(__file__).resolve().parents[1] / "tools" / "pokers_engine_oracle.py"

    result = subprocess.run(
        [sys.executable, str(script), "--help"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    assert "--seed" in result.stdout
