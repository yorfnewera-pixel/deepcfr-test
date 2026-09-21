import importlib
import subprocess
import sys
from pathlib import Path

import pytest

target_oracle = importlib.import_module("tools.d2cfr_target_oracle")


def test_one_step_target_oracle_matches_exact_q_v_r():
    report = target_oracle.run_one_step_target_oracle(
        config_path=Path("config.yaml"),
        legal_payoffs={1: -20.0, 3: 40.0, 5: 80.0},
        advantages={1: 0.0, 3: 1.0, 5: 3.0},
        iteration=7,
        traversing_player=0,
        device="cpu",
    )

    assert report["passed"] is True
    assert report["root_return"] == pytest.approx(70.0)
    assert report["expected"]["strategy"] == pytest.approx([0.0, 0.0, 0.0, 0.25, 0.0, 0.75])
    assert report["expected"]["raw_action_values"] == pytest.approx([0.0, -20.0, 0.0, 40.0, 0.0, 80.0])
    assert report["expected"]["raw_state_value"] == pytest.approx(70.0)
    assert report["expected"]["raw_regrets"] == pytest.approx([0.0, -90.0, 0.0, -30.0, 0.0, 10.0])
    assert report["recorded"]["state_value"] == pytest.approx(0.35)
    assert report["recorded"]["action_values"] == pytest.approx([0.0, -0.1, 0.0, 0.2, 0.0, 0.4])
    assert report["recorded"]["regrets"] == pytest.approx([0.0, -0.45, 0.0, -0.15, 0.0, 0.05])
    assert report["max_abs_errors"]["regret"] <= 1e-6


def test_target_oracle_script_can_run_from_tools_directory():
    script = Path(__file__).resolve().parents[1] / "tools" / "d2cfr_target_oracle.py"

    result = subprocess.run(
        [sys.executable, str(script), "--help"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    assert "--payoff" in result.stdout
