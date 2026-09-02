import numpy as np
import subprocess
import sys
from types import SimpleNamespace
from pathlib import Path

from src.core.action_space import NUM_ACTIONS, legal_action_mask, resolve_action
from src.runtime_search.paired_evaluation import _root_diagnostic, run_runtime_paired_evaluation
from src.runtime_search.policy import RuntimeSearchConfig


class UniformBlueprint:
    def probabilities(self, state, player_id=None):
        del player_id
        mask = legal_action_mask(state)
        return mask / mask.sum()

    def probabilities_batch(self, states):
        return np.stack([self.probabilities(state) for state in states])

    def choose_action(self, state, rng):
        slot = int(rng.choice(NUM_ACTIONS, p=self.probabilities(state)))
        return resolve_action(slot, state).action


def test_paired_runner_preserves_each_seed_and_solver_usage():
    result = run_runtime_paired_evaluation(
        UniformBlueprint(),
        config=RuntimeSearchConfig(belief_particles=2),
        seeds=(107, 109),
        num_deals=1,
        num_players=2,
        rotate_seats=False,
    )

    assert [run.seed for run in result.runs] == [107, 109]
    assert all(run.samples == 1 for run in result.runs)
    assert result.total_samples == 2
    assert all(run.search_decisions >= run.reach_weighted_decisions for run in result.runs)
    assert all(run.search_decisions >= run.blueprint_fallback_decisions for run in result.runs)
    assert all(run.action_comparisons == run.search_decisions for run in result.runs)
    assert all(run.action_changes <= run.action_comparisons for run in result.runs)
    assert all(sum(run.search_decisions_by_stage.values()) == run.search_decisions for run in result.runs)
    assert all(sum(run.action_changes_by_stage.values()) == run.action_changes for run in result.runs)
    assert all(run.policy_l1_sum >= 0.0 for run in result.runs)
    assert all(run.policy_l1_max >= 0.0 for run in result.runs)
    assert all(run.policy_argmax_changes <= run.search_decisions for run in result.runs)
    assert all(sum(run.policy_l1_sum_by_stage.values()) == run.policy_l1_sum for run in result.runs)
    assert all(
        sum(run.policy_argmax_changes_by_stage.values()) == run.policy_argmax_changes
        for run in result.runs
    )
    assert result.to_dict()["total_samples"] == 2


def test_runtime_paired_evaluation_cli_starts_from_project_root():
    project_root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [sys.executable, "tools/run_runtime_paired_evaluation.py", "--help"],
        cwd=project_root,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "Paired EV" in result.stdout


def test_root_diagnostic_contains_sequential_resampling_telemetry():
    diagnostic = _root_diagnostic(
        SimpleNamespace(
            action_label="check",
            root_blueprint_policy=np.array([0.5, 0.5]),
            root_search_policy=np.array([0.4, 0.6]),
            raw_ev=np.array([1.0, 2.0]),
            belief_ess=12.0,
            belief_ess_ratio=0.75,
            belief_resample_count=2,
            best_gap=None,
            best_gap_se=None,
            best_gap_zscore=None,
            value_scale=None,
            eta_effective=None,
            diagnostic_flags=("reach_sequential_resampling",),
        )
    )

    assert diagnostic["belief_resample_count"] == 2
    assert diagnostic["belief_ess_ratio"] == 0.75
