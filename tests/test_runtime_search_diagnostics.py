import subprocess
import sys
from pathlib import Path

import numpy as np

from src.core.action_space import NUM_ACTIONS, legal_action_mask
from src.runtime_search.diagnostics import _sample_slot, collect_turn_river_diagnostics


class UniformBlueprint:
    def probabilities(self, state, player_id=None):
        del player_id
        mask = legal_action_mask(state)
        return mask / mask.sum()

    def probabilities_batch(self, states):
        return np.stack([self.probabilities(state) for state in states])


def _without_latency(value):
    if isinstance(value, dict):
        return {
            key: _without_latency(item)
            for key, item in value.items()
            if "latency" not in key
        }
    if isinstance(value, list):
        return [_without_latency(item) for item in value]
    return value


def test_turn_river_diagnostics_are_reproducible_and_json_serializable():
    first = collect_turn_river_diagnostics(
        UniformBlueprint(),
        num_deals=3,
        belief_particles=1,
        seed=107,
    )
    second = collect_turn_river_diagnostics(
        UniformBlueprint(),
        num_deals=3,
        belief_particles=1,
        seed=107,
    )

    assert _without_latency(first.to_dict()) == _without_latency(second.to_dict())
    assert first.natural.completed_deals == 3
    assert first.natural.failed_deals == 0
    assert set(first.natural.streets) == {"turn", "river"}
    assert set(first.constructed_spots) == {"turn", "river", "response_to_bet_turn"}
    assert first.search_probe.action_label
    assert first.search_probe.action_is_legal
    assert first.search_probe.belief_particles == 0
    assert "belief_history_missing_blueprint_fallback" in first.search_probe.diagnostic_flags

    for street in first.natural.streets.values():
        assert street.decisions >= 0
        assert len(street.mean_probability_mass) == NUM_ACTIONS
        assert len(street.chosen_action_frequency) == NUM_ACTIONS
        assert np.isfinite(street.mean_latency_milliseconds)

    for spot in first.constructed_spots.values():
        assert spot.decisions == 1
        assert np.isclose(sum(spot.mean_probability_mass), 1.0)
        assert np.isfinite(spot.mean_entropy)


def test_turn_river_diagnostics_cli_starts_from_project_root():
    project_root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [sys.executable, "tools/run_turn_river_diagnostics.py", "--help"],
        cwd=project_root,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "Диагностика turn/river" in result.stdout


def test_sampling_tolerates_float32_probability_rounding():
    probabilities = np.ones(NUM_ACTIONS, dtype=np.float32) / NUM_ACTIONS

    action_slot = _sample_slot(probabilities.astype(np.float64), seed=107)

    assert 0 <= action_slot < NUM_ACTIONS
