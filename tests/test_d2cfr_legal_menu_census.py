import importlib

import numpy as np
import pokers as pkrs


probe = importlib.import_module("tools.d2cfr_legal_menu_census")


def test_classify_raise_menu_distinguishes_half_pot_from_all_in_only():
    mask = np.array([1, 0, 1, 1, 0, 1], dtype=np.float32)

    assert probe.classify_raise_menu(mask) == "half_pot_and_all_in_without_pot"


def test_summarise_visits_reports_legacy_all_in_only_counterfactual():
    visits = [
        {
            "street": "Flop",
            "after_raise": True,
            "mask": np.array([1, 0, 1, 1, 0, 1], dtype=np.float32),
            "policy": np.array([0.1, 0.0, 0.1, 0.6, 0.0, 0.2], dtype=np.float64),
            "sampled_slot": 3,
        },
        {
            "street": "Turn",
            "after_raise": False,
            "mask": np.array([1, 0, 1, 1, 1, 1], dtype=np.float32),
            "policy": np.array([0.1, 0.0, 0.2, 0.2, 0.2, 0.3], dtype=np.float64),
            "sampled_slot": 5,
        },
    ]

    result = probe.summarise_visits(visits)

    assert result["overall"]["menu_counts"]["half_pot_and_all_in_without_pot"] == 1
    assert result["after_raise"]["visits"] == 1
    assert result["after_raise"]["half_pot_legal_visits"] == 1
    assert result["after_raise"]["legacy_all_in_only_raise_visits"] == 1
    assert result["after_raise"]["policy_mass_mean"]["raise_0.5pot"] == 0.6


def test_after_raise_detection_supports_pokers_action_enum():
    state = pkrs.State.from_seed(
        n_players=2, button=0, sb=1.0, bb=2.0, stake=200.0, seed=7
    ).apply_action(pkrs.Action(pkrs.ActionEnum.Raise, 2.0))

    assert probe._after_raise_on_current_street(state) is True
