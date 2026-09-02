import numpy as np

from src.runtime_search.mmds import mmds_update


def test_mmds_keeps_probability_on_legal_actions_and_normalizes_output():
    result = mmds_update(
        pi=np.array([0.4, 0.6, 0.0]),
        mc_values=np.array([0.1, -0.2, 100.0]),
        mask=np.array([1.0, 1.0, 0.0]),
        eta=1.0,
        alpha=0.05,
    )

    assert np.isclose(result.sum(), 1.0)
    assert result[2] == 0.0
    assert np.all(result[:2] > 0.0)


def test_mmds_with_zero_eta_and_no_floor_keeps_blueprint_policy():
    result = mmds_update(
        pi=np.array([0.2, 0.3, 0.5]),
        mc_values=np.array([-3.0, 0.0, 4.0]),
        mask=np.array([1.0, 1.0, 1.0]),
        eta=0.0,
        alpha=0.5,
        floor=0.0,
    )

    assert np.allclose(result, np.array([0.2, 0.3, 0.5]))


def test_mmds_increases_probability_of_action_with_higher_value():
    blueprint = np.array([0.5, 0.5])
    result = mmds_update(
        pi=blueprint,
        mc_values=np.array([0.0, 1.0]),
        mask=np.array([1.0, 1.0]),
        eta=2.0,
        alpha=0.05,
    )

    assert result[1] > blueprint[1]


def test_mmds_applies_floor_only_to_legal_actions():
    result = mmds_update(
        pi=np.array([1.0, 0.0, 0.0]),
        mc_values=np.array([0.0, -10.0, 100.0]),
        mask=np.array([1.0, 1.0, 0.0]),
        eta=1.0,
        alpha=0.0,
        floor=0.1,
    )

    assert result[0] >= 0.1
    assert result[1] >= 0.1
    assert result[2] == 0.0


def test_mmds_with_blueprint_reference_matches_mds_with_effective_eta():
    blueprint = np.array([0.2, 0.3, 0.5])
    values = np.array([-0.4, 0.1, 0.7])
    eta = 2.0
    alpha = 0.5
    eta_eff = eta / (1.0 + alpha * eta)
    log_weights = np.log(blueprint) + eta_eff * values
    expected = np.exp(log_weights - np.max(log_weights))
    expected /= expected.sum()

    result = mmds_update(
        pi=blueprint,
        mc_values=values,
        mask=np.array([1.0, 1.0, 1.0]),
        eta=eta,
        alpha=alpha,
        rho=blueprint,
        floor=0.0,
    )

    assert np.allclose(result, expected)
