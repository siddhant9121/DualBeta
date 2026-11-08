import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import pytest

from dapm.capm import capm_beta
from dapm.data import simulate_single_factor_market
from dapm.dapm_model import best_beta, upper_lower_betas


def test_single_factor_normal_market_gives_symmetric_betas():
    true_betas = np.array([0.5, 1.0, 1.5, 2.0])
    assets, market, rf = simulate_single_factor_market(
        n_periods=4000, betas=true_betas, seed=1
    )
    xm = (market - rf).to_numpy()

    for j, col in enumerate(assets.columns):
        xi = (assets[col] - rf).to_numpy()
        result = upper_lower_betas(xi, xm)
        true_b = true_betas[j]

        assert abs(result.beta_plus - true_b) < 0.15
        assert abs(result.beta_minus - true_b) < 0.15
        assert abs(result.asymmetry_ratio - 1.0) < 0.25


def test_recovers_planted_asymmetry():
    true_betas = np.array([1.0, 1.0])
    assets, market, rf = simulate_single_factor_market(
        n_periods=6000,
        betas=true_betas,
        asymmetric_assets=(1,),
        asymmetry_strength=2.0,
        seed=11,
    )
    xm = (market - rf).to_numpy()

    symmetric = upper_lower_betas((assets.iloc[:, 0] - rf).to_numpy(), xm)
    asymmetric = upper_lower_betas((assets.iloc[:, 1] - rf).to_numpy(), xm)

    assert abs(symmetric.beta_plus - symmetric.beta_minus) < 0.15

    assert abs(asymmetric.beta_plus - 1.0) < 0.15
    assert abs(asymmetric.beta_minus - 2.0) < 0.20

    assert abs(asymmetric.asymmetry_ratio - 0.5) < 0.12
    assert asymmetric.asymmetry_ratio < symmetric.asymmetry_ratio


def test_regime_counts_partition_the_sample():
    assets, market, rf = simulate_single_factor_market(n_periods=500, seed=3)
    xm = (market - rf).to_numpy()
    xi = (assets.iloc[:, 0] - rf).to_numpy()

    result = upper_lower_betas(xi, xm)
    assert result.n_up + result.n_down == len(xm)
    assert result.n_up > 0 and result.n_down > 0


def test_raises_when_a_regime_is_too_thin():
    xm = np.array([0.01] * 50 + [-0.01] * 2)
    xi = xm * 1.2
    with pytest.raises(ValueError, match="Not enough observations"):
        upper_lower_betas(xi, xm, min_obs=5)


def test_best_beta_rejects_degenerate_market():
    with pytest.raises(ValueError, match="Zero"):
        best_beta(np.array([0.1, 0.2, 0.3]), np.zeros(3))


def test_capm_beta_recovers_known_slope():
    rng = np.random.default_rng(5)
    xm = rng.normal(0, 0.01, size=3000)
    xi = 1.4 * xm + rng.normal(0, 0.002, size=3000)
    assert abs(capm_beta(xi, xm) - 1.4) < 0.05


def test_capm_beta_rejects_zero_variance_market():
    with pytest.raises(ValueError, match="Zero market variance"):
        capm_beta(np.array([0.1, 0.2, 0.3]), np.array([0.01, 0.01, 0.01]))


def test_best_beta_sits_between_the_conditional_betas():
    assets, market, rf = simulate_single_factor_market(
        n_periods=4000, betas=np.array([1.0]),
        asymmetric_assets=(0,), asymmetry_strength=2.0, seed=7,
    )
    xm = (market - rf).to_numpy()
    xi = (assets.iloc[:, 0] - rf).to_numpy()

    result = upper_lower_betas(xi, xm)
    lo, hi = sorted([result.beta_plus, result.beta_minus])
    assert lo <= result.best_beta <= hi
