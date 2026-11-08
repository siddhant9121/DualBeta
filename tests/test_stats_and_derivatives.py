import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import pytest

from dapm.cross_market_derivatives import (
    implied_beta_from_prices,
    price_cross_market_options,
)
from dapm.dapm_model import upper_lower_betas
from dapm.data import simulate_single_factor_market
from dapm.gain_loss import gain_loss_ratio, relative_gain_loss_ratio
from dapm.stats_tests import compare_pricing_errors, cross_sectional_pricing_errors


@pytest.fixture(scope="module")
def market_fixture():
    return simulate_single_factor_market(
        n_periods=1500,
        n_assets=12,
        asymmetric_assets=(0, 1, 2, 3, 4, 5),
        asymmetry_strength=2.0,
        seed=99,
    )


def test_pricing_errors_cover_every_asset(market_fixture):
    assets, market, rf = market_fixture
    df = cross_sectional_pricing_errors(assets, market, rf)

    assert len(df) == assets.shape[1]
    assert (df["capm_sq_error"] >= 0).all()
    assert (df["dapm_sq_error"] >= 0).all()


def test_dapm_reduces_pricing_error_on_asymmetric_data(market_fixture):
    assets, market, rf = market_fixture
    df = cross_sectional_pricing_errors(assets, market, rf)
    summary = compare_pricing_errors(df)

    assert summary["mean_dapm_sq_error"] < summary["mean_capm_sq_error"]
    assert summary["pct_error_reduction"] > 0
    assert 0.0 <= summary["wilcoxon_pvalue"] <= 1.0
    assert summary["n_assets"] == assets.shape[1]


def test_wilcoxon_is_skipped_on_tiny_samples():
    import pandas as pd

    df = pd.DataFrame({
        "asset": ["A", "B", "C"],
        "capm_sq_error": [1e-6, 2e-6, 3e-6],
        "dapm_sq_error": [0.5e-6, 1e-6, 2e-6],
    })
    summary = compare_pricing_errors(df)
    assert np.isnan(summary["wilcoxon_pvalue"])


def test_gain_loss_ratio_basic_cases():
    assert gain_loss_ratio(np.array([1.0, -1.0])) == pytest.approx(1.0)
    assert gain_loss_ratio(np.array([2.0, -1.0])) == pytest.approx(2.0)
    assert np.isinf(gain_loss_ratio(np.array([1.0, 2.0])))
    assert np.isnan(gain_loss_ratio(np.zeros(4)))


def test_relative_gain_loss_ratio():
    assert relative_gain_loss_ratio(np.array([2.0, 4.0]), np.array([1.0, 1.0])) == pytest.approx(3.0)
    assert np.isinf(relative_gain_loss_ratio(np.array([1.0]), np.array([0.0])))


def test_implied_betas_match_the_direct_estimator(market_fixture):
    assets, market, rf = market_fixture
    xm = (market - rf).to_numpy()
    xi = (assets.iloc[:, 0] - rf).to_numpy()

    direct = upper_lower_betas(xi, xm)
    prices = price_cross_market_options(xi, xm)
    implied_plus, implied_minus = implied_beta_from_prices(prices)

    assert implied_plus == pytest.approx(direct.beta_plus, rel=0.35)
    assert implied_minus == pytest.approx(direct.beta_minus, rel=0.35)


def test_option_prices_are_non_negative_for_the_market_legs(market_fixture):
    _, market, rf = market_fixture
    xm = (market - rf).to_numpy()
    prices = price_cross_market_options(xm, xm)

    assert prices.c_m >= 0
    assert prices.p_m >= 0
    assert prices.c_i == pytest.approx(prices.c_m)
    assert prices.p_i == pytest.approx(prices.p_m)
