import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import pytest

from dapm.backtest import (
    asymmetry_screen,
    backtest_asymmetry_portfolio,
    rolling_dapm_betas,
)
from dapm.data import simulate_single_factor_market


@pytest.fixture(scope="module")
def panel_fixture():
    assets, market, rf = simulate_single_factor_market(
        n_periods=800,
        n_assets=10,
        asymmetric_assets=(0, 1, 2),
        asymmetry_strength=2.0,
        seed=42,
    )
    panel = rolling_dapm_betas(assets, market, rf, window=126, step=21)
    return assets, panel


def test_rolling_panel_has_expected_shape(panel_fixture):
    assets, panel = panel_fixture
    assert not panel.empty
    assert set(panel.columns) >= {
        "date", "asset", "beta_plus", "beta_minus",
        "best_beta", "capm_beta", "asymmetry_ratio",
    }
    assert panel["asset"].nunique() == assets.shape[1]
    assert (panel["n_up"] > 0).all()
    assert (panel["n_down"] > 0).all()


def test_screen_is_sorted_descending_and_finite(panel_fixture):
    _, panel = panel_fixture
    snap = asymmetry_screen(panel)

    ratios = snap["asymmetry_ratio"].to_numpy()
    assert np.all(np.diff(ratios) <= 1e-9), "screen must be sorted descending"
    assert np.isfinite(ratios).all(), "inf/nan must be dropped"
    assert len(snap) == panel[panel["date"] == panel["date"].max()].shape[0]


def test_screen_ranks_planted_asymmetry_at_the_bottom(panel_fixture):
    _, panel = panel_fixture
    snap = asymmetry_screen(panel)

    bottom_three = set(snap.tail(3)["asset"])
    planted = {"ASSET_01", "ASSET_02", "ASSET_03"}
    assert len(bottom_three & planted) >= 2


def test_backtest_returns_finite_stats(panel_fixture):
    assets, panel = panel_fixture
    res = backtest_asymmetry_portfolio(assets, panel, top_q=0.3, bottom_q=0.3)

    assert np.isfinite(res.annualized_return)
    assert np.isfinite(res.annualized_vol)
    assert np.isfinite(res.sharpe)
    assert res.annualized_vol > 0
    assert res.max_drawdown <= 0
    assert len(res.equity_curve) == len(res.daily_returns)
    assert res.daily_returns.index.is_monotonic_increasing
    assert not res.daily_returns.index.has_duplicates


def test_screen_respects_explicit_as_of(panel_fixture):
    _, panel = panel_fixture
    dates = sorted(panel["date"].unique())
    earlier = dates[len(dates) // 2]

    snap = asymmetry_screen(panel, as_of=earlier)
    assert (snap["date"] == earlier).all()
