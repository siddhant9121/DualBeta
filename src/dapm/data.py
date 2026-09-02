import numpy as np
import pandas as pd


def simulate_single_factor_market(n_periods=1000, n_assets=10, betas=None,
                                   market_mu=0.0004, market_sigma=0.010,
                                   idio_sigma=0.006, riskfree=0.00008,
                                   asymmetric_assets=(), asymmetry_strength=2.0,
                                   seed=7):

    rng = np.random.default_rng(seed)
    if betas is None:
        betas = rng.uniform(0.4, 1.6, size=n_assets)
    else:
        betas = np.asarray(betas, dtype=float)
        n_assets = len(betas)

    market_excess = rng.normal(market_mu, market_sigma, size=n_periods)
    market_returns = riskfree + market_excess
    down_mask = market_excess <= 0

    asset_returns = np.zeros((n_periods, n_assets))
    for j in range(n_assets):
        b_series = np.where(down_mask, betas[j] * asymmetry_strength, betas[j]) \
            if j in asymmetric_assets else np.full(n_periods, betas[j])
        eps = rng.normal(0.0, idio_sigma, size=n_periods)
        asset_returns[:, j] = riskfree + b_series * market_excess + eps

    dates = pd.bdate_range("2015-01-01", periods=n_periods)
    cols = [f"ASSET_{i+1:02d}" for i in range(n_assets)]
    return pd.DataFrame(asset_returns, index=dates, columns=cols), \
           pd.Series(market_returns, index=dates, name="MARKET"), riskfree


def load_market_data(tickers, benchmark="^GSPC", start="2019-01-01", end=None,
                      riskfree_ticker="^IRX"):
    import yfinance as yf
    all_tickers = tickers + [benchmark, riskfree_ticker]
    raw = yf.download(all_tickers, start=start, end=end, auto_adjust=True)["Close"]
    rets = raw.pct_change().dropna(how="all")
    irx_level = raw[riskfree_ticker].reindex(rets.index).ffill()
    riskfree_daily = (irx_level / 100.0) / 252.0
    return rets[tickers].dropna(), rets[benchmark].dropna(), riskfree_daily.dropna()