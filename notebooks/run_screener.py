import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from dapm.data import load_market_data
from dapm.backtest import rolling_dapm_betas, asymmetry_screen

tickers = [
    # Tech
    "AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "ADBE", "CRM",
    # Finance
    "JPM", "BAC", "GS", "MS", "V", "MA",
    # Healthcare
    "JNJ", "UNH", "PFE", "MRK",
    # Energy
    "XOM", "CVX",
    # Consumer
    "KO", "PEP", "WMT", "MCD", "NKE",
    # Industrials
    "BA", "CAT", "GE",
    # Telecom / Media
    "DIS", "NFLX", "VZ",
]
assets, market, rf = load_market_data(tickers, start="2021-01-01", end="2024-01-01")

panel = rolling_dapm_betas(assets, market, rf, window=126, step=21)
print(f"Computed {len(panel)} beta observations across {panel['asset'].nunique()} stocks")

latest_screen = asymmetry_screen(panel)
print("\nLatest cross-sectional screen, ranked by asymmetry_ratio (beta_plus/beta_minus):")
print(latest_screen[["asset", "beta_plus", "beta_minus", "asymmetry_ratio"]].to_string(index=False))
from dapm.backtest import backtest_asymmetry_portfolio

result = backtest_asymmetry_portfolio(assets, panel, top_q=0.3, bottom_q=0.3)
print(f"\nBacktest results:")
print(f"Annualized return: {result.annualized_return:.2%}")
print(f"Annualized volatility: {result.annualized_vol:.2%}")
print(f"Sharpe ratio: {result.sharpe:.3f}")
print(f"Max drawdown: {result.max_drawdown:.2%}")
from dapm.stats_tests import cross_sectional_pricing_errors, compare_pricing_errors

err_df = cross_sectional_pricing_errors(assets, market, rf)
summary = compare_pricing_errors(err_df)
print(f"\nDAPM vs CAPM pricing error comparison ({summary['n_assets']} assets):")
print(f"Mean CAPM squared error: {summary['mean_capm_sq_error']:.8f}")
print(f"Mean DAPM squared error: {summary['mean_dapm_sq_error']:.8f}")
print(f"Error reduction: {summary['pct_error_reduction']:.1f}%")
print(f"Wilcoxon p-value: {summary['wilcoxon_pvalue']}")