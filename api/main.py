import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from typing import List, Optional

from fastapi import FastAPI, HTTPException, Query

from dapm.data import load_market_data
from dapm.dapm_model import upper_lower_betas
from dapm.capm import capm_beta
from dapm.backtest import (
    rolling_dapm_betas,
    asymmetry_screen,
    backtest_asymmetry_portfolio,
)
from dapm.stats_tests import cross_sectional_pricing_errors, compare_pricing_errors

app = FastAPI(
    title="DAPM",
    description="Asymmetric beta analytics — upper/lower market betas per the "
                "Dichotomous Asset Pricing Model (Zou, 2005).",
    version="1.0.0",
)

DEFAULT_START = "2021-01-01"
DEFAULT_END = "2024-01-01"


def _parse_tickers(tickers: str) -> List[str]:
    out = [t.strip().upper() for t in tickers.split(",") if t.strip()]
    if not out:
        raise ValueError("No tickers supplied")
    return out


def _excess_returns(ticker: str, benchmark: str, start: str, end: Optional[str]):
    assets, market, rf = load_market_data(
        [ticker], benchmark=benchmark, start=start, end=end
    )

    if ticker not in assets.columns:
        raise ValueError(f"No data returned for {ticker}")

    xm = (market - rf).dropna()
    xi = (assets[ticker] - rf).dropna()
    common = xm.index.intersection(xi.index)

    if len(common) < 30:
        raise ValueError(
            f"Only {len(common)} overlapping observations for {ticker} against "
            f"{benchmark} — check the symbol and the date range"
        )

    return common, xm.loc[common], xi.loc[common]


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/beta/{ticker}")
def get_beta(
    ticker: str,
    start: str = DEFAULT_START,
    end: Optional[str] = DEFAULT_END,
    benchmark: str = "^GSPC",
) -> dict:
    """Upper and lower market betas for a single ticker."""
    try:
        ticker = ticker.strip().upper()
        _, xm_s, xi_s = _excess_returns(ticker, benchmark, start, end)
        xm, xi = xm_s.to_numpy(), xi_s.to_numpy()

        result = upper_lower_betas(xi, xm)
        cb = capm_beta(xi, xm)

        return {
            "ticker": ticker,
            "benchmark": benchmark,
            "start": start,
            "end": end,
            "beta_plus": result.beta_plus,
            "beta_minus": result.beta_minus,
            "best_beta": result.best_beta,
            "asymmetry_ratio": result.asymmetry_ratio,
            "capm_beta": cb,
            "n_up": result.n_up,
            "n_down": result.n_down,
        }

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/series/{ticker}")
def get_series(
    ticker: str,
    start: str = DEFAULT_START,
    end: Optional[str] = DEFAULT_END,
    benchmark: str = "^GSPC",
) -> dict:
    """Daily excess returns, in percent, with each day's market regime."""
    try:
        ticker = ticker.strip().upper()
        common, xm, xi = _excess_returns(ticker, benchmark, start, end)

        rows = [
            {
                "date": d.strftime("%Y-%m-%d"),
                "mkt": float(m) * 100,
                "stock": float(s) * 100,
                "regime": "up" if m >= 0 else "down",
            }
            for d, m, s in zip(common, xm.to_numpy(), xi.to_numpy())
        ]
        return {"ticker": ticker, "benchmark": benchmark, "n": len(rows), "rows": rows}

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/screen")
def get_screen(
    tickers: str = Query(..., description="Comma-separated, e.g. AAPL,MSFT,JPM"),
    start: str = DEFAULT_START,
    end: Optional[str] = DEFAULT_END,
    benchmark: str = "^GSPC",
    window: int = 126,
    step: int = 21,
) -> dict:
    """Cross-sectional ranking by asymmetry ratio, as of the latest estimation date."""
    try:
        symbols = _parse_tickers(tickers)
        assets, market, rf = load_market_data(
            symbols, benchmark=benchmark, start=start, end=end
        )

        panel = rolling_dapm_betas(assets, market, rf, window=window, step=step)
        if panel.empty:
            raise ValueError("No beta estimates — try a longer date range")

        snap = asymmetry_screen(panel)
        as_of = str(panel["date"].max().date())

        return {
            "as_of": as_of,
            "benchmark": benchmark,
            "n_assets": len(snap),
            "ranked": [
                {
                    "rank": i + 1,
                    "asset": r["asset"],
                    "beta_plus": float(r["beta_plus"]),
                    "beta_minus": float(r["beta_minus"]),
                    "capm_beta": float(r["capm_beta"]),
                    "asymmetry_ratio": float(r["asymmetry_ratio"]),
                }
                for i, r in enumerate(snap.to_dict(orient="records"))
            ],
        }

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/backtest")
def get_backtest(
    tickers: str = Query(..., description="Comma-separated, e.g. AAPL,MSFT,JPM"),
    start: str = DEFAULT_START,
    end: Optional[str] = DEFAULT_END,
    benchmark: str = "^GSPC",
    window: int = 126,
    step: int = 21,
    top_q: float = 0.3,
    bottom_q: float = 0.3,
) -> dict:
    """Walk-forward long-short backtest ranked on beta asymmetry.

    Returns are gross — no transaction costs, slippage, or short borrow fees — and
    the reported ratio is annualised return over annualised volatility, without
    subtracting the risk-free rate.
    """
    try:
        symbols = _parse_tickers(tickers)
        assets, market, rf = load_market_data(
            symbols, benchmark=benchmark, start=start, end=end
        )

        panel = rolling_dapm_betas(assets, market, rf, window=window, step=step)
        if panel.empty:
            raise ValueError("No beta estimates — try a longer date range")

        res = backtest_asymmetry_portfolio(assets, panel, top_q=top_q, bottom_q=bottom_q)

        return {
            "n_assets": int(panel["asset"].nunique()),
            "n_rebalances": int(panel["date"].nunique()),
            "annualized_return": res.annualized_return,
            "annualized_vol": res.annualized_vol,
            "return_vol_ratio": res.sharpe,
            "max_drawdown": res.max_drawdown,
            "gross_of_costs": True,
        }

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/pricing-errors")
def get_pricing_errors(
    tickers: str = Query(..., description="Comma-separated, e.g. AAPL,MSFT,JPM"),
    start: str = DEFAULT_START,
    end: Optional[str] = DEFAULT_END,
    benchmark: str = "^GSPC",
) -> dict:
    """CAPM vs DAPM conditional pricing errors, with a Wilcoxon signed-rank test.

    This comparison is in-sample: DAPM fits two betas against two conditional
    moments where CAPM fits one, so part of the improvement is structural.
    """
    try:
        symbols = _parse_tickers(tickers)
        assets, market, rf = load_market_data(
            symbols, benchmark=benchmark, start=start, end=end
        )

        df = cross_sectional_pricing_errors(assets, market, rf)
        if df.empty:
            raise ValueError("No assets produced pricing errors")

        summary = compare_pricing_errors(df)
        return {
            "n_assets": summary["n_assets"],
            "mean_capm_sq_error": summary["mean_capm_sq_error"],
            "mean_dapm_sq_error": summary["mean_dapm_sq_error"],
            "pct_error_reduction": summary["pct_error_reduction"],
            "wilcoxon_stat": None if summary["wilcoxon_stat"] != summary["wilcoxon_stat"]
                             else float(summary["wilcoxon_stat"]),
            "wilcoxon_pvalue": None if summary["wilcoxon_pvalue"] != summary["wilcoxon_pvalue"]
                               else float(summary["wilcoxon_pvalue"]),
            "in_sample": True,
        }

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
