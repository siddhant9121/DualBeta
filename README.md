# DAPM — Asymmetric Beta Analytics

An implementation of the **Dichotomous Asset Pricing Model** (Zou, 2005), which splits a
stock's market beta into two conditional betas instead of one:

| | |
|---|---|
| **β⁺** | sensitivity on days the market's excess return was positive |
| **β⁻** | sensitivity on days it was negative |
| **asymmetry ratio** | β⁺ / β⁻ — how differently a stock behaves in rallies vs selloffs |

A single CAPM beta averages these two behaviours together. That average hides exactly the
thing investors care about most: how a stock behaves when the market falls.

## Why it matters

Two stocks can both show a CAPM beta of 1.0 and behave completely differently in a drawdown —
one falling 8% with the market and recovering, the other falling 14% and rebounding only 6%.
DAPM separates them.

## The estimator

Both betas use the paper's second-moment form, a regression through the origin on each
sub-sample, where `x_i` and `x_m` are **excess** returns (the risk-free rate subtracted):

```
β = E[x_m · x_i | regime] / E[x_m² | regime]
```

The split is at zero excess market return. This differs from Ang–Chen–Xing downside beta
(a conditional covariance) and from Bawa–Lindenberg lower-partial-moment beta.

## Layout

```
src/dapm/
  data.py                      yfinance loader + synthetic market simulator
  capm.py                      classical beta, Cov(x_i,x_m)/Var(x_m)
  dapm_model.py                upper_lower_betas() — the core estimator
  stats_tests.py               CAPM vs DAPM pricing errors + Wilcoxon test
  backtest.py                  rolling betas, asymmetry screen, long-short backtest
  gain_loss.py                 gain/loss ratio metrics
  cross_market_derivatives.py  regimes reframed as option payoffs
api/main.py                    FastAPI service — GET /beta/{ticker}
desktop/dashboard.py           PySide6 dashboard
notebooks/run_screener.py      end-to-end research pipeline
tests/                         pytest suite
```

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Running it

**Research pipeline** — rolling betas, cross-sectional screen, backtest, and the CAPM
comparison, across a 31-stock universe:

```bash
python notebooks/run_screener.py
```

**API service:**

```bash
uvicorn api.main:app --reload
```

Interactive docs at `http://localhost:8000/docs`.

| Endpoint | Purpose |
|---|---|
| `GET /health` | Liveness check |
| `GET /beta/{ticker}` | β⁺, β⁻, CAPM β and the asymmetry ratio for one ticker |
| `GET /series/{ticker}` | Daily excess returns with each day's market regime |
| `GET /screen?tickers=A,B,C` | Cross-sectional ranking by asymmetry ratio |
| `GET /backtest?tickers=A,B,C` | Walk-forward long-short performance |
| `GET /pricing-errors?tickers=A,B,C` | CAPM vs DAPM errors with a Wilcoxon test |

```bash
curl "localhost:8000/beta/AAPL"
curl "localhost:8000/screen?tickers=AAPL,MSFT,JPM,XOM,KO,PFE"
```

**Desktop dashboard:**

```bash
python desktop/dashboard.py
```

Single-ticker analysis across four tabs — regression statistics, rolling betas,
return distribution by regime, raw data with CSV export — plus a cross-sectional
screen tab that ranks a whole universe.

**Tests:**

```bash
pytest -q                    # everything
pytest -q -m "not network"   # offline only, no yfinance calls
```

## Validation

1. **Synthetic ground truth.** `simulate_single_factor_market()` generates assets with
   hard-coded betas. On a symmetric market the estimator recovers β⁺ ≈ β⁻ ≈ the planted
   value, confirming it reads true when no asymmetry exists.

2. **Real-data pricing errors.** For each asset, squared pricing error is measured against
   its conditional up-day and down-day mean returns, under CAPM's single beta and under
   DAPM's two. A Wilcoxon signed-rank test checks the improvement holds consistently across
   the universe rather than being driven by a few names.

3. **Walk-forward backtest.** Betas are estimated on a trailing 126-day window, the universe
   is ranked by asymmetry ratio, and the book goes long the top 30% / short the bottom 30%,
   rebalanced every 21 trading days. Returns are always realised forward of the estimation
   window.

### Results

`python notebooks/run_screener.py` over a 31-stock universe, 2021-01-01 to 2024-01-01,
benchmarked against `^GSPC`:

| Metric | Value |
|---|---|
| Mean CAPM squared pricing error | 7.2e-7 |
| Mean DAPM squared pricing error | 4.0e-7 |
| Error reduction | 43.9% |
| Wilcoxon signed-rank p-value | 0.0059 |
| Long-short annualised return | 11.70% |
| Annualised volatility | 16.41% |
| Return / volatility | 0.713 |
| Max drawdown | −14.48% |

Read these alongside the limitations below — particularly that the pricing-error
comparison is in-sample and the backtest figures are gross of costs.

## Known limitations

These are deliberate and worth stating plainly:

- **The pricing-error comparison is in-sample.** DAPM fits two parameters where CAPM fits
  one, against two conditional moments, so some of the improvement is structural. The
  Wilcoxon test shows the gain is *consistent across assets*, not that DAPM predicts better
  out of sample.
- **Backtest performance is gross.** No transaction costs, slippage, or short borrow fees.
  `_perf_stats` reports return ÷ volatility without subtracting the risk-free rate, so it is
  not a true Sharpe ratio.
- **The universe has survivorship bias.** The 31 tickers were chosen knowing they survived
  to 2024.
- **yfinance is not execution-grade.** It is unofficial, delayed, and revises silently.
  `auto_adjust=True` also restates past prices after corporate actions, which is a
  look-ahead concern for any live use.
- **Two estimators coexist.** `src/dapm` and the API use the paper's through-origin
  moment estimator. The dashboard's β⁺ and β⁻ use OLS with an intercept (`linreg`),
  which it also uses for standard errors, t-statistics, and R². Its split betas can
  therefore differ materially from the API's. The dashboard shows a note about this.
- **This is a research tool, not a trading system.** The signal is daily-frequency with a
  monthly rebalance; it does not update intraday and there is no broker integration.

## Reference

Zou, L. (2005). *The dichotomous asset pricing model.*
