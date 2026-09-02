import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from fastapi import FastAPI, HTTPException

from dapm.data import load_market_data
from dapm.dapm_model import upper_lower_betas
from dapm.capm import capm_beta

app = FastAPI()


@app.get("/beta/{ticker}")
def get_beta(ticker: str) -> dict:
    try:
        # Step 1: get real data — [ticker] wraps the single string into a
        # one-item list, since load_market_data expects a list of tickers.
        assets, market, rf = load_market_data(
            [ticker], start="2021-01-01", end="2024-01-01"
        )

        # Step 2: compute excess returns (subtract risk-free rate)
        xm = (market - rf).to_numpy()
        xi = (assets[ticker] - rf).to_numpy()

        # Step 3: run the DAPM math — same functions you already tested
        result = upper_lower_betas(xi, xm)
        cb = capm_beta(xi, xm)

        # Step 4: package as a plain dict — FastAPI turns this into JSON
        return {
            "ticker": ticker,
            "beta_plus": result.beta_plus,
            "beta_minus": result.beta_minus,
            "best_beta": result.best_beta,
            "asymmetry_ratio": result.asymmetry_ratio,
            "capm_beta": cb,
        }

    except Exception as e:
        # If the ticker is invalid or data fetch fails, return a proper
        # HTTP error instead of crashing the whole server.
        raise HTTPException(status_code=400, detail=str(e))