import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from fastapi.testclient import TestClient

from api.main import _parse_tickers, app

client = TestClient(app)


def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}


def test_openapi_lists_every_endpoint():
    paths = client.get("/openapi.json").json()["paths"]
    for expected in ("/beta/{ticker}", "/series/{ticker}", "/screen",
                     "/backtest", "/pricing-errors"):
        assert expected in paths


def test_parse_tickers_normalises_input():
    assert _parse_tickers(" aapl , msft ,, jpm ") == ["AAPL", "MSFT", "JPM"]


def test_parse_tickers_rejects_empty():
    with pytest.raises(ValueError, match="No tickers"):
        _parse_tickers("  ,  ,  ")


def test_screen_requires_tickers():
    assert client.get("/screen").status_code == 422


def test_backtest_requires_tickers():
    assert client.get("/backtest").status_code == 422


@pytest.mark.network
def test_bad_ticker_returns_400_not_500():
    r = client.get("/beta/__NOT_A_REAL_TICKER__")
    assert r.status_code == 400
    assert "detail" in r.json()


@pytest.mark.network
def test_known_ticker_returns_usable_betas():
    r = client.get("/beta/AAPL")
    assert r.status_code == 200
    body = r.json()
    for key in ("beta_plus", "beta_minus", "capm_beta", "asymmetry_ratio"):
        assert body[key] is not None, f"{key} came back null"
    assert body["n_up"] > 0 and body["n_down"] > 0
