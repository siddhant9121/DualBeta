from dataclasses import dataclass
import numpy as np


@dataclass
class CMOptionPrices:
    c_i: float
    p_i: float
    c_m: float
    p_m: float
    parity_gap: float


def price_cross_market_options(xi: np.ndarray, xm: np.ndarray, r0: float = 1.0) -> CMOptionPrices:
    xi = np.asarray(xi, dtype=float)
    xm = np.asarray(xm, dtype=float)
    up = xm > 0
    down = ~up

    c_i = float(np.mean(np.where(up, xi, 0.0)) / r0)
    p_i = float(np.mean(np.where(down, -xi, 0.0)) / r0)
    c_m = float(np.mean(np.where(up, xm, 0.0)) / r0)
    p_m = float(np.mean(np.where(down, -xm, 0.0)) / r0)

    return CMOptionPrices(c_i=c_i, p_i=p_i, c_m=c_m, p_m=p_m, parity_gap=c_i - p_i)


def implied_beta_from_prices(prices: CMOptionPrices) -> tuple[float, float]:
    beta_plus_implied = prices.c_i / prices.c_m if prices.c_m != 0 else np.nan
    beta_minus_implied = prices.p_i / prices.p_m if prices.p_m != 0 else np.nan
    return beta_plus_implied, beta_minus_implied
