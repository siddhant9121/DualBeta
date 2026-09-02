import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
from dapm.data import simulate_single_factor_market
from dapm.dapm_model import upper_lower_betas


def test_single_factor_normal_market_gives_symmetric_betas():
    """
    Under a linear single-factor model (r_i = r_0 + b_i * x_m + noise),
    beta_plus and beta_minus should both land close to the TRUE beta we
    picked ourselves — proving upper_lower_betas actually recovers the
    right answer, not just runs without crashing.
    """
    true_betas = np.array([0.5, 1.0, 1.5, 2.0])
    assets, market, rf = simulate_single_factor_market(
        n_periods=4000, betas=true_betas, seed=1
    )
    xm = (market - rf).to_numpy()

    for j, col in enumerate(assets.columns):
        xi = (assets[col] - rf).to_numpy()
        result = upper_lower_betas(xi, xm)
        true_b = true_betas[j]

        print(f"{col}: true={true_b}, beta_plus={result.beta_plus:.3f}, "
              f"beta_minus={result.beta_minus:.3f}")

        assert abs(result.beta_plus - true_b) < 0.15
        assert abs(result.beta_minus - true_b) < 0.15