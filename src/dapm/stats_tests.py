import numpy as np
import pandas as pd
from scipy import stats
from .dapm_model import upper_lower_betas
from .capm import capm_beta


def cross_sectional_pricing_errors(asset_returns, market_returns, riskfree):
    rf = pd.Series(riskfree, index=asset_returns.index) if isinstance(riskfree, (int, float)) \
        else riskfree.reindex(asset_returns.index).ffill()
    xm = (market_returns - rf).to_numpy()
    ex_m_plus = np.mean(xm[xm > 0])
    ex_m_minus = np.mean(xm[xm <= 0])

    rows = []
    for asset in asset_returns.columns:
        xi = (asset_returns[asset] - rf).to_numpy()
        if np.any(np.isnan(xi)):
            continue
        up = xm > 0
        ex_i_plus, ex_i_minus = np.mean(xi[up]), np.mean(xi[~up])
        try:
            res = upper_lower_betas(xi, xm)
        except ValueError:
            continue
        cb = capm_beta(xi, xm)
        capm_err = (ex_i_plus - cb * ex_m_plus) ** 2 + (ex_i_minus - cb * ex_m_minus) ** 2
        dapm_err = (ex_i_plus - res.beta_plus * ex_m_plus) ** 2 + (ex_i_minus - res.beta_minus * ex_m_minus) ** 2
        rows.append({"asset": asset, "capm_beta": cb, "beta_plus": res.beta_plus,
                      "beta_minus": res.beta_minus, "capm_sq_error": capm_err, "dapm_sq_error": dapm_err})
    return pd.DataFrame(rows)


def compare_pricing_errors(df):
    capm_err, dapm_err = df["capm_sq_error"].to_numpy(), df["dapm_sq_error"].to_numpy()
    mean_capm, mean_dapm = float(np.mean(capm_err)), float(np.mean(dapm_err))
    improvement = 100.0 * (mean_capm - mean_dapm) / mean_capm if mean_capm else np.nan
    if len(capm_err) >= 8 and not np.allclose(capm_err, dapm_err):
        stat, pvalue = stats.wilcoxon(capm_err, dapm_err, alternative="greater")
    else:
        stat, pvalue = np.nan, np.nan
    return {"mean_capm_sq_error": mean_capm, "mean_dapm_sq_error": mean_dapm,
            "pct_error_reduction": improvement, "wilcoxon_stat": stat,
            "wilcoxon_pvalue": pvalue, "n_assets": len(df)}
