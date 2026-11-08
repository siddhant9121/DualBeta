from dataclasses import dataclass
import numpy as np
import pandas as pd
from .dapm_model import upper_lower_betas
from .capm import capm_beta


def rolling_dapm_betas(asset_returns, market_returns, riskfree, window=126,
                        step=21, min_obs_per_regime=5):
    rf = pd.Series(riskfree, index=asset_returns.index) if isinstance(riskfree, (int, float)) \
        else riskfree.reindex(asset_returns.index).ffill()
    xm_full = market_returns - rf
    records = []
    idx = asset_returns.index
    for end in range(window, len(idx) + 1, step):
        window_dates = idx[end - window:end]
        as_of = idx[end - 1]
        xm = xm_full.loc[window_dates].to_numpy()
        for asset in asset_returns.columns:
            xi = (asset_returns[asset].loc[window_dates] - rf.loc[window_dates]).to_numpy()
            if np.any(np.isnan(xi)) or np.any(np.isnan(xm)):
                continue
            try:
                res = upper_lower_betas(xi, xm, min_obs=min_obs_per_regime)
                cb = capm_beta(xi, xm)
            except ValueError:
                continue
            records.append({"date": as_of, "asset": asset, "beta_plus": res.beta_plus,
                             "beta_minus": res.beta_minus, "best_beta": res.best_beta,
                             "capm_beta": cb, "asymmetry_ratio": res.asymmetry_ratio,
                             "n_up": res.n_up, "n_down": res.n_down})
    return pd.DataFrame.from_records(records)


def asymmetry_screen(beta_panel, as_of=None):
    as_of = as_of or beta_panel["date"].max()
    snap = beta_panel[beta_panel["date"] == as_of].copy()
    snap = snap.replace([np.inf, -np.inf], np.nan).dropna(subset=["asymmetry_ratio"])
    return snap.sort_values("asymmetry_ratio", ascending=False).reset_index(drop=True)


@dataclass
class BacktestResult:
    equity_curve: pd.Series
    daily_returns: pd.Series
    annualized_return: float
    annualized_vol: float
    sharpe: float
    max_drawdown: float


def _perf_stats(daily_rets, periods_per_year=252):
    equity = (1 + daily_rets).cumprod()
    ann_ret = equity.iloc[-1] ** (periods_per_year / len(daily_rets)) - 1 if len(daily_rets) else np.nan
    ann_vol = daily_rets.std(ddof=1) * np.sqrt(periods_per_year)
    sharpe = ann_ret / ann_vol if ann_vol not in (0, np.nan) else np.nan
    drawdown = (equity / equity.cummax() - 1).min()
    return BacktestResult(equity, daily_rets, float(ann_ret), float(ann_vol), float(sharpe), float(drawdown))


def backtest_asymmetry_portfolio(asset_returns, beta_panel, top_q=0.3, bottom_q=0.3):
    dates = sorted(beta_panel["date"].unique())
    daily_rets = []
    for i, d in enumerate(dates):
        snap = asymmetry_screen(beta_panel, as_of=d)
        if snap.empty:
            continue
        n = len(snap)
        longs = snap.head(max(1, int(np.ceil(n * top_q))))["asset"].tolist()
        shorts = snap.tail(max(1, int(np.ceil(n * bottom_q))))["asset"].tolist()
        hold_end = dates[i + 1] if i + 1 < len(dates) else asset_returns.index.max()
        hold_dates = asset_returns.index[(asset_returns.index > d) & (asset_returns.index <= hold_end)]
        if len(hold_dates) == 0:
            continue
        period_ret = asset_returns.loc[hold_dates, longs].mean(axis=1) - asset_returns.loc[hold_dates, shorts].mean(axis=1)
        daily_rets.append(period_ret)
    full = pd.concat(daily_rets).sort_index()
    full = full[~full.index.duplicated(keep="first")]
    return _perf_stats(full)
