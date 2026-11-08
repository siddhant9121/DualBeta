from __future__ import annotations
from dataclasses import dataclass
import numpy as np


@dataclass
class DAPMResult:
    beta_plus: float
    beta_minus: float
    best_beta: float
    asymmetry_ratio: float
    n_up: int
    n_down: int


def best_beta(xi: np.ndarray, xm: np.ndarray) -> float:
    denom = np.mean(xm ** 2)
    if denom == 0:
        raise ValueError("Zero E(x_m^2) — cannot compute best beta")
    return float(np.mean(xm * xi) / denom)


def upper_lower_betas(xi: np.ndarray, xm: np.ndarray, min_obs: int = 5) -> DAPMResult:
    xi = np.asarray(xi, dtype=float)
    xm = np.asarray(xm, dtype=float)

    up_mask = xm > 0
    down_mask = ~up_mask

    n_up, n_down = int(up_mask.sum()), int(down_mask.sum())
    if n_up < min_obs or n_down < min_obs:
        raise ValueError(
            f"Not enough observations per regime (up={n_up}, down={n_down}, "
            f"need >= {min_obs} each) — use a longer window."
        )

    xm_up, xi_up = xm[up_mask], xi[up_mask]
    xm_dn, xi_dn = xm[down_mask], xi[down_mask]

    denom_up = np.mean(xm_up ** 2)
    denom_dn = np.mean(xm_dn ** 2)
    if denom_up == 0 or denom_dn == 0:
        raise ValueError("Zero within-regime E(x_m^2) — cannot compute DAPM betas")

    beta_plus = float(np.mean(xm_up * xi_up) / denom_up)
    beta_minus = float(np.mean(xm_dn * xi_dn) / denom_dn)
    bbeta = best_beta(xi, xm)

    ratio = beta_plus / beta_minus if beta_minus != 0 else np.nan

    return DAPMResult(
        beta_plus=beta_plus,
        beta_minus=beta_minus,
        best_beta=bbeta,
        asymmetry_ratio=ratio,
        n_up=n_up,
        n_down=n_down,
    )
