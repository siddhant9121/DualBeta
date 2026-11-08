import numpy as np


def capm_beta(xi: np.ndarray, xm: np.ndarray) -> float:
    cov_matrix = np.cov(xi, xm, ddof=1)
    cov_xi_xm = cov_matrix[0][1]

    var_xm = np.var(xm, ddof=1)

    if var_xm == 0:
        raise ValueError("Zero market variance — cannot compute beta")

    beta = cov_xi_xm / var_xm
    return float(beta)
