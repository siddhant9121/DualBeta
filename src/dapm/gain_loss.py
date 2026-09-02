import numpy as np
def gain_loss_ratio(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    gains = np.mean(np.maximum(x, 0.0))     # average of only the positive values (0 elsewhere)
    losses = np.mean(-np.minimum(x, 0.0))   # average magnitude of only the negative values
    if losses == 0:
        return np.inf if gains > 0 else np.nan
    return float(gains / losses)


def relative_gain_loss_ratio(xi_upper: np.ndarray, xi_lower: np.ndarray) -> float:
    num = np.mean(np.asarray(xi_upper, dtype=float))
    den = np.mean(np.asarray(xi_lower, dtype=float))
    if den == 0:
        return np.inf if num > 0 else np.nan
    return float(num / den)