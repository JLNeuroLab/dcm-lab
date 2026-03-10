import numpy as np

Array = np.ndarray

def gaussian_log_likelihood(
        y_obs: Array,
        y_preds: Array,
        sigma: float,
) -> float:
    
    residuals = y_obs - y_preds
    T, l = y_obs.shape

    ll = -0.5 * np.sum(residuals ** 2) / sigma ** 2
    ll -= 0.5 * T * l * np.log(sigma ** 2 + 2 * np.pi)

    return ll