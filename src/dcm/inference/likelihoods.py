import numpy as np

Array = np.ndarray

def gaussian_log_likelihood(
        y_obs: Array,
        y_pred: Array,
        sigma: Array,
) -> float:
    
    if y_obs.shape != y_pred.shape:
        raise ValueError("y_obs and y_pred must have same shape")
    
    residuals = y_obs - y_pred
    #T, l = y_obs.shape

    ll = -0.5 * np.sum((residuals / sigma) ** 2)
    # Drop constant terms
    #ll -= 0.5 * T * l * np.log(sigma ** 2 + 2 * np.pi)

    return ll