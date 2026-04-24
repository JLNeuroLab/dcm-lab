import numpy as np
import torch

Array = np.ndarray
Tensor = torch.Tensor

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

def gaussian_log_likelihood_torch(
    y_obs: Tensor,
    y_pred: Tensor,
    sigma: Tensor,
) -> Tensor:
    """
    Torch equivalent of NumPy Gaussian log-likelihood:

        ll = -0.5 * Σ ((y_obs - y_pred) / sigma)^2

    Constant terms are dropped (as in your NumPy version).
    """

    if y_obs.shape != y_pred.shape:
        raise ValueError("y_obs and y_pred must have same shape")

    sigma = torch.clamp(sigma, min=1e-8)

    residuals = y_obs - y_pred
    ll = -0.5 * torch.sum((residuals / sigma) ** 2)

    return ll