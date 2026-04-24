import numpy as np
import torch

Array = np.ndarray
Tensor = torch.Tensor

def gaussian_log_prior(
        theta: Array,
        mu: Array,
        sigma: Array
) -> float:
    
    if theta.shape != mu.shape:
        raise ValueError("theta and mu must have same shape")
    
    diff = theta - mu
    #n = theta.size

    lp = -0.5 * np.sum((diff / sigma)**2)
    # Drop the constant terms
    # lp -= 0.5 * n * np.log(sigma ** 2 * 2 * np.pi)
    return lp

def gaussian_log_prior_torch(
    theta: Tensor,
    mu: Tensor,
    sigma: Tensor,
) -> Tensor:
    """
    Torch equivalent of Gaussian prior (diagonal covariance):

        lp = -0.5 * Σ ((θ - μ) / σ)^2

    Constant terms are omitted (consistent with likelihood).
    """

    if theta.shape != mu.shape:
        raise ValueError("theta and mu must have same shape")

    sigma = torch.clamp(sigma, min=1e-8)

    residuals = theta - mu
    lp = -0.5 * torch.sum((residuals / sigma) ** 2)

    return lp