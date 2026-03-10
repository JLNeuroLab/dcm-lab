import numpy as np

Array = np.ndarray

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