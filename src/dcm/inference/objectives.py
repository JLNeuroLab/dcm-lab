import numpy as np

from dcm.inference.forward_adapter import ForwardAdapter
from dcm.inference.likelihoods import gaussian_log_likelihood
from dcm.inference.priors import gaussian_log_prior

Array = np.ndarray

def gaussian_log_posterior(
        theta: Array,
        y_obs: Array,
        adapter: ForwardAdapter,
        sigma: Array,
        mu: Array,
        sigma_prior: Array 
) -> float:
    
    y_pred = adapter(theta)

    ll = gaussian_log_likelihood(y_obs=y_obs, y_pred=y_pred, sigma=sigma)
    lp = gaussian_log_prior(theta=theta, mu=mu, sigma=sigma_prior)

    log_posterior = ll + lp

    return log_posterior