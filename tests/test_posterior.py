import numpy as np
from dcm.inference.objectives import gaussian_log_posterior

def test_log_posterior_scalar(adapter, dummy_data, theta_init):
    sigma_obs = 0.1
    mu_prior = np.zeros_like(theta_init)
    sigma_prior = 1.0
    lp = gaussian_log_posterior(
        theta=theta_init,
        y_obs=dummy_data,
        adapter=adapter,
        sigma=sigma_obs,
        mu=mu_prior,
        sigma_prior=sigma_prior
    )
    assert np.isscalar(lp)

def test_log_posterior_increases_near_true(adapter, synthetic_data, theta_init):
    sigma_obs = 0.1
    mu_prior = np.zeros_like(theta_init)
    sigma_prior = 1.0

    # small perturbation should reduce posterior
    lp0 = gaussian_log_posterior(theta_init, synthetic_data, adapter, sigma_obs, mu_prior, sigma_prior)
    lp1 = gaussian_log_posterior(theta_init + 0.1, synthetic_data, adapter, sigma_obs, mu_prior, sigma_prior)
    assert lp1 <= lp0