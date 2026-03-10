import numpy as np
from dcm.inference.priors import gaussian_log_prior

def test_gaussian_prior_scalar_output():
    theta = np.zeros(3)
    mu = np.zeros(3)
    sigma = np.ones(3)
    lp = gaussian_log_prior(theta, mu, sigma)
    assert np.isscalar(lp)

def test_gaussian_prior_prefers_mean():
    mu = np.array([1.0, 2.0])
    sigma = np.array([1.0, 1.0])

    lp_mean = gaussian_log_prior(mu, mu, sigma)
    lp_far = gaussian_log_prior(mu + 1.0, mu, sigma)

    assert lp_mean > lp_far