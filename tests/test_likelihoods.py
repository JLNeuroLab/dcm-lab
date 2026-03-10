import numpy as np
from dcm.inference.likelihoods import gaussian_log_likelihood

def test_gaussian_likelihood_scalar_output():
    y_obs = np.zeros((5, 2))
    y_pred = np.zeros((5, 2))
    sigma = 1.0
    ll = gaussian_log_likelihood(y_obs, y_pred, sigma)
    # should be scalar
    assert np.isscalar(ll)

def test_gaussian_likelihood_max_when_match():
    y_obs = np.array([[1.0, -1.0]])
    y_pred = np.array([[1.0, -1.0]])
    sigma = 0.5
    ll = gaussian_log_likelihood(y_obs, y_pred, sigma)
    # should be highest possible value (closest to zero)
    assert ll > -1e-10