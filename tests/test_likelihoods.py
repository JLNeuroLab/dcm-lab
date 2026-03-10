import numpy as np
from dcm.inference.likelihoods import gaussian_log_likelihood

def test_gaussian_likelihood_scalar_output():
    y_obs = np.zeros((5, 2))
    y_pred = np.zeros((5, 2))
    sigma = 1.0
    ll = gaussian_log_likelihood(y_obs, y_pred, sigma)
    # should be scalar
    assert np.isscalar(ll)

def test_gaussian_likelihood_prefers_correct_prediction():
    y_obs = np.array([[1.0, -1.0]])
    sigma = 0.5

    ll_match = gaussian_log_likelihood(y_obs, y_obs, sigma)
    ll_wrong = gaussian_log_likelihood(y_obs, y_obs + 0.5, sigma)

    assert ll_match > ll_wrong