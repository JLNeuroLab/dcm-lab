# =============================================================================
# NUMPY VERSION (SciPy MAP OBJECTIVE)
# =============================================================================

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

# =============================================================================
# PYTORCH VERSION (DIFFERENTIABLE MAP OBJECTIVE)
# =============================================================================

import torch
import torch.nn as nn
Tensor = torch.Tensor
class DCMInferenceModel(nn.Module):
    """
    Differentiable MAP objective for DCM.

    Computes the log-posterior:
        log p(y | θ) + log p(θ)

    This module does NOT perform optimization.
    It is used by external optimizers (torch.optim, LBFGS, etc.).
    """

    def __init__(
        self,
        forward_model,
        likelihood_fn,
        prior_fn,
        y_obs,
        sigma,
        mu,
        sigma_prior,
        t_eval,
        u_fn,
        z0,
        x0,
    ):
        super().__init__()

        self.forward_model = forward_model
        self.likelihood_fn = likelihood_fn
        self.prior_fn = prior_fn

        self.register_buffer("y_obs", y_obs)
        self.register_buffer("sigma", sigma)
        self.register_buffer("mu", mu)
        self.register_buffer("sigma_prior", sigma_prior)
        self.register_buffer("t_eval", t_eval)

        self.u_fn = u_fn
        self.z0 = z0
        self.x0 = x0

    def unpack_theta(self, theta):
        l = self.forward_model.l
        m = self.forward_model.neuronal.m

        A_size = l * l
        B_size = l * l * m
        C_size = l * m

        A = theta[:A_size].reshape(l, l)
        B = theta[A_size:A_size + B_size].reshape(m, l, l)
        C = theta[A_size + B_size:].reshape(l, m)

        return A, B, C

    def forward(self, theta):

        A, B, C = self.unpack_theta(theta)

        self.forward_model.neuronal.A = A
        self.forward_model.neuronal.B = B
        self.forward_model.neuronal.C = C

        _, Y = self.forward_model.simulate(
            u=self.u_fn,
            t_eval=self.t_eval,
            x0=self.x0,
            z0=self.z0
        )

        ll = self.likelihood_fn(
            y_obs=self.y_obs,
            y_pred=Y,
            sigma=self.sigma
        )

        lp = self.prior_fn(
            theta=theta,
            mu=self.mu,
            sigma=self.sigma_prior
        )

        return ll + lp