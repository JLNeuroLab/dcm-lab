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

        neuronal = self.forward_model.neuronal
        orig_A, orig_B, orig_C = neuronal.A, neuronal.B, neuronal.C

        try:
            # bypass nn.Module.__setattr__ type check so plain tensors derived
            # from theta can sit in _parameters — computation graph stays intact
            neuronal._parameters['A'] = A
            neuronal._parameters['B'] = B
            neuronal._parameters['C'] = C

            _, Y = self.forward_model.simulate(
                u=self.u_fn,
                t_eval=self.t_eval,
                x0=self.x0,
                z0=self.z0
            )
        finally:
            neuronal._parameters['A'] = orig_A
            neuronal._parameters['B'] = orig_B
            neuronal._parameters['C'] = orig_C

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
    
class EEGInferenceModel(nn.Module):

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

    def unpack_theta(self, theta):
        l = self.forward_model.l
        m = self.forward_model.neuronal.m

        cf_size = l * l
        cb_size = l * l
        cl_size = l * l
        p_size = l * m

        offset = 0

        C_F = theta[offset: offset + cf_size].reshape(l, l); offset += cf_size
        C_B = theta[offset:offset + cb_size].reshape(l, l); offset += cb_size
        C_L = theta[offset:offset + cl_size].reshape(l, l); offset += cl_size
        P   = theta[offset:offset + p_size ].reshape(l, m)

        return C_F, C_B, C_L, P
    
    def forward(self, theta):

        C_F, C_B, C_L, P = self.unpack_theta(theta)

        neuronal = self.forward_model.neuronal
        origin_cf, origin_cb, origin_cl, origin_p = neuronal.C_F, neuronal.C_B, neuronal.C_L, neuronal.P

        try:
            neuronal._parameters['C_F'] = C_F
            neuronal._parameters['C_B'] = C_B
            neuronal._parameters['C_L'] = C_L
            neuronal._parameters['P']   = P

            _, Y = self.forward_model.simulate(
                u=self.u_fn,
                t_eval=self.t_eval,
                z0=self.z0,
            )
        finally:
            neuronal._parameters['C_F'] = origin_cf
            neuronal._parameters['C_B'] = origin_cb
            neuronal._parameters['C_L'] = origin_cl
            neuronal._parameters['P']   = origin_p

        ll = self.likelihood_fn(
            y_obs=self.y_obs,
            y_pred=Y,
            sigma=self.sigma,
        )

        lp = self.prior_fn(
            theta=theta,
            mu=self.mu,
            sigma=self.sigma_prior,
        )

        return ll + lp