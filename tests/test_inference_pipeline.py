import torch
import numpy as np
import pytest

from dcm.torch.neuronal_torch import (
    NeuronalBilinearTorch,
    BilinearParametersTorch,
)
from dcm.torch.hemodynamic_torch import (
    HemodynamicBalloonTorch,
    HemodynamicParametersTorch,
)
from dcm.torch.forward_torch import ForwardModelTorch
from dcm.inference.optim import map_estimation_torch
from dcm.inference.objectives import DCMInferenceModel
from dcm.inference.likelihoods import gaussian_log_likelihood_torch
from dcm.inference.priors import gaussian_log_prior_torch


@pytest.fixture
def dcm_setup():
    l, m = 2, 1
    device = torch.device("cpu")

    # ---------------- Neuronal ----------------
    A = torch.tensor([[-0.5, 0.1],
                      [ 0.0, -0.6]], dtype=torch.float32)

    B = torch.zeros((m, l, l), dtype=torch.float32)
    C = torch.tensor([[1.0], [0.5]], dtype=torch.float32)

    neuronal = NeuronalBilinearTorch(
        BilinearParametersTorch(A, B, C)
    )

    # ---------------- Hemodynamic ----------------
    hemodynamic = HemodynamicBalloonTorch(
        HemodynamicParametersTorch.with_defaults(l)
    )

    forward = ForwardModelTorch(
        neuronal_model=neuronal,
        hemodynamic_model=hemodynamic
    )

    return forward, l, m

# =============================================================================
# 1. OBJECTIVE SANITY TEST
# =============================================================================

def test_inference_objective_finite_differentiable(dcm_setup):

    forward, l, m = dcm_setup

    T = 100
    t_eval = torch.linspace(0, 10, T)

    def u(t):
        return torch.tensor([1.0 if t > 2.0 else 0.0])

    z0 = torch.zeros(l)
    x0 = forward.hemodynamic.initial_state()

    y_obs = torch.zeros((T, l))

    inference_model = DCMInferenceModel(
        forward_model=forward,
        likelihood_fn=gaussian_log_likelihood_torch,
        prior_fn=gaussian_log_prior_torch,
        y_obs=y_obs,
        sigma=torch.ones_like(y_obs),
        mu=torch.zeros(l),
        sigma_prior=torch.ones(l),
        t_eval=t_eval,
        u_fn=u,
        z0=z0,
        x0=x0,
    )
    theta = torch.randn(l, requires_grad=True)

    loss = inference_model(theta)

    # ------------------- checks -----------------
    assert torch.isfinite(loss)

    loss.backward()

    assert theta.grad is not None
    assert torch.isfinite(theta.grad).all()

# =============================================================================
# 2. OPTIMIZER SANITY TEST (MAP RECOVERY SMOKE TEST)
# =============================================================================

def test_map_optimizer_runs(dcm_setup):

    forward, l, m = dcm_setup

    T = 50
    t_eval = torch.linspace(0, 5, T)

    def u(t):
        return torch.tensor([1.0 if t > 1 else 0.0])

    z0 = torch.zeros(l)
    x0 = forward.hemodynamic.initial_state()

    # ---------------- ground truth theta ----------------
    theta_true = torch.randn(l)

    with torch.no_grad():
        _, y_obs = forward.simulate(
            u=u,
            t_eval=t_eval,
            z0=z0,
            x0=x0,
        )

    # ---------------- inference model ----------------
    inference_model = DCMInferenceModel(
        forward_model=forward,
        likelihood_fn=gaussian_log_likelihood_torch,
        prior_fn=gaussian_log_prior_torch,
        y_obs=y_obs,
        sigma=torch.ones_like(y_obs),
        mu=torch.zeros(l),
        sigma_prior=torch.ones(l),
        t_eval=t_eval,
        u_fn=u,
        z0=z0,
        x0=x0,
    )

    theta_init = torch.randn(l, requires_grad=True)

    theta_hat, trace = map_estimation_torch(
        model=inference_model,
        theta=theta_init,
        n_steps=20,
        method="lbfgs",
        verbose=False,
    )

    # ---------------- checks ----------------
    assert torch.isfinite(theta_hat).all()
    assert torch.isfinite(trace).all()

    # optimization should not explode
    assert trace[-1] <= trace[0] + 1e3  # loose stability bound

    print("theta_true:", theta_true)
    print("theta_hat:", theta_hat)