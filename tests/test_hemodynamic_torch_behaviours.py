import numpy as np
import torch

from dcm.models.hemodynamic_balloon import (
        HemodynamicBalloonModel,
        HemodynamicParameters,
)
from dcm.torch.hemodynamic_torch import HemodynamicBalloonTorch, HemodynamicParametersTorch


def test_hemodynamic_torch_matches_numpy():
    """
    Checks element-wise agreement between NumPy and PyTorch implementations
    of the hemodynamic dynamics for identical states and inputs.
    """
    l = 3

    kappa = np.random.randn(l) * 0.1
    gamma = np.random.randn(l) * 0.1
    tau   = np.random.randn(l) * 0.1 + 1.0
    alpha = np.random.randn(l) * 0.1 + 0.3
    rho   = np.random.randn(l) * 0.1 + 0.3

    params_np = HemodynamicParameters(l, kappa, gamma, tau, alpha, rho, 0.02)
    model_np = HemodynamicBalloonModel(params_np)

    params_torch = HemodynamicParametersTorch(
        l=l,
        kappa=torch.tensor(kappa, dtype=torch.float32),
        gamma=torch.tensor(gamma, dtype=torch.float32),
        tau=torch.tensor(tau, dtype=torch.float32),
        alpha=torch.tensor(alpha, dtype=torch.float32),
        rho=torch.tensor(rho, dtype=torch.float32),
    )
    model_torch = HemodynamicBalloonTorch(params_torch)

    x = np.random.randn(4 * l)
    z = np.random.randn(l)

    dx_np = model_np.dynamics(0.0, x, z)

    dx_torch = model_torch.dynamics(
        0.0,
        torch.tensor(x, dtype=torch.float32),
        torch.tensor(z, dtype=torch.float32),
    ).detach().numpy()

    assert np.allclose(dx_np, dx_torch, atol=1e-5)

def test_hemodynamic_torch_finite():
    """
    Ensures hemodynamic dynamics produce finite values and correct output shape
    under random valid inputs.
    """
    import torch

    l = 3

    params = HemodynamicParametersTorch(
        l=l,
        kappa=torch.randn(l) * 0.1,
        gamma=torch.randn(l) * 0.1,
        tau=torch.randn(l) * 0.1 + 1.0,
        alpha=torch.randn(l) * 0.1 + 0.3,
        rho=torch.randn(l) * 0.1 + 0.3,
    )

    model = HemodynamicBalloonTorch(params)

    x = torch.randn(4 * l)
    z = torch.randn(l)

    dx = model.dynamics(0.0, x, z)

    assert torch.all(torch.isfinite(dx))
    assert dx.shape == (4 * l,)

def test_hemodynamic_bold_sanity():
    """
    Verifies that the BOLD observation function returns finite outputs
    with correct dimensionality for random states.
    """
    import torch

    l = 3

    params = HemodynamicParametersTorch(
        l=l,
        kappa=torch.ones(l),
        gamma=torch.ones(l),
        tau=torch.ones(l),
        alpha=torch.ones(l) * 0.3,
        rho=torch.ones(l) * 0.3,
    )

    model = HemodynamicBalloonTorch(params)

    x = torch.randn(4 * l)

    y = model.bold(x)

    assert y.shape == (l,)
    assert torch.all(torch.isfinite(y))