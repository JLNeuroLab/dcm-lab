import numpy as np
import torch

from dcm.torch.neuronal_torch import (
    BilinearNeuronalTorch,
    BilinearParametersTorch,
)

from dcm.torch.hemodynamic_torch import (
    HemodynamicBalloonTorch,
    HemodynamicParametersTorch,
)

from dcm.torch.forward_torch import ForwardModelTorch


# ---------------------------------------------------------------------
# FIXTURE BUILDER
# ---------------------------------------------------------------------

def build_model():
    """
    Minimal deterministic 1-region DCM for stable tests.
    """

    l, m = 1, 1

    # ---------------- Neuronal ----------------
    A = torch.tensor([[-0.5]], dtype=torch.float32)
    B = torch.zeros((m, l, l), dtype=torch.float32)
    C = torch.tensor([[1.0]], dtype=torch.float32)

    neuronal = BilinearNeuronalTorch(
        BilinearParametersTorch(A=A, B=B, C=C)
    )

    # ---------------- Hemodynamic ----------------

    hemo = HemodynamicBalloonTorch(
        HemodynamicParametersTorch.with_defaults(l)
    )

    model = ForwardModelTorch(neuronal, hemo)

    t_eval = torch.linspace(0.0, 60.0, 601)
    return model, t_eval, l, m


# ---------------------------------------------------------------------
# 1. ZERO INPUT: EQUILIBRIUM
# ---------------------------------------------------------------------

def test_forward_zero_input():
    """
    Under zero input, system should remain near equilibrium:
    neuronal stays ~0 and BOLD stays ~0.
    """

    model, t_eval, l, m = build_model()

    def u(t):
        return torch.zeros(m)

    S, Y = model.simulate(u=u, t_eval=t_eval)

    z = S[:, :l]

    assert torch.allclose(z, torch.zeros_like(z), atol=1e-5)
    assert torch.allclose(Y, torch.zeros_like(Y), atol=1e-5)

# ---------------------------------------------------------------------
# 4. FINITE STABILITY
# ---------------------------------------------------------------------

def test_forward_finite():
    """
    Ensures no NaN or Inf appear in simulation.
    """

    model, t_eval, l, m = build_model()

    def u(t):
        return torch.tensor([1.0 if 10.0 <= t <= 20.0 else 0.0])

    S, Y = model.simulate(u=u, t_eval=t_eval)

    assert torch.all(torch.isfinite(S))
    assert torch.all(torch.isfinite(Y))


# ---------------------------------------------------------------------
# 5. STRUCTURAL CONSISTENCY
# ---------------------------------------------------------------------

def test_forward_shapes():
    """
    Ensures correct dimensional structure:
    S = (T, 5l), Y = (T, l)
    """

    model, t_eval, l, m = build_model()

    def u(t):
        return torch.zeros(m)

    S, Y = model.simulate(u=u, t_eval=t_eval)

    assert S.shape[1] == 5 * l
    assert Y.shape[1] == l


# ---------------------------------------------------------------------
# 6. DETERMINISM
# ---------------------------------------------------------------------

def test_forward_determinism():
    """
    Same input must produce identical trajectories.
    """

    model, t_eval, l, m = build_model()

    def u(t):
        return torch.tensor([1.0 if t > 10.0 else 0.0])

    S1, Y1 = model.simulate(u=u, t_eval=t_eval)
    S2, Y2 = model.simulate(u=u, t_eval=t_eval)

    assert torch.allclose(S1, S2, atol=1e-6)
    assert torch.allclose(Y1, Y2, atol=1e-6)