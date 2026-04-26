import torch
import pytest

from dcm.torch.neuronal_torch import BilinearNeuronalTorch, BilinearParametersTorch
from ml.mlp import ResidualMLP 
from hybrid.residual_dcm import ResidualDCM
from dcm.simulate.integrators import rk4_integrate_torch


# ------------------------------------------------------------
# FIXTURE (setup condiviso)
# ------------------------------------------------------------
@pytest.fixture
def model_setup():
    torch.manual_seed(0)

    l = 2
    m = 1

    A = torch.tensor([[-0.5, 0.1],
                      [0.0, -0.4]])

    B = torch.zeros((m, l, l))
    C = torch.zeros((l, m))

    # fake hemodynamic minimal model
    class DummyHemo:
        def __init__(self, l):
            self.l = l

        def initial_state(self, x0=None):
            return torch.zeros(4 * self.l)

        def dynamics(self, t, x, z):
            return torch.zeros_like(x)

        def bold(self, x):
            return x[:self.l]

    params = BilinearParametersTorch(A=A, B=B, C=C)
    dcm = BilinearNeuronalTorch(params)
    hemo = DummyHemo(l)

    mlp = ResidualMLP(l, m)

    model = ResidualDCM(
        bilinear=dcm,
        hemodynamic=hemo,
        mlp=mlp,
        alpha=0.1
    )

    return model, mlp, l, m


# ------------------------------------------------------------
# 1. FORWARD SHAPE
# ------------------------------------------------------------
def test_forward_shape(model_setup):
    model, _, l, m = model_setup

    z = torch.randn(l)
    x = torch.zeros(4 * l)
    u = torch.randn(m)

    state = torch.cat([z, x])

    ds = model.dynamics(0.0, state, u)

    assert ds.shape == (5 * l,)


# ------------------------------------------------------------
# 2. GRADIENT FLOWS TO MLP
# ------------------------------------------------------------
def test_gradient_flow(model_setup):
    model, mlp, l, m = model_setup

    z = torch.randn(l, requires_grad=True)
    x = torch.zeros(4 * l)
    u = torch.randn(m)

    state = torch.cat([z, x])

    ds = model.dynamics(0.0, state, u)
    loss = ds.pow(2).mean()

    loss.backward()

    grad_norm = sum(
        p.grad.norm().item()
        for p in mlp.parameters()
        if p.grad is not None
    )

    assert grad_norm > 0.0


# # ------------------------------------------------------------
# # 3. SIMULATION DOES NOT CRASH
# # ------------------------------------------------------------
# def test_simulation_runs(model_setup):
#     model, _, l, m = model_setup

#     z = torch.zeros(l)
#     u = torch.randn(m)

#     dt = 0.1

#     for _ in range(20):
#         dz = model(z, u)
#         z = z + dt * dz

#     assert torch.isfinite(z).all()


# ------------------------------------------------------------
# 4. BACKPROP THROUGH TIME
# ------------------------------------------------------------
def test_simulation_runs(model_setup):
    model, _, l, m = model_setup

    z0 = torch.zeros(l)
    x0 = torch.zeros(4 * l)

    def u(t):
        return torch.randn(m)

    t_eval = torch.linspace(0, 2, 20)

    S, Y = model.simulate(
        u=u,
        t_eval=t_eval,
        z0=z0,
        x0=x0,
        integrator=rk4_integrate_torch
    )

    assert torch.isfinite(S).all()
    assert torch.isfinite(Y).all()