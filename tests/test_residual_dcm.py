import torch
import pytest

from dcm.torch.neuronal_torch import BilinearNeuronalTorch, BilinearParametersTorch
from ml.mlp import ResidualMLP 
from hybrid.residual_dcm import ResidualDCM


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

    params = BilinearParametersTorch(A=A, B=B, C=C)
    dcm = BilinearNeuronalTorch(params)

    mlp = ResidualMLP(l, m)

    model = ResidualDCM(dcm, mlp, alpha=0.1)

    return model, mlp, l, m


# ------------------------------------------------------------
# 1. FORWARD SHAPE
# ------------------------------------------------------------
def test_forward_shape(model_setup):
    model, _, l, m = model_setup

    z = torch.randn(l)
    u = torch.randn(m)

    dz = model(z, u)

    assert dz.shape == (l,)


# ------------------------------------------------------------
# 2. GRADIENT FLOWS TO MLP
# ------------------------------------------------------------
def test_gradient_flow(model_setup):
    model, mlp, l, m = model_setup

    z = torch.randn(l, requires_grad=True)
    u = torch.randn(m)

    dz = model(z, u)
    loss = dz.pow(2).mean()
    loss.backward()

    grad_norm = sum(
        p.grad.norm().item()
        for p in mlp.parameters()
        if p.grad is not None
    )

    assert grad_norm > 0.0


# ------------------------------------------------------------
# 3. SIMULATION DOES NOT CRASH
# ------------------------------------------------------------
def test_simulation_runs(model_setup):
    model, _, l, m = model_setup

    z = torch.zeros(l)
    u = torch.randn(m)

    dt = 0.1

    for _ in range(20):
        dz = model(z, u)
        z = z + dt * dz

    assert torch.isfinite(z).all()


# ------------------------------------------------------------
# 4. BACKPROP THROUGH TIME
# ------------------------------------------------------------
def test_backprop_through_time(model_setup):
    model, mlp, l, m = model_setup

    z = torch.zeros(l, requires_grad=True)
    u = torch.randn(m)

    dt = 0.1
    traj = []

    for _ in range(20):
        dz = model(z, u)
        z = z + dt * dz
        traj.append(z)

    traj = torch.stack(traj)

    loss = traj.pow(2).mean()
    loss.backward()

    grad_norm = sum(
        p.grad.norm().item()
        for p in mlp.parameters()
        if p.grad is not None
    )

    assert grad_norm > 0.0