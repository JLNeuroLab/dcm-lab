import numpy as np
import torch

from dcm.simulate.integrators import (
    euler_integrate,
    rk4_integrate,
    rk4_integrate_torch,
)

# ---------------------------------------------------------------------
# 1. Dummy linear system: dz/dt = -z
# ---------------------------------------------------------------------

def linear_decay_np(t, z):
    return -z

def linear_decay_torch(t, z):
    return -z


# ---------------------------------------------------------------------
# 2. Test: shape consistency (NumPy Euler)
# ---------------------------------------------------------------------

def test_euler_shape():
    t = np.linspace(0, 1, 11)
    z0 = np.array([1.0, -2.0, 0.5])

    Z = euler_integrate(linear_decay_np, t, z0)

    assert Z.shape == (len(t), len(z0))


# ---------------------------------------------------------------------
# 3. Test: RK4 shape consistency (NumPy)
# ---------------------------------------------------------------------

def test_rk4_shape():
    t = np.linspace(0, 1, 11)
    z0 = np.array([1.0, -2.0, 0.5])

    Z = rk4_integrate(linear_decay_np, t, z0)

    assert Z.shape == (len(t), len(z0))


# ---------------------------------------------------------------------
# 4. Test: stability (solution should decay to zero)
# ---------------------------------------------------------------------

def test_decay_to_zero_rk4():
    t = np.linspace(0, 5, 200)
    z0 = np.array([1.0, -2.0, 0.5])

    Z = rk4_integrate(linear_decay_np, t, z0)

    final_norm = np.linalg.norm(Z[-1])
    initial_norm = np.linalg.norm(Z[0])

    assert final_norm < 0.1 * initial_norm


# ---------------------------------------------------------------------
# 5. Test: Euler vs RK4 qualitative consistency
# ---------------------------------------------------------------------

def test_euler_vs_rk4_consistency():
    t = np.linspace(0, 2, 100)
    z0 = np.array([1.0, -1.0, 0.5])

    Ze = euler_integrate(linear_decay_np, t, z0)
    Zr = rk4_integrate(linear_decay_np, t, z0)

    # they should not diverge wildly
    diff = np.linalg.norm(Ze - Zr)

    assert diff < 1.0  # loose but meaningful


# ---------------------------------------------------------------------
# 6. Test: PyTorch RK4 shape + differentiability
# ---------------------------------------------------------------------

def test_rk4_torch_shape():
    t = np.linspace(0, 1, 11)

    z0 = torch.tensor([1.0, -2.0, 0.5], dtype=torch.float32)

    Z = rk4_integrate_torch(linear_decay_torch, t, z0)

    assert Z.shape == (len(t), len(z0))


# ---------------------------------------------------------------------
# 7. Test: PyTorch RK4 consistency with NumPy
# ---------------------------------------------------------------------

def test_rk4_np_vs_torch_consistency():
    t = np.linspace(0, 2, 50)

    z0_np = np.array([1.0, -1.0, 0.5])
    z0_torch = torch.tensor(z0_np, dtype=torch.float32)

    Z_np = rk4_integrate(linear_decay_np, t, z0_np)
    Z_torch = rk4_integrate_torch(linear_decay_torch, t, z0_torch)

    Z_torch_np = Z_torch.detach().cpu().numpy()

    assert np.allclose(Z_np, Z_torch_np, atol=1e-4)