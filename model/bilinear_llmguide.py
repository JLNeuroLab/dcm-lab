# dcm/models/neuronal_bilinear.py
# Core neuronal (bilinear) DCM state equation:
#   z_dot = (A + sum_j u_j(t) * B[j]) z + C u(t)
#
# Keep this module "pure": no hemodynamics, no inversion, no plotting.

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Protocol

import numpy as np


Array = np.ndarray


class InputFn(Protocol):
    """u(t) -> (m,) array"""
    def __call__(self, t: float) -> Array: ...


@dataclass(frozen=True)
class BilinearParams:
    """
    l = #regions (state dimension)
    m = #inputs
    """
    A: Array         # (l, l) intrinsic coupling
    B: Array         # (m, l, l) modulatory coupling per input
    C: Array         # (l, m) driving input coupling

    def __post_init__(self) -> None:
        A, B, C = self.A, self.B, self.C
        if A.ndim != 2 or A.shape[0] != A.shape[1]:
            raise ValueError(f"A must be (l,l), got {A.shape}")
        l = A.shape[0]
        if B.ndim != 3 or B.shape[1:] != (l, l):
            raise ValueError(f"B must be (m,l,l) with l={l}, got {B.shape}")
        m = B.shape[0]
        if C.shape != (l, m):
            raise ValueError(f"C must be (l,m) = ({l},{m}), got {C.shape}")

    @property
    def l(self) -> int:
        return int(self.A.shape[0])

    @property
    def m(self) -> int:
        return int(self.B.shape[0])


class BilinearNeuronalModel:
    """
    Neuronal-only state-space model. Later you can wrap it in a coupled model
    with hemodynamics by treating its output z(t) as the input to the hemo states.
    """

    def __init__(self, params: BilinearParams):
        self.params = params

    @property
    def state_dim(self) -> int:
        return self.params.l

    def dynamics(self, t: float, z: Array, u_t: Array) -> Array:
        """
        Compute z_dot at time t for state z and input u_t.
        z: (l,)
        u_t: (m,)
        returns: (l,)
        """
        p = self.params
        if z.shape != (p.l,):
            raise ValueError(f"z must be shape ({p.l},), got {z.shape}")
        if u_t.shape != (p.m,):
            raise ValueError(f"u(t) must be shape ({p.m},), got {u_t.shape}")

        # Effective connectivity: A_eff(t) = A + Σ_j u_j(t) B_j
        # Using tensordot for speed/clarity:
        # u_t: (m,), B: (m,l,l) -> (l,l)
        A_eff = p.A + np.tensordot(u_t, p.B, axes=(0, 0))  # (l, l)

        return A_eff @ z + p.C @ u_t

    def initial_state(self, z0: Optional[Array] = None) -> Array:
        if z0 is None:
            return np.zeros((self.params.l,), dtype=float)
        z0 = np.asarray(z0, dtype=float)
        if z0.shape != (self.params.l,):
            raise ValueError(f"z0 must be shape ({self.params.l},), got {z0.shape}")
        return z0


# -----------------------------------------------------------------------------
# dcm/integrate/solver.py  (you can move this to its own file later)
# -----------------------------------------------------------------------------

def simulate_neuronal(
    model: BilinearNeuronalModel,
    u: InputFn,
    t_eval: Array,
    z0: Optional[Array] = None,
    method: str = "RK45",
    rtol: float = 1e-6,
    atol: float = 1e-9,
) -> Array:
    """
    Integrate neuronal states z(t) over times t_eval using scipy.solve_ivp.

    Returns:
        Z: (len(t_eval), l)
    """
    from scipy.integrate import solve_ivp  # local import keeps core module light

    t_eval = np.asarray(t_eval, dtype=float)
    if t_eval.ndim != 1 or t_eval.size < 2:
        raise ValueError("t_eval must be a 1D array with at least 2 time points")
    if not np.all(np.diff(t_eval) > 0):
        raise ValueError("t_eval must be strictly increasing")

    z0_ = model.initial_state(z0)

    def rhs(t: float, z: Array) -> Array:
        u_t = np.asarray(u(t), dtype=float)
        return model.dynamics(t, z, u_t)

    sol = solve_ivp(
        rhs,
        (float(t_eval[0]), float(t_eval[-1])),
        z0_,
        t_eval=t_eval,
        method=method,
        rtol=rtol,
        atol=atol,
        vectorized=False,
    )
    if not sol.success:
        raise RuntimeError(f"Integration failed: {sol.message}")

    # sol.y is (l, T); return (T, l)
    return sol.y.T


# -----------------------------------------------------------------------------
# Example usage (put in experiments/00_neuronal_forward/run.py)
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    # Toy 3-region, 2-input example
    l, m = 3, 2

    A = np.array([
        [-0.6,  0.2,  0.0],
        [ 0.1, -0.5,  0.3],
        [ 0.0,  0.2, -0.4],
    ], dtype=float)

    # B[0] modulates connection 1->2; B[1] modulates 2->3 (example)
    B = np.zeros((m, l, l), dtype=float)
    B[0, 1, 0] = 0.5
    B[1, 2, 1] = 0.4

    # C drives region 1 with input0, region2 with input1
    C = np.zeros((l, m), dtype=float)
    C[0, 0] = 1.0
    C[1, 1] = 0.8

    params = BilinearParams(A=A, B=B, C=C)
    model = BilinearNeuronalModel(params)

    # Define u(t): two boxcar-ish inputs
    def u(t: float) -> Array:
        u0 = 1.0 if 10.0 <= t <= 30.0 else 0.0
        u1 = 1.0 if 40.0 <= t <= 70.0 else 0.0
        return np.array([u0, u1], dtype=float)

    t = np.linspace(0.0, 100.0, 1001)
    Z = simulate_neuronal(model, u=u, t_eval=t)

    # Minimal sanity print (plot in a notebook or separate script)
    print("Z shape:", Z.shape)      # (T, l)
    print("Z final:", Z[-1])