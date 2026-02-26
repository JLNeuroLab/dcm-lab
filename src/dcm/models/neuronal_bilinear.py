
# Core neuronal (bilinear) DCM state equation:
#   z_dot = (A + sum_j u_j(t) * B[j]) z + C u(t)
from __future__ import annotations # This help defining type hints without python evaluating immediately (e.g def f(self) -> B:)
import numpy as np
from dataclasses import dataclass # Python decorator that automatically creates classes to store data 
from typing import Callable, Optional, Protocol

# Define parameters class for bilinear equation
Array = np.ndarray

@dataclass(frozen=True)
class BilinearParameters:
    """
    l = number of brain regions
    m = number of inputs
    """
    A: Array    # (l, l) intrinsic connectivity between regions
    B: Array    # (m, l, l) effective connectivity, modulatory coupling per input
    C: Array    # (l, m) input driven connectivity 

    def __post_init__(self) -> None:
        A, B, C = self.A, self.B, self.C

        if A.ndim != 2 or A.shape[0] != A.shape[1]:
            raise ValueError(f"A must be (l,l), got {A.shape}")
        l = A.shape[0]

        if B.ndim != 3 or B.shape[1:] != (l, l):
            raise ValueError(f"B must be (m, l, l) with l = {l}, got {B.shape}")
        m = B.shape[0]

        if C.shape != (l, m):
            raise ValueError(f"C must be (l, m) = ({l}, {m}), got {C.shape}")

    @property
    def l(self) -> int:
        return self.A.shape[0]
    @property
    def m(self) -> int:
        return self.B.shape[0]

class BilinearNeuronalModel:
    """
    Neuronal state-space model (generative model according to Friston et al. (2003))
    """

    def __init__(self, params: BilinearParameters):
        self.params = params

    def dynamics(self, t: float, z: Array, u_t: Array) -> Array:
        """
        Compute z_dot at time t given state z and input u_t
        z: (l,)
        u_t: (m,)
        returns: (l,)
        """
        p = self.params

        if z.shape != (p.l, ):
            raise ValueError(f"z must be shape ({p.l},), got {z.shape}")
        if u_t.shape != (p.m, ):
            raise ValueError(f"u_t must be shape ({p.m},), got {u_t.shape}")

        # Effective connectivity: A_eff(t) = A + Σ_j u_j(t) B_j
        # Using tensordot for speed/clarity:
        # u_t: (m,), B: (m,l,l) -> (l,l)
        A_eff = p.A + np.tensordot(u_t, p.B, axes=(0, 0))

        # z_dot = (A + Σ_j u_j(t) B_j) z + C u
        return A_eff @ z + p.C @ u_t
    
    def initial_state(self, z0: Optional[Array] = None) -> Array:
        if z0 is None:
            return np.zeros((self.params.l,), dtype=float)
        z0 = np.asarray(z0, dtype=float)
        if z0.shape != (self.params.l,):
            raise ValueError(f"z0 must be shape ({self.params.l},), got {z0.shape}")
        return z0

# -----------------------------------------------------------------------------
# dcm/integrate/solver.py  (move this to its own file later)
# -----------------------------------------------------------------------------

InputFn = Callable[[float], np.ndarray]

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
    from scipy.integrate import solve_ivp

    t_eval = np.asarray(t_eval, dtype=float)
    if t_eval.ndim != 1 or t_eval.size < 2:
        raise ValueError("t_eval must be a 1D array with at least 2 time points")
    if not np.all(np.diff(t_eval) > 0):
        raise ValueError("t_eval must be strictly increasing")
    
    z0_ = model.initial_state(z0)

    def rhs(t: float, z: Array) -> Array:

        u_t = np.asarray(u(t), dtype=float)
        dynamics = model.dynamics(t, z, u_t)
        return dynamics
    
    sol = solve_ivp(
        fun=rhs,
        t_span=(float(t_eval[0]), float(t_eval[-1])),
        t_eval=t_eval,
        y0=z0_,
        method=method,
        rtol=rtol,
        atol=atol,
        vectorized=False
    )

    if not sol.success:
        raise RuntimeError(f"Integration failed: {sol.message}")
    # sol.y is (l, T); return (T, l)
    return sol.y.T

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

    params = BilinearParameters(A=A, B=B, C=C)
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