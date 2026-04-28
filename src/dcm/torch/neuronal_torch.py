
# Core neuronal (bilinear) DCM state equation:
#   z_dot = (A + sum_j u_j(t) * B[j]) z + C u(t)
from __future__ import annotations # This help defining type hints without python evaluating immediately (e.g def f(self) -> B:)
import torch
import torch.nn as nn
from dataclasses import dataclass # Python decorator that automatically creates classes to store data 
from typing import Optional

# Define parameters class for bilinear equation
Tensor = torch.Tensor

@dataclass(frozen=True)
class BilinearParametersTorch:
    """
    Torch-native parameters for bilinear neuronal DCM.

    l = number of brain regions
    m = number of inputs
    """
    A: Tensor    # (l, l) intrinsic connectivity between regions
    B: Tensor    # (m, l, l) effective connectivity, modulatory coupling per input
    C: Tensor    # (l, m) input driven connectivity 

    @property
    def l(self) -> int:
        return self.A.shape[0]
    @property
    def m(self) -> int:
        return self.B.shape[0]

class BilinearNeuronalTorch(nn.Module):
    """
    Torch implementation of bilinear neuronal DCM dynamics.

    This version is:
    - fully differentiable
    - MAP-compatible (external optimization)
    - parameter-free in terms of training (Option A design)
    """

    def __init__(self, params: BilinearParametersTorch):
        super().__init__()

        # frozen physics
        self.register_buffer("A", params.A.clone().detach())
        self.register_buffer("B", params.B.clone().detach())
        self.register_buffer("C", params.C.clone().detach())
        
        self.l = params.l
        self.m = params.m


    def dynamics(self, t: float, z: Tensor, u_t: Tensor) -> Tensor:
        """
        z: (l,)
        u: (m,)
        """

        A = self.A.to(z.device)          # (l, l)
        B = self.B.to(z.device)          # (m, l, l)
        C = self.C.to(z.device)          # (l, m)

        # sanity checks
        if z.shape != (self.l,):
            raise ValueError(f"z must be ({self.l},), got {z.shape}")
        if u_t.shape != (self.m,):
            raise ValueError(f"u_t must be ({self.m},), got {u_t.shape}")

        # Effective connectivity: A_eff(t) = A + Σ_j u_j(t) B_j
        # u_t: (m,), B: (m,l,l) -> (l,l)
        dz = A @ z

        for j in range(u_t.shape[0]):
            dz = dz + u_t[j] * (B[j] @ z)

        # input driving term
        dz = dz + C @ u_t

        return dz
    
    def forward(self, z: Tensor, u: Tensor) -> Tensor:
        return self.dynamics(0.0, z, u)

    def initial_state(self, z0: Optional[Tensor] = None) -> Tensor:
        if z0 is None:
            return torch.zeros(self.l)
        return z0

if __name__ == "__main__":

    # ------------------------------------------------------------
    # Toy example: 3 regions, 2 inputs
    # ------------------------------------------------------------
    from dcm.simulate.adapters import neuronal_rhs_factory_torch
    from dcm.simulate.integrators import rk4_integrate_torch
    l, m = 3, 2

    A = torch.tensor([
        [-0.6,  0.2,  0.0],
        [ 0.1, -0.5,  0.3],
        [ 0.0,  0.2, -0.4],
    ], dtype=torch.float32)

    B = torch.zeros((m, l, l), dtype=torch.float32)
    B[0, 1, 0] = 0.5
    B[1, 2, 1] = 0.4

    C = torch.zeros((l, m), dtype=torch.float32)
    C[0, 0] = 1.0
    C[1, 1] = 0.8

    # ------------------------------------------------------------
    # Build model
    # ------------------------------------------------------------

    params = BilinearParametersTorch(A=A, B=B, C=C)
    model = BilinearNeuronalTorch(params)

    # ------------------------------------------------------------
    # Initial state
    # ------------------------------------------------------------

    z0 = torch.zeros(l)
    
    # ------------------------------------------------------------
    # Input function (same idea as NumPy version)
    # ------------------------------------------------------------

    def u_t(t: float) -> torch.Tensor:
        u0 = 1.0 if 10.0 <= t <= 30.0 else 0.0
        u1 = 1.0 if 40.0 <= t <= 70.0 else 0.0
        return torch.tensor([u0, u1], dtype=torch.float32)

    # ------------------------------------------------------------
    # Simple forward simulation (RK4 integration)
    # ------------------------------------------------------------
    dt = 0.1
    T = 100.0
    t_eval = torch.arange(0.0, T, dt)


    f = neuronal_rhs_factory_torch(model, u_t)
    Z = rk4_integrate_torch(f, t_eval, z0)

    # ------------------------------------------------------------
    # Sanity checks
    # ------------------------------------------------------------

    print("Z shape:", Z.shape)      # (T, l)
    print("Final state:", Z[-1])
    print("Mean activity:", Z.mean(dim=0))