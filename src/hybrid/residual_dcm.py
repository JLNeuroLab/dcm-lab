from __future__ import annotations
import torch
import torch.nn as nn
from typing import Callable, Optional

from dcm.torch.neuronal_torch import BilinearNeuronalTorch
from dcm.torch.hemodynamic_torch import HemodynamicBalloonTorch
from dcm.simulate.integrators import rk4_integrate_torch
from ml.mlp import ResidualMLP

Tensor = torch.Tensor
InputFn = Callable[[float], Tensor]

class ResidualDCM(nn.Module):

    def __init__(self, 
                 bilinear: BilinearNeuronalTorch,
                 hemodynamic: HemodynamicBalloonTorch,
                 mlp: ResidualMLP,
                 alpha: float = 0.1
            ):
        super().__init__()
        self.bilinear = bilinear
        self.hemodynamic = hemodynamic
        self.mlp = mlp
        self.integrator = rk4_integrate_torch
        self.alpha = alpha

        self.l = self.bilinear.l
        self.m = self.bilinear.m

    def pack(self, z: Tensor, x: Tensor) -> Tensor:
        return torch.cat([z, x], dim=0)

    def unpack(self, state: Tensor) -> tuple[Tensor, Tensor]:
        if state.shape != (5 * self.l,):
            raise ValueError(
                f"state must be shape ({5*self.l},), got {tuple(state.shape)}"
            )

        z = state[:self.l]
        x = state[self.l:]
        return z, x
    
    def initial_state(
        self,
        z0: Optional[Tensor] = None,
        x0: Optional[Tensor] = None,
    ) -> Tensor:
        z0_ = self.bilinear.initial_state(z0)
        x0_ = self.hemodynamic.initial_state(x0)
        return self.pack(z0_, x0_)

    def dynamics(self, t: float, state: Tensor, u_t: Tensor) -> Tensor:
        z, x = self.unpack(state)

        dz_bilinear = self.bilinear.dynamics(t, z, u_t)
        dz_res = self.alpha * self.mlp(z, u_t)

        dz = dz_bilinear + dz_res

        dx = self.hemodynamic.dynamics(t, x, z)

        return self.pack(dz, dx)
    
    def simulate(
        self,
        u: InputFn,
        t_eval: Tensor,
        z0: Optional[Tensor] = None,
        x0: Optional[Tensor] = None,
    ) -> tuple[Tensor, Tensor]:

        s0 = self.initial_state(z0, x0)

        def f(t: float, state: Tensor):
            u_t = u(t)
            u_t = torch.as_tensor(u_t, device=state.device, dtype=state.dtype)
            return self.dynamics(t, state, u_t)

        S = self.integrator(f, t_eval, s0)
        Y = torch.stack([self.hemodynamic.bold(self.unpack(s)[1]) for s in S])

        return S, Y