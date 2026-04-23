from __future__ import annotations

import torch
from dataclasses import dataclass
from typing import Callable, Optional

Tensor = torch.Tensor
InputFn = Callable[[float], Tensor]


# ---------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------

@dataclass(frozen=True)
class HemodynamicParametersTorch:
    """
    l = number of brain regions
    All parameters are torch tensors of shape (l,)
    """

    l: int
    kappa: Tensor  # signal decay
    gamma: Tensor  # flow-dependent elimination
    tau: Tensor    # transit time
    alpha: Tensor  # Grubb exponent
    rho: Tensor    # resting oxygen extraction fraction

    V0: float = 0.02

    def to(self, device: torch.device):
        return HemodynamicParametersTorch(
            l=self.l,
            kappa=self.kappa.to(device),
            gamma=self.gamma.to(device),
            tau=self.tau.to(device),
            alpha=self.alpha.to(device),
            rho=self.rho.to(device),
            V0=self.V0,
        )


# ---------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------

class HemodynamicBalloonTorch:
    """
    Balloon model fully compatible with PyTorch autograd.
    """

    def __init__(self, params: HemodynamicParametersTorch):
        self.params = params

    # -----------------------------
    # Oxygen extraction
    # -----------------------------
    def _E(self, f: Tensor, rho: Tensor) -> Tensor:
        f = torch.clamp(f, 1e-6)
        return 1.0 - torch.pow(1.0 - rho, 1.0 / f)

    # -----------------------------
    # Initial state
    # -----------------------------
    def initial_state(self, x0: Optional[Tensor] = None) -> Tensor:
        l = self.params.l

        if x0 is None:
            s0 = torch.zeros(l)
            f0 = torch.ones(l)
            v0 = torch.ones(l)
            q0 = torch.ones(l)
            return torch.cat([s0, f0, v0, q0], dim=0)

        return x0

    # -----------------------------
    # unpack / pack
    # -----------------------------
    def unpack(self, x: Tensor):
        l = self.params.l
        s = x[:l]
        f = x[l:2*l]
        v = x[2*l:3*l]
        q = x[3*l:4*l]
        return s, f, v, q

    def pack(self, s: Tensor, f: Tensor, v: Tensor, q: Tensor) -> Tensor:
        return torch.cat([s, f, v, q], dim=0)

    # -----------------------------
    # dynamics (DIFFERENTIABLE)
    # -----------------------------
    def dynamics(self, t: float, x: Tensor, z_t: Tensor) -> Tensor:
        """
        x: (4l,)
        z_t: (l,)
        """

        p = self.params
        l = p.l

        s, f, v, q = self.unpack(x)

        f_safe = torch.clamp(f, 1e-6)
        v_safe = torch.clamp(v, 1e-6)

        f_out = torch.pow(v_safe, 1.0 / p.alpha)
        E = self._E(f_safe, p.rho)

        s_dot = z_t - p.kappa * s - p.gamma * (f_safe - 1.0)
        f_dot = s
        v_dot = (f_safe - f_out) / p.tau
        q_dot = (f_safe * E / p.rho - f_out * q / v_safe) / p.tau

        return self.pack(s_dot, f_dot, v_dot, q_dot)

    # -----------------------------
    # BOLD observation
    # -----------------------------
    def bold(self, x: Tensor) -> Tensor:
        p = self.params
        s, f, v, q = self.unpack(x)

        v_safe = torch.clamp(v, 1e-6)

        k1 = 7 * p.rho
        k2 = torch.full_like(k1, 2.0)
        k3 = 2 * p.rho + 0.2

        y = p.V0 * (
            k1 * (1 - q)
            + k2 * (1 - q / v_safe)
            + k3 * (1 - v)
        )

        return y