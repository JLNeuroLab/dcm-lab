from __future__ import annotations

import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import Optional

Tensor = torch.Tensor


# ---------------------------------------------------------------------
# Parameters container
# ---------------------------------------------------------------------

@dataclass(frozen=True)
class HemodynamicParametersTorch:
    l: int
    kappa: Tensor
    gamma: Tensor
    tau: Tensor
    alpha: Tensor
    rho: Tensor
    V0: float = 0.02

    @staticmethod
    def with_defaults(l: int, device: Optional[torch.device] = None):
        """
        Create a biologically plausible default parameter set.

        This mirrors the NumPy implementation:
        used for tests and quick simulations.
        """

        device = device or torch.device("cpu")

        return HemodynamicParametersTorch(
            l=l,
            kappa=torch.full((l,), 0.65, device=device),
            gamma=torch.full((l,), 0.41, device=device),
            tau=torch.full((l,), 0.98, device=device),
            alpha=torch.full((l,), 0.32, device=device),
            rho=torch.full((l,), 0.34, device=device),
            V0=0.02,
        )


# ---------------------------------------------------------------------
# Hemodynamic model
# ---------------------------------------------------------------------

class HemodynamicBalloonTorch(nn.Module):
    """
    Balloon model (ODE-based, differentiable, nn.Module-compatible)
    """

    def __init__(self, params: HemodynamicParametersTorch):
        super().__init__()

        # store constants
        self.l = params.l
        self.V0 = params.V0

        # fixed biological parameters as buffers
        self.register_buffer("kappa", params.kappa.clone().detach())
        self.register_buffer("gamma", params.gamma.clone().detach())
        self.register_buffer("tau", params.tau.clone().detach())
        self.register_buffer("alpha", params.alpha.clone().detach())
        self.register_buffer("rho", params.rho.clone().detach())

    # -----------------------------------------------------------------
    # Oxygen extraction
    # -----------------------------------------------------------------

    def _E(self, f: Tensor, rho: Tensor) -> Tensor:
        f = torch.clamp(f, 1e-6)
        return 1.0 - torch.pow(1.0 - rho, 1.0 / f)

    # -----------------------------------------------------------------
    # State init
    # -----------------------------------------------------------------

    def initial_state(self, x0: Optional[Tensor] = None) -> Tensor:
        device = self.kappa.device

        if x0 is None:
            return torch.cat([
                torch.zeros(self.l, device=device),
                torch.ones(self.l, device=device),
                torch.ones(self.l, device=device),
                torch.ones(self.l, device=device),
            ], dim=0)

        return x0.to(device=self.kappa.device)

    # -----------------------------------------------------------------
    # unpack / pack
    # -----------------------------------------------------------------

    def unpack(self, x: Tensor):
        l = self.l
        s = x[:l]
        f = x[l:2*l]
        v = x[2*l:3*l]
        q = x[3*l:4*l]
        return s, f, v, q

    def pack(self, s: Tensor, f: Tensor, v: Tensor, q: Tensor) -> Tensor:
        return torch.cat([s, f, v, q], dim=0)

    # -----------------------------------------------------------------
    # dynamics
    # -----------------------------------------------------------------

    def dynamics(self, t: float, x: Tensor, z_t: Tensor) -> Tensor:
        l = self.l
        device = x.device

        s, f, v, q = self.unpack(x)

        f_safe = torch.clamp(f, 1e-6)
        v_safe = torch.clamp(v, 1e-6)

        inv_alpha = 1.0 / self.alpha
        f_out = torch.pow(v_safe, inv_alpha)
        E = self._E(f_safe, self.rho)

        s_dot = z_t - self.kappa * s - self.gamma * (f_safe - 1.0)
        f_dot = s
        v_dot = (f_safe - f_out) / self.tau
        q_dot = (f_safe * E / self.rho - f_out * q / v_safe) / self.tau

        return self.pack(s_dot, f_dot, v_dot, q_dot)

    # -----------------------------------------------------------------
    # BOLD observation
    # -----------------------------------------------------------------

    def bold(self, x: Tensor) -> Tensor:
        s, f, v, q = self.unpack(x)

        v_safe = torch.clamp(v, 1e-6)

        k1 = 7 * self.rho
        k2 = 2.0 * torch.ones_like(q)
        k3 = 2 * self.rho + 0.2

        return self.V0 * (
            k1 * (1 - q)
            + k2 * (1 - q / v_safe)
            + k3 * (1 - v)
        )

if __name__ == "__main__":

    import torch

    from dcm.simulate.adapters import hemodynamic_rhs_factory_torch
    from dcm.simulate.integrators import rk4_integrate_torch

    # ------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------

    l = 3

    kappa = torch.full((l,), 0.65)
    gamma = torch.full((l,), 0.41)
    tau   = torch.full((l,), 0.98)
    alpha = torch.full((l,), 0.32)
    rho   = torch.full((l,), 0.34)

    params = HemodynamicParametersTorch(
        l=l,
        kappa=kappa,
        gamma=gamma,
        tau=tau,
        alpha=alpha,
        rho=rho,
        V0=0.02,
    )

    model = HemodynamicBalloonTorch(params)

    # ------------------------------------------------------------
    # Neuronal drive z(t)
    # ------------------------------------------------------------

    def z_t(t: float) -> torch.Tensor:
        z1 = 1.0 if 5.0 <= t <= 15.0 else 0.0
        z2 = 0.6 if 10.0 <= t <= 25.0 else 0.0
        z3 = 0.8 if 20.0 <= t <= 35.0 else 0.0
        return torch.tensor([z1, z2, z3], dtype=torch.float32)

    # ------------------------------------------------------------
    # Time grid
    # ------------------------------------------------------------

    dt = 0.1
    T = 60.0
    t_eval = torch.arange(0.0, T, dt)

    # ------------------------------------------------------------
    # Initial state
    # ------------------------------------------------------------

    x0 = model.initial_state()

    # ------------------------------------------------------------
    # Integrate
    # ------------------------------------------------------------

    f = hemodynamic_rhs_factory_torch(model, z_t)

    X = rk4_integrate_torch(f, t_eval, x0)

    # ------------------------------------------------------------
    # BOLD
    # ------------------------------------------------------------

    Y = torch.stack([model.bold(x) for x in X])

    # ------------------------------------------------------------
    # Sanity checks
    # ------------------------------------------------------------

    print("X shape:", X.shape)   # (T, 4l)
    print("Y shape:", Y.shape)   # (T, l)

    print("Final state:", X[-1])
    print("Final BOLD:", Y[-1])

    print("Mean BOLD:", Y.mean(dim=0))