from __future__ import annotations
import torch
from typing import Callable, Optional

from dcm.torch.neuronal_torch import NeuronalBilinearTorch
from dcm.torch.hemodynamic_torch import HemodynamicBalloonTorch
from dcm.simulate.integrators import rk4_integrate_torch

Tensor = torch.Tensor
InputFn = Callable[[float], Tensor]


class ForwardModelTorch:
    """
    Full DCM generative model (Torch version).

    state = [z, x]
        z: (l,)
        x: (4l,)
        total: (5l,)
    """

    def __init__(
        self,
        neuronal_model: NeuronalBilinearTorch,
        hemodynamic_model: HemodynamicBalloonTorch,
    ):
        if neuronal_model.l != hemodynamic_model.l:
            raise ValueError("Neuronal and hemodynamic models must have same l")

        self.neuronal = neuronal_model
        self.hemodynamic = hemodynamic_model
        self.l = neuronal_model.l

    # ------------------------------------------------------------------
    # State helpers
    # ------------------------------------------------------------------

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
        z0_ = self.neuronal.initial_state(z0)
        x0_ = self.hemodynamic.initial_state(x0)
        return self.pack(z0_, x0_)

    # ------------------------------------------------------------------
    # Dynamics
    # ------------------------------------------------------------------

    def dynamics(self, t: float, state: Tensor, u_t: Tensor) -> Tensor:
        z, x = self.unpack(state)

        z_dot = self.neuronal.dynamics(t, z, u_t)
        x_dot = self.hemodynamic.dynamics(t, x, z)

        return self.pack(z_dot, x_dot)

    # ------------------------------------------------------------------
    # Observation
    # ------------------------------------------------------------------

    def observe(self, state: Tensor) -> Tensor:
        _, x = self.unpack(state)
        return self.hemodynamic.bold(x)

    # ------------------------------------------------------------------
    # Simulation
    # ------------------------------------------------------------------

    def simulate(
        self,
        u: InputFn,
        t_eval: Tensor,
        z0: Optional[Tensor] = None,
        x0: Optional[Tensor] = None,
    ) -> tuple[Tensor, Tensor]:

        s0 = self.initial_state(z0, x0)

        def f(t: float, state: Tensor) -> Tensor:
            z, x = self.unpack(state)

            u_t = u(t)
            u_t = torch.as_tensor(u_t, dtype=state.dtype, device=state.device)

            z_dot = self.neuronal.dynamics(t, z, u_t)
            x_dot = self.hemodynamic.dynamics(t, x, z)

            return self.pack(z_dot, x_dot)

        S = rk4_integrate_torch(f, t_eval, s0)
        Y = torch.stack([self.observe(s) for s in S])

        return S, Y
    
if __name__ == "__main__":

    import torch

    from dcm.torch.neuronal_torch import (
        NeuronalBilinearTorch,
        BilinearParametersTorch,
    )
    from dcm.torch.hemodynamic_torch import (
        HemodynamicBalloonTorch,
        HemodynamicParametersTorch,
    )
    from dcm.simulate.integrators import rk4_integrate_torch
    from dcm.simulate.adapters import neuronal_rhs_factory_torch

    # ------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------

    l, m = 1, 1
    device = torch.device("cpu")

    # ---------------- Neuronal ----------------
    A = torch.tensor([[-0.5]], dtype=torch.float32, device=device)
    B = torch.zeros((m, l, l), dtype=torch.float32, device=device)
    C = torch.tensor([[1.0]], dtype=torch.float32, device=device)

    neuronal_params = BilinearParametersTorch(A=A, B=B, C=C)
    neuronal_model = NeuronalBilinearTorch(neuronal_params)

    # ---------------- Hemodynamic ----------------
    kappa = torch.full((l,), 0.65, device=device)
    gamma = torch.full((l,), 0.41, device=device)
    tau   = torch.full((l,), 0.98, device=device)
    alpha = torch.full((l,), 0.32, device=device)
    rho   = torch.full((l,), 0.34, device=device)

    hemo_params = HemodynamicParametersTorch(
        l=l,
        kappa=kappa,
        gamma=gamma,
        tau=tau,
        alpha=alpha,
        rho=rho,
        V0=0.02,
    )

    hemo_model = HemodynamicBalloonTorch(hemo_params)

    # ------------------------------------------------------------
    # Initial state
    # ------------------------------------------------------------

    z0 = torch.zeros(l, device=device)
    x0 = hemo_model.initial_state()

    state0 = torch.cat([z0, x0])  # (5l,)

    # ------------------------------------------------------------
    # Input function
    # ------------------------------------------------------------

    def u(t: float) -> torch.Tensor:
        val = 1.0 if 10.0 <= t <= 20.0 else 0.0
        return torch.tensor([val], dtype=torch.float32, device=device)

    # ------------------------------------------------------------
    # Time grid
    # ------------------------------------------------------------

    dt = 0.1
    T = 60.0
    t_eval = torch.arange(0.0, T, dt, device=device)

    # ------------------------------------------------------------
    # Build RHS
    # ------------------------------------------------------------

    z_rhs = neuronal_rhs_factory_torch(neuronal_model, u)

    def f(t: float, state: torch.Tensor) -> torch.Tensor:
        z = state[:l]
        x = state[l:]

        z_dot = z_rhs(t, z)
        x_dot = hemo_model.dynamics(t, x, z)

        return torch.cat([z_dot, x_dot], dim=0)

    # ------------------------------------------------------------
    # Integrate
    # ------------------------------------------------------------

    S = rk4_integrate_torch(f, t_eval, state0)

    # ------------------------------------------------------------
    # Observation
    # ------------------------------------------------------------

    Y = torch.stack([
        hemo_model.bold(s[l:])
        for s in S
    ])

    # ------------------------------------------------------------
    # Sanity checks
    # ------------------------------------------------------------

    print("State shape:", S.shape)   # (T, 5l)
    print("BOLD shape:", Y.shape)   # (T, l)
    print("Final state:", S[-1])
    print("Final BOLD:", Y[-1])