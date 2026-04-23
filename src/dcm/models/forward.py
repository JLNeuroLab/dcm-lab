# -----------------------------------------------------------------------------
# Full DCM (neuronal + hemodynamic) generative model
#
# Joint state vector:
#   state = [ z, s, f, v, q ]  -> shape (5l,)
#
# State equations:
#   z_dot = (A + sum_j u_j(t) B[j]) z + C u(t)
#   x_dot = hemodynamic_balloon(z, x)
#
# Observation equation:
#   y = bold(x)
# -----------------------------------------------------------------------------

from __future__ import annotations
import numpy as np
from typing import Callable, Optional

from dcm.models.neuronal_bilinear import (
    BilinearNeuronalModel,
    BilinearParameters,
)
from dcm.models.hemodynamic_balloon import (
    HemodynamicBalloonModel,
    HemodynamicParameters,
)
from dcm.simulate.integrators import rk4_integrate
from dcm.simulate.adapters import neuronal_rhs_factory

Array = np.ndarray
InputFn = Callable[[float], np.ndarray]


# -----------------------------------------------------------------------------
# Full DCM model
# -----------------------------------------------------------------------------

class ForwardModel:
    """
    Full DCM generative model combining:
        - Bilinear neuronal model
        - Balloon hemodynamic model

    Joint state vector:
        state = [z, x] where:
            z : (l,)
            x : (4l,)
    """

    def __init__(
        self,
        neuronal_model: BilinearNeuronalModel,
        hemodynamic_model: HemodynamicBalloonModel,
    ):
        if neuronal_model.params.l != hemodynamic_model.params.l:
            raise ValueError("Neuronal and hemodynamic models must have same l")

        self.neuronal = neuronal_model
        self.hemodynamic = hemodynamic_model
        self.l = neuronal_model.params.l

    # -------------------------------------------------------------------------
    # State helpers
    # -------------------------------------------------------------------------

    def pack(self, z: Array, x: Array) -> Array:
        """
        Concatenate neuronal and hemodynamic states.

        z : (l,)
        x : (4l,)
        returns: (5l,)
        """
        return np.concatenate([z, x], axis=0)

    def unpack(self, state: Array) -> tuple[Array, Array]:
        """
        Split joint state into (z, x).

        state: (5l,)
        """
        if state.shape != (5 * self.l,):
            raise ValueError(
                f"state must be shape ({5 * self.l},), got {state.shape}"
            )

        z = state[: self.l]
        x = state[self.l :]
        return z, x

    def initial_state(
        self,
        z0: Optional[Array] = None,
        x0: Optional[Array] = None,
    ) -> Array:
        """
        Construct joint initial state.
        """
        z0_ = self.neuronal.initial_state(z0)
        x0_ = self.hemodynamic.initial_state(x0)
        return self.pack(z0_, x0_)

    # -------------------------------------------------------------------------
    # Dynamics
    # -------------------------------------------------------------------------

    def dynamics(
        self,
        t: float,
        state: Array,
        u_t: Array,
    ) -> Array:
        """
        Joint ODE:
            z_dot = neuronal(z, u)
            x_dot = hemodynamic(x, z)

        state: (5l,)
        u_t:   (m,)
        returns: (5l,)
        """
        z, x = self.unpack(state)

        z_dot = self.neuronal.dynamics(t, z, u_t)
        x_dot = self.hemodynamic.dynamics(t, x, z)

        return self.pack(z_dot, x_dot)

    # -------------------------------------------------------------------------
    # Observation
    # -------------------------------------------------------------------------

    def observe(self, state: Array) -> Array:
        """
        BOLD observation model.

        state: (5l,)
        returns: (l,)
        """
        _, x = self.unpack(state)
        return self.hemodynamic.bold(x)


  # -------------------------------------------------------------------------
    # NEW: simulation using shared integrators
    # -------------------------------------------------------------------------

    def simulate(
        self,
        u: InputFn,
        t_eval: Array,
        z0: Optional[Array] = None,
        x0: Optional[Array] = None,
    ) -> tuple[Array, Array]:
        """
        Simulate full DCM system using RK4 integrator.

        Returns
        -------
        S : (T, 5l)
        Y : (T, l)
        """

        s0 = self.initial_state(z0, x0)

        # Build neuronal RHS
        z_rhs = neuronal_rhs_factory(self.neuronal, u)

        # We define full RHS for joint system
        def f(t: float, state: Array) -> Array:
            z, x = self.unpack(state)

            z_dot = z_rhs(t, z)
            x_dot = self.hemodynamic.dynamics(t, x, z)

            return self.pack(z_dot, x_dot)

        S = rk4_integrate(f, t_eval, s0)
        Y = np.vstack([self.observe(s) for s in S])

        return S, Y


# -----------------------------------------------------------------------------
# Minimal sanity demo
# -----------------------------------------------------------------------------

if __name__ == "__main__":

    # 1-region, 1-input example
    l, m = 1, 1

    A = np.array([[-0.5]], dtype=float)
    B = np.zeros((m, l, l), dtype=float)
    C = np.array([[1.0]], dtype=float)

    neuronal_params = BilinearParameters(A=A, B=B, C=C)
    neuronal_model = BilinearNeuronalModel(neuronal_params)

    hemo_params = HemodynamicParameters.with_defaults(l)
    hemo_model = HemodynamicBalloonModel(hemo_params)

    model = ForwardModel(neuronal_model, hemo_model)

    def u(t: float) -> Array:
        return np.array([1.0 if 10.0 <= t <= 20.0 else 0.0], dtype=float)

    t = np.linspace(0.0, 60.0, 601)

    S, Y = model.simulate(u=u, t_eval=t)

    print("State shape:", S.shape)  # (T, 5l)
    print("BOLD shape:", Y.shape)   # (T, l)
    print("Final state:", S[-1])
    print("Final BOLD:", Y[-1])