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


# -----------------------------------------------------------------------------
# Simulation utility (temporary; move later to simulate/)
# -----------------------------------------------------------------------------

def simulate_forward(
    model: ForwardModel,
    u: InputFn,
    t_eval: Array,
    z0: Optional[Array] = None,
    x0: Optional[Array] = None,
    method: str = "RK45",
    rtol: float = 1e-6,
    atol: float = 1e-9,
    max_step: float | None = None,
) -> tuple[Array, Array]:
    """
    Integrate full DCM system.

    Returns:
        S: (T, 5l)
        Y: (T, l)
    """
    from scipy.integrate import solve_ivp

    if max_step is None:
        max_step = (t_eval[-1] - t_eval[0]) / 100

    t_eval = np.asarray(t_eval, dtype=float)
    if t_eval.ndim != 1 or t_eval.size < 2:
        raise ValueError("t_eval must be 1D with at least 2 points")
    if not np.all(np.diff(t_eval) > 0):
        raise ValueError("t_eval must be strictly increasing")

    s0 = model.initial_state(z0, x0)

    def rhs(t: float, state: Array) -> Array:
        u_t = np.asarray(u(t), dtype=float)
        return model.dynamics(t, state, u_t)

    sol = solve_ivp(
        fun=rhs,
        t_span=(float(t_eval[0]), float(t_eval[-1])),
        y0=s0,
        t_eval=t_eval,
        method=method,
        rtol=rtol,
        atol=atol,
        max_step=max_step,
        vectorized=False,
    )

    if not sol.success:
        raise RuntimeError(f"Integration failed: {sol.message}")

    S = sol.y.T  # (T, 5l)
    Y = np.vstack([model.observe(s) for s in S])  # (T, l)

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

    S, Y = simulate_forward(model, u=u, t_eval=t)

    print("State shape:", S.shape)  # (T, 5l)
    print("BOLD shape:", Y.shape)   # (T, l)
    print("Final state:", S[-1])
    print("Final BOLD:", Y[-1])