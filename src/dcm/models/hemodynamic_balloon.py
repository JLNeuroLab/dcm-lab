# The model describes the transformation from neuronal activity z(t)
# to BOLD signal through four hemodynamic state variables per region:
    # s(t) : vasodilatory signal
    # f(t) : normalized cerebral blood inflow
    # v(t) : normalized venous blood volume
    # q(t) : normalized deoxyhemoglobin content   
# State equations (Friston et al., 2003, Eq. 3):
#     s_dot = z - κ s - γ (f - 1)
#     f_dot = s
#     v_dot = (f - v^(1/α)) / τ
#     q_dot = (f E(f, ρ)/ρ - v^(1/α) q / v) / τ
# 

from __future__ import annotations # This help defining type hints without python evaluating immediately (e.g def f(self) -> B:)
import numpy as np
from dataclasses import dataclass # Python decorator that automatically creates classes to store data 
from typing import Callable, Optional

# Define parameters class for bilinear equation
Array = np.ndarray
InputFn = Callable[[float], np.ndarray]  # e.g. z(t) returning (l,)

@dataclass(frozen=True)
class HemodynamicParameters:
    """
    l = number of brain regions
    Parameters are region-wise arrays of shape (l,).
    Defaults (means) follow the Friston (2003) priors table where applicable:
      kappa (signal decay), gamma (flow-dependent elimination),
      tau (transit time), alpha (Grubb exponent), rho (resting OEF).
    """
    l: int
    kappa: Array  # (l,) rate of signal decay
    gamma: Array  # (l,) rate of flow-dependent elimination
    tau: Array    # (l,) transit time (seconds)
    alpha: Array  # (l,) Grubb exponent
    rho: Array    # (l,) resting oxygen extraction fraction

    V0: float = 0.02 # scalar, resting blood volume fraction

    def __post_init__(self) -> None:
        
        if self.l <= 0:
            raise ValueError("l must be positive")
        
        for name in ["kappa", "gamma", "tau", "alpha", "rho"]:
            arr = np.asarray(getattr(self, name), dtype=float)

            if arr.shape != (self.l, ):
                raise ValueError(f"{name} must be shape ({self.l},), got {arr.shape}")
            if not np.all(np.isfinite(arr)):
                raise ValueError(f"{name} contains non-finite values")
            object.__setattr__(self, name, arr)
        
        if self.V0 <= 0:
            raise ValueError("V0 must be > zero")

    @staticmethod
    def with_defaults(l: int) -> "HemodynamicParameters":
        """
        Convenience constructor using typical Friston (2003) prior means:
          kappa=0.65, gamma=0.41, tau=0.98, alpha=0.32, rho=0.34
        (All in the paper's Table of priors.) Adjust as needed.
        """
        kappa = np.full((l,), 0.65, dtype=float)
        gamma = np.full((l,), 0.41, dtype=float)
        tau   = np.full((l,), 0.98, dtype=float)
        alpha = np.full((l,), 0.32, dtype=float)
        rho   = np.full((l,), 0.34, dtype=float)
        return HemodynamicParameters(l=l, kappa=kappa, gamma=gamma, tau=tau, alpha=alpha, rho=rho)
    
class HemodynamicBalloonModel:
    """
    Hemodynamic forward model (Friston DCM):
      s_dot = z - kappa*s - gamma*(f - 1)
      f_dot = s
      v_dot = (f - v^(1/alpha)) / tau
      q_dot = (f * E(f,rho)/rho - v^(1/alpha) * q / v) / tau

    BOLD output (per region):
      y = V0 * [ k1*(1-q) + k2*(1 - q/v) + k3*(1 - v) ]
      k1 = 7*rho, k2 = 2, k3 = 2*rho + 0.2
    """

    def __init__(self, params: HemodynamicParameters):
        self.params = params

    @staticmethod
    def _E(f: Array, rho: Array) -> Array:
        """
        Oxygen extraction fraction as a function of flow:
          E(f,rho) = 1 - (1 - rho)^(1/f)
        """
        f = np.clip(f, 1e-6, None)  # avoid division by zero / extreme exponents
        return 1.0 - np.power(1.0 - rho, 1.0 / f)
    
    def initial_state(self, x0: Optional[Array] = None) -> Array:
        """
        State vector is flattened as:
          x = [s(0..l-1), f(0..l-1), v(0..l-1), q(0..l-1)]  -> shape (4l,)
        Default baseline is (s=0, f=1, v=1, q=1).
        """
        l = self.params.l
        if x0 is None:
            s0 = np.zeros((l,), dtype=float)
            f0 = np.ones((l,), dtype=float)
            v0 = np.ones((l,), dtype=float)
            q0 = np.ones((l,), dtype=float)
            return np.concatenate([s0, f0, v0, q0], axis=0)
        
        x0 = np.asarray(x0, dtype=float)
        if x0.shape != (4 * l,):
            raise ValueError(f"x0 must be shape ({4 * l},), got shape {x0.shape}")
        return x0
    
    def unpack(self, x: Array) -> tuple[Array, Array, Array, Array]:
        l = self.params.l
        if x.shape != (4 * l,):
            raise ValueError(f"x must be shape ({4*l},), got {x.shape}")
        s = x[0:l]
        f = x[l:2*l]
        v = x[2*l:3*l]
        q = x[3*l:4*l]
        return s, f, v, q

    def pack(self, s: Array, f: Array, v: Array, q: Array) -> Array:
        return np.concatenate([s, f, v, q], axis=0)
    
    def dynamics(self, t: float, x: Array, z_t: Array):
        """
        Equation (3) from Friston et al. (2003)
        Compute x_dot at time t given hemodynamic state x and neuronal drive z_t.

        x: (4l,)
        z_t: (l,) neuronal activity per region (typically the DCM neuronal state z)
        returns: (4l,)
        """
        p = self.params
        l = p.l

        z_t = np.asarray(z_t, dtype=float)
        if z_t.shape != (l,):
            raise ValueError(f"z_t must be shape ({l},), got {z_t.shape}")
        
        s, f, v, q = self.unpack(np.asarray(x, dtype=float))

        # keep physiological states in safe ranges for numerical stability
        f_safe = np.clip(f, 1e-6, None)
        v_safe = np.clip(v, 1e-6, None)

        # outflow term: v^(1/alpha), following Friston's notation this corresponds to f_out as a function of v, 
        # so v_out = f_out(v)
        f_out = np.power(v_safe, 1.0 / p.alpha)

        # E(f, rho)
        E = self._E(f_safe, p.rho)

        s_dot = z_t - p.kappa * s - p.gamma * (f_safe - 1.0)  # vasodilatory signal
        f_dot = s  # change in inflow
        v_dot = (f_safe - f_out) / p.tau   # change in volume caused by inflow
        q_dot = (f_safe * E / p.rho - f_out * q / v_safe) / p.tau  # change in deoxyhemoglobin caused by inflow

        # hemodynamic state vector x = [s, f, v, q]
        return self.pack(s_dot, f_dot, v_dot, q_dot)
    
    # --- observation model ---
    def bold(self, x: Array) -> Array:
        """
        Equation (4) from Friston et al. (2003)
        Compute BOLD y per region from hemodynamic state x.

        returns: (l,)
        """
        p = self.params

        s, f, v, q = self.unpack(np.asarray(x, dtype=float))
        v_safe = np.clip(v, 1e-6, None)

        k1 = 7 * p.rho
        k2 = 2.0
        k3 = 2 * p.rho - 0.2

        y = p.V0 * (k1 * (1 - q) + k2 * (1 - q / v_safe) + k3 * (1 - v))
        return y
    
def simulate_hemodynamic(
    model: HemodynamicBalloonModel,
    z: InputFn,            # neuronal drive function z(t) -> (l,)
    t_eval: Array,
    x0: Optional[Array] = None,
    method: str = "RK45",
    rtol: float = 1e-6,
    atol: float = 1e-9,
    return_bold: bool = True,
) -> tuple[Array, Optional[Array]]:
    """
    Integrate hemodynamic states x(t) over t_eval.

    Returns:
      X: (T, 4l)
      Y: (T, l) if return_bold else None
    """
    from scipy.integrate import solve_ivp

    t_eval = np.asarray(t_eval, dtype=float)
    if t_eval.ndim != 1 or t_eval.size < 2:
        raise ValueError("t_eval must be a 1D array with at least 2 time points")
    if not np.all(np.diff(t_eval) > 0):
        raise ValueError("t_eval must be strictly increasing")

    x0_ = model.initial_state(x0)

    def rhs(t: float, x: Array) -> Array:
        z_t = np.asarray(z(t), dtype=float)
        return model.dynamics(t, x, z_t)

    sol = solve_ivp(
        fun=rhs,
        t_span=(float(t_eval[0]), float(t_eval[-1])),
        t_eval=t_eval,
        y0=x0_,
        method=method,
        rtol=rtol,
        atol=atol,
        vectorized=False,
    )
    if not sol.success:
        raise RuntimeError(f"Integration failed: {sol.message}")

    X = sol.y.T  # (T, 4l)
    if not return_bold:
        return X, None

    Y = np.vstack([model.bold(Xk) for Xk in X])  # (T, l)
    return X, Y

if __name__ == "__main__":
    # Demo: 3-region hemodynamics driven by a toy neuronal signal z(t)
    l = 3
    p = HemodynamicParameters.with_defaults(l)
    model = HemodynamicBalloonModel(p)

    def z(t: float) -> Array:
        # simple "neuronal" boxcars per region
        z1 = 1.0 if 5.0 <= t <= 15.0 else 0.0
        z2 = 0.6 if 10.0 <= t <= 25.0 else 0.0
        z3 = 0.8 if 20.0 <= t <= 35.0 else 0.0
        return np.array([z1, z2, z3], dtype=float)

    t = np.linspace(0.0, 60.0, 601)
    X, Y = simulate_hemodynamic(model, z=z, t_eval=t, return_bold=True)
    print("X shape:", X.shape)  # (T, 4l)
    print("Y shape:", Y.shape)  # (T, l)
    print("Final BOLD:", Y[-1])