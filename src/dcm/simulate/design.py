from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Sequence, Union

import numpy as np

Array = np.ndarray
ArrayLike = Union[Array, Sequence[float]]


# -----------------------------
# 1) TIME GRID
# -----------------------------

def make_time_grid(T: float, dt: float, t0: float = 0.0) -> Array:
    """
    Create a uniform time grid used for:
      - defining inputs u(t) on a discrete grid
      - sampling the ODE solution (t_eval)

    Returns
    -------
    t : (n_time,) array
        t = [t0, t0+dt, ..., t0+T]
    """
    if T <= 0:
        raise ValueError("T must be > 0")
    if dt <= 0:
        raise ValueError("dt must be > 0")

    n = int(np.round(T / dt))
    t = t0 + dt * np.arange(n + 1)
    return t


# -----------------------------
# 2) BUILD 1D INPUT REGRESSORS
# -----------------------------

def _as_1d(x: Union[float, ArrayLike], n: int) -> Array:
    """Broadcast scalar or check length for event-wise amplitudes."""
    if np.isscalar(x):
        return np.full(n, float(x))
    x = np.asarray(x, dtype=float).ravel()
    if x.size != n:
        raise ValueError(f"Expected length {n}, got {x.size}")
    return x


def boxcar(
    t: Array,
    onsets: ArrayLike,
    durations: ArrayLike,
    amplitudes: Union[float, ArrayLike] = 1.0,
) -> Array:
    """
    Create a boxcar regressor on grid t.

    u[k] = sum_i amp[i] * 1{ onset[i] <= t[k] < onset[i] + duration[i] }
    """
    t = np.asarray(t, dtype=float).ravel()
    onsets = np.asarray(onsets, dtype=float).ravel()
    durations = np.asarray(durations, dtype=float).ravel()

    if onsets.size != durations.size:
        raise ValueError("onsets and durations must have same length")
    if np.any(durations < 0):
        raise ValueError("durations must be >= 0")

    amps = _as_1d(amplitudes, onsets.size)

    u = np.zeros_like(t, dtype=float)
    for onset, dur, amp in zip(onsets, durations, amps):
        start = onset
        stop = onset + dur
        mask = (t >= start) & (t < stop)
        u[mask] += amp
    return u


def events(
    t: Array,
    onsets: ArrayLike,
    amplitudes: Union[float, ArrayLike] = 1.0,
    mode: str = "nearest",
) -> Array:
    """
    Create a stick-function event regressor on grid t.

    For each event onset, choose an index on the grid and add amplitude there.
    """
    t = np.asarray(t, dtype=float).ravel()
    onsets = np.asarray(onsets, dtype=float).ravel()
    amps = _as_1d(amplitudes, onsets.size)

    if np.any(np.diff(t) <= 0):
        raise ValueError("t must be strictly increasing")

    u = np.zeros_like(t, dtype=float)

    for onset, amp in zip(onsets, amps):
        if mode == "nearest":
            idx = int(np.argmin(np.abs(t - onset)))
        elif mode == "floor":
            idx = int(np.searchsorted(t, onset, side="right") - 1)
            idx = np.clip(idx, 0, t.size - 1)
        else:
            raise ValueError("mode must be 'nearest' or 'floor'")
        u[idx] += amp

    return u

def event_pulses(
    t: np.ndarray,
    onsets,
    duration: float,
    amplitudes: float | np.ndarray = 1.0,
) -> np.ndarray:
    """
    Represent events as short boxcar pulses of fixed duration.
    """
    onsets = np.asarray(onsets, dtype=float).ravel()
    durations = np.full_like(onsets, float(duration))
    return boxcar(t, onsets=onsets, durations=durations, amplitudes=amplitudes)


# -----------------------------
# 3) STACK INTO U(t) MATRIX
# -----------------------------

def stack_inputs(*u_list: Array) -> Array:
    """
    Stack multiple 1D regressors into U of shape (T, m).
    """
    if len(u_list) == 0:
        raise ValueError("Provide at least one regressor")

    u_list = [np.asarray(u, dtype=float).ravel() for u in u_list]
    T = u_list[0].size
    if any(u.size != T for u in u_list):
        raise ValueError("All regressors must have same length")

    U = np.column_stack(u_list)  # (T, m)
    return U


# -----------------------------
# 4) MAKE u(t) CALLABLE FOR ODE
# -----------------------------

def u_callable(
    t: Array,
    U: Array,
    kind: str = "zoh",
    fill_value: float = 0.0,
) -> Callable[[float], Array]:
    """
    Create u(t_scalar) -> (m,) for any float t_scalar.

    kind="zoh" is recommended for DCM inputs.
    """
    t = np.asarray(t, dtype=float).ravel()
    U = np.asarray(U, dtype=float)
    if U.ndim == 1:
        U = U[:, None]
    if U.shape[0] != t.size:
        raise ValueError("U must have shape (len(t), m)")
    if np.any(np.diff(t) <= 0):
        raise ValueError("t must be strictly increasing")

    m = U.shape[1]

    if kind == "zoh":
        def f(ts: float) -> Array:
            if ts < t[0] or ts > t[-1]:
                return np.full((m,), fill_value, dtype=float)
            idx = int(np.searchsorted(t, ts, side="right") - 1)
            idx = np.clip(idx, 0, t.size - 1)
            return U[idx].astype(float, copy=True)

    elif kind == "linear":
        def f(ts: float) -> Array:
            if ts < t[0] or ts > t[-1]:
                return np.full((m,), fill_value, dtype=float)
            out = np.empty((m,), dtype=float)
            for j in range(m):
                out[j] = np.interp(ts, t, U[:, j])
            return out
    else:
        raise ValueError("kind must be 'zoh' or 'linear'")

    return f


# -----------------------------
# 5) OPTIONAL CONTAINER
# -----------------------------

@dataclass(frozen=True)
class InputDesign:
    t: Array              # (T,)
    U: Array              # (T, m)
    names: Optional[Sequence[str]] = None

    def callable(self, kind: str = "zoh", fill_value: float = 0.0) -> Callable[[float], Array]:
        return u_callable(self.t, self.U, kind=kind, fill_value=fill_value)

    @property
    def m(self) -> int:
        return int(self.U.shape[1])
    
    @property
    def time(self) -> np.ndarray:
        return self.t

if __name__ == "__main__":
    import matplotlib.pyplot as plt

    # --- Build a simple 2-input design ---
    t = make_time_grid(T=100.0, dt=0.1)

    # Two boxcar inputs (condition 0 and condition 1)
    u0 = boxcar(t, onsets=[10.0], durations=[20.0])  # ON from 10 to 30
    u1 = boxcar(t, onsets=[40.0], durations=[30.0])  # ON from 40 to 70

    U = stack_inputs(u0, u1)  # shape (n_time, 2)
    inputs = InputDesign(t, U, names=["u0", "u1"])

    # --- Build u(t) callable for solve_ivp ---
    u = inputs.callable(kind="zoh")  # u(t_scalar) -> (m,) vector, piecewise constant

    # --- Sanity checks ---
    print("t shape:", inputs.t.shape)          # (n_time,)
    print("U shape:", inputs.U.shape)          # (n_time, m)
    print("m (n_inputs):", inputs.m)

    # Check that u(t) returns the correct shape at arbitrary times
    test_times = [0.0, 9.95, 10.0, 10.05, 29.99, 30.0, 39.99, 40.0, 69.99, 70.0, 100.0]
    print("\nSample u(t) values (ZOH):")
    for ts in test_times:
        ut = u(ts)
        print(f"t={ts:6.2f} -> u(t)={ut}  shape={ut.shape}")

    # Check that the discrete U matches expectation at grid indices
    # (these checks depend on dt=0.1 and the [start, stop) convention)
    assert np.allclose(U[t == 0.0][0], [0.0, 0.0])
    assert np.allclose(U[t == 10.0][0], [1.0, 0.0])
    assert np.allclose(U[t == 30.0][0], [0.0, 0.0])   # off at exactly 30 because stop is exclusive
    assert np.allclose(U[t == 40.0][0], [0.0, 1.0])
    assert np.allclose(U[t == 70.0][0], [0.0, 0.0])   # off at exactly 70
    print("\nDiscrete grid assertions passed.")

    # --- Plot the inputs ---
    fig, ax = plt.subplots(figsize=(10, 3))
    ax.plot(t, U[:, 0], label="u0", lw=2)
    ax.plot(t, U[:, 1], label="u1", lw=2)
    ax.set_xlabel("time [s]")
    ax.set_ylabel("u(t)")
    ax.set_title("Input design sanity check")
    ax.set_ylim(-0.1, 1.1)
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()