from __future__ import annotations

import numpy as np
import torch
from typing import Callable

Array = np.ndarray
Tensor = torch.Tensor


# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------

def _check_time_grid(t_eval: Array) -> Array:
    """
    Validate time grid.

    Parameters
    ----------
    t_eval : array-like
        Time points (must be strictly increasing)

    Returns
    -------
    t_eval : np.ndarray
    """
    t_eval = np.asarray(t_eval, dtype=float)

    if t_eval.ndim != 1 or t_eval.size < 2:
        raise ValueError("t_eval must be 1D with at least 2 points")

    if not np.all(np.diff(t_eval) > 0):
        raise ValueError("t_eval must be strictly increasing")

    return t_eval

def euler_integrate(
        f: Callable[[float, Array], Array],
        t_eval: Array,
        z0: Array
    ) -> Array:
    """
    Explicit Euler integrator (NumPy).

    Parameters
    ----------
    f : callable
        RHS function: f(t, z) -> dz/dt

    t_eval : (T,) array
        Time grid

    z0 : (d,) array
        Initial state

    Returns
    -------
    Z : (T, d) array
        Simulated trajectory
    """
    t_eval = _check_time_grid(t_eval=t_eval)

    z = np.asarray(z0, dtype=float)
    if z.ndim != 1:
        raise ValueError("z0 must be 1D (state vector)")

    Z = [z.copy()]

    for i in range(len(t_eval) - 1):
        t = t_eval[i]
        dt = t_eval[i+1] - t

        dz = f(t, z)

        if dz.shape != z.shape:
            raise ValueError(f"f(t,z) returned shape {dz.shape}, expected {z.shape}")
        
        z = z + dt * dz

        Z.append(z.copy())
    
    return np.stack(Z, axis=0)

def rk4_integrate(
    f: Callable[[float, Array], Array],
    t_eval: Array,
    z0: Array,
) -> Array:
    """
    Classical Runge-Kutta 4th order integrator (NumPy).

    Parameters
    ----------
    f : callable
        RHS function: f(t, z) -> dz/dt

    t_eval : (T,) array
        Time grid

    z0 : (d,) array
        Initial state

    Returns
    -------
    Z : (T, d) array
        Simulated trajectory
    """
    t_eval = _check_time_grid(t_eval)

    z = np.asarray(z0, dtype=float)
    if z.ndim != 1:
        raise ValueError("z0 must be 1D (state vector)")
    
    Z = [z.copy()]

    for i in range(len(t_eval) - 1):
        t = t_eval[i]
        dt = t_eval[i + 1] - t

        k1 = f(t, z)
        if k1.shape != z.shape:
            raise ValueError(f"f(t,z) returned shape {k1.shape}, expected {z.shape}")
        
        k2 = f(t + 0.5 * dt, z + 0.5 * dt * k1)
        k3 = f(t + 0.5 * dt, z + 0.5 * dt * k2)
        k4 = f(t + dt, z + dt * k3)

        z = z + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)

        Z.append(z.copy())

    return np.stack(Z, axis=0)

# -----------------------------------------------------------------------------
# RK4 Integrator (PyTorch)
# -----------------------------------------------------------------------------

def rk4_integrate_torch(
    f: Callable[[float, Tensor], Tensor],
    t_eval: Array,
    z0: Tensor,
) -> Tensor:
    """
    Classical Runge-Kutta 4th order integrator (PyTorch).

    Fully differentiable → suitable for MAP / Laplace inference.

    Parameters
    ----------
    f : callable
        RHS function: f(t, z) -> dz/dt (torch.Tensor)

    t_eval : (T,) array-like
        Time grid

    z0 : (d,) torch.Tensor
        Initial state

    Returns
    -------
    Z : (T, d) torch.Tensor
        Simulated trajectory
    """
    t_eval = _check_time_grid(t_eval)

    z = z0
    Z = [z.clone()]

    for i in range(len(t_eval) - 1):
        t = float(t_eval[i])
        dt = float(t_eval[i + 1] - t_eval[i])

        dt_tensor = dt

        k1 = f(t, z)
        k2 = f(t + 0.5 * dt, z + 0.5 * dt_tensor * k1)
        k3 = f(t + 0.5 * dt, z + 0.5 * dt_tensor * k2)
        k4 = f(t + dt, z + dt_tensor * k3)

        z = z + (dt_tensor / 6.0) * (k1 + 2*k2 + 2*k3 + k4)

        Z.append(z.clone())

    return torch.stack(Z, dim=0)