from __future__ import annotations

import numpy as np
import torch
from typing import Callable

Array = np.ndarray
Tensor = torch.Tensor


# -----------------------------------------------------------------------------
# NEURONAL MODEL ADAPTERS
# -----------------------------------------------------------------------------

def neuronal_rhs_factory(model, input_fn: Callable[[float], Array]) -> Callable:
    """
    Create a NumPy-compatible RHS function for neuronal DCM.

    Parameters
    ----------
    model : BilinearNeuronalModel
        Neuronal model with method:
            dynamics(t, z, u_t) -> dz/dt

    input_fn : Callable[[float], Array]
        Function u(t) returning input vector of shape (m,)

    Returns
    -------
    f : Callable
        RHS function compatible with integrators:
            f(t, z) -> dz/dt, shape (l,)
    """

    m = model.params.m
    l = model.params.l

    def f(t: float, z: Array) -> Array:
        z = np.asarray(z, dtype=float)

        if z.shape != (l,):
            raise ValueError(f"z must be shape ({l},), got {z.shape}")

        u_t = np.asarray(input_fn(t), dtype=float)

        if u_t.shape != (m,):
            raise ValueError(f"u_t must be shape ({m},), got {u_t.shape}")

        return model.dynamics(t, z, u_t)

    return f


def neuronal_rhs_factory_torch(model, input_fn: Callable[[float], Tensor]) -> Callable:
    """
    Create a PyTorch-compatible RHS function for neuronal DCM.

    Fully differentiable and suitable for MAP / Laplace inference.

    Parameters
    ----------
    model : NeuronalBilinearTorch
        Torch neuronal model

    input_fn : Callable[[float], Tensor or array-like]
        Function u(t) returning shape (m,)

    Returns
    -------
    f : Callable
        RHS function:
            f(t, z) -> dz/dt (torch.Tensor)
    """

    m = model.m
    l = model.l

    def f(t: float, z: Tensor) -> Tensor:
        if z.shape != (l,):
            raise ValueError(f"z must be shape ({l},), got {tuple(z.shape)}")

        u_t = input_fn(t)
        u_t = torch.as_tensor(u_t, dtype=z.dtype, device=z.device)

        if u_t.shape != (m,):
            raise ValueError(f"u_t must be shape ({m},), got {tuple(u_t.shape)}")

        return model.dynamics(t, z, u_t)

    return f


# -----------------------------------------------------------------------------
# HEMODYNAMIC MODEL ADAPTERS
# -----------------------------------------------------------------------------

def hemodynamic_rhs_factory(
    model,
    neuronal_state_fn: Callable[[float], Array],
) -> Callable:
    """
    Create a NumPy-compatible RHS function for the hemodynamic model.

    Parameters
    ----------
    model : HemodynamicBalloonModel
        Model with:
            dynamics(t, x, z_t) -> dx/dt

    neuronal_state_fn : Callable[[float], Array]
        Function z(t) returning neuronal state (l,)

    Returns
    -------
    f : Callable
        RHS function:
            f(t, x) -> dx/dt, shape (4l,)
    """

    l = model.params.l
    state_dim = 4 * l

    def f(t: float, x: Array) -> Array:
        x = np.asarray(x, dtype=float)

        if x.shape != (state_dim,):
            raise ValueError(f"x must be shape ({state_dim},), got {x.shape}")

        z_t = np.asarray(neuronal_state_fn(t), dtype=float)

        if z_t.shape != (l,):
            raise ValueError(f"z_t must be shape ({l},), got {z_t.shape}")

        return model.dynamics(t, x, z_t)

    return f


def hemodynamic_rhs_factory_torch(
    model,
    neuronal_state_fn: Callable[[float], Tensor],
) -> Callable:
    """
    Create a PyTorch-compatible RHS function for the hemodynamic model.

    Parameters
    ----------
    model : HemodynamicBalloonModelTorch (or torch-compatible)
        Model with:
            dynamics(t, x, z_t)

    neuronal_state_fn : Callable[[float], Tensor or array-like]
        Function z(t) returning shape (l,)

    Returns
    -------
    f : Callable
        RHS function:
            f(t, x) -> dx/dt (torch.Tensor)
    """

    l = model.l
    state_dim = 4 * l

    def f(t: float, x: Tensor) -> Tensor:
        if x.shape != (state_dim,):
            raise ValueError(f"x must be shape ({state_dim},), got {tuple(x.shape)}")

        z_t = neuronal_state_fn(t)
        z_t = torch.as_tensor(z_t, dtype=x.dtype, device=x.device)

        if z_t.shape != (l,):
            raise ValueError(f"z_t must be shape ({l},), got {tuple(z_t.shape)}")

        return model.dynamics(t, x, z_t)

    return f