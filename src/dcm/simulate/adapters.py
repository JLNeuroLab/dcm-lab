import numpy as np
import torch
from typing import Callable

def neuronal_rhs_factory(model, input_fn: Callable):
    """
    Wraps a neuronal DCM model into a function compatible with integrators.

    Returns:
        f(t, z) -> dz/dt
    """
     
    def f(t: float, z: np.ndarray) -> np.ndarray:
        u_t = np.asarray(input_fn(t), dtype=float)
        return model.dynamics(t, z, u_t)

    return f

def neuronal_rhs_factory_torch(model, input_fn: Callable):
    """
    Differentiable version for MAP / Laplace inference.

    Returns:
        f(t, z) -> dz/dt (torch)
    """

    def f(t: float, z: torch.Tensor) -> torch.Tensor:
        u_t = input_fn(t)
        if not torch.is_tensor(u_t):
            u_t = torch.tensor(u_t, dtype=z.dtype, device=z.device)

        return model.dynamics(t, z, u_t)

    return f