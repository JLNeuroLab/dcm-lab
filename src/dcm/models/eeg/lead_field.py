from __future__ import annotations
import torch
import torch.nn as nn
from typing import Optional

Tensor = torch.Tensor

class LeadFieldParametrization(nn.Module):
    """
    EEG observation model (David et al. 2006, Eq. 7).

        y = L @ (K * x0)

    x0 : (l,)          pyramidal net depolarization  (state[0] per region)
    K  : (l,)          per-source amplitude scaling  (learnable nn.Parameter)
    L  : (n_sensors, l) lead field matrix             (fixed register_buffer, default = identity)
    """

    def __init__(self, l: int, n_sensors: Optional[int] = None, L: Optional[Tensor] = None):
        super().__init__()

        self.l = l
        n_sensors = n_sensors if n_sensors is not None else l

        if L is None:
            L = torch.eye(n_sensors, l)
        else:
            if L.shape != (n_sensors, l):
                raise ValueError(f"L must be ({n_sensors}, {l}), got {tuple(L.shape)}")
            
        self.register_buffer("L", L)
        self.K = nn.Parameter(torch.ones(l))
        
    def forward(self, z: Tensor) -> Tensor:
        """z : (9*l,) → y : (n_sensors,)"""

        x0 = z.reshape(9, self.l)[0]
        return self.L @ (self.K * x0)
    
if __name__ == "__main__":
    l = 3
    obs = LeadFieldParametrization(l=l)
    z = torch.randn(9 * l)
    y = obs(z)
    assert y.shape == (l,), f"shape mismatch: {y.shape}"
    y.sum().backward()
    print("OK — output shape:", y.shape)