import torch
import torch.nn as nn

Tensor = torch.Tensor

class ResidualMLP(nn.Module):

    def __init__(self, l, m):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(l + m, 32),
            nn.Tanh(),
            nn.Linear(32, 32),
            nn.Tanh(),
            nn.Linear(32, l)
        )
    def forward(self, z, u):
        x = torch.cat([z, u], dim=-1)
        return self.mlp(x)
    
class EEGCouplingMLP(nn.Module):

    def __init__(self, l: int, m: int, hidden_dim: int = 32, u_scale: float = 1.0):
        super().__init__()
        self.register_buffer("u_scale", torch.tensor(u_scale, dtype=torch.float32))
        self.net = nn.Sequential(
            nn.Linear(l + m, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 2*l)
        )

    def forward(self, z: Tensor, u: Tensor) -> Tensor:
        return self.net(torch.cat([z, u / self.u_scale], dim=-1))