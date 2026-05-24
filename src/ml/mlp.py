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

    def __init__(self, l: int, m: int):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(l + m, 32),
            nn.Tanh(),
            nn.Linear(32, 32),
            nn.Tanh(),
            nn.Linear(32, 2*l)
        )

    def forward(self, z: Tensor, u: Tensor) -> Tensor:
        return self.net(torch.cat([z, u], dim=-1))