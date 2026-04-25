import torch
import torch.nn as nn

class Residual_MLP:

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
    
