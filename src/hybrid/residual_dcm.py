import torch
import torch.nn as nn

from dcm.torch.neuronal_torch import BilinearNeuronalTorch
from ml.mlp import ResidualMLP

class ResidualDCM(nn.Module):

    def __init__(self, 
                 bilinear: BilinearNeuronalTorch, 
                 mlp: ResidualMLP,
                 alpha: float = 0.1
            ):
        super().__init__()
        self.bilinear = bilinear
        self.mlp = mlp
        self.alpha = alpha

    def dynamics(self, t, z, u_t):
        dz_bilinear = self.bilinear.dynamics(t, z, u_t)
        dz_res = self.alpha * self.mlp(z, u_t)

        return dz_bilinear + dz_res
    
    def forward(self, z, u_t):

        return self.dynamics(0.0, z, u_t)

