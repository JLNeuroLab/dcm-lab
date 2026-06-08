import torch
import torch.nn as nn

Tensor = torch.Tensor

class ResidualMLP(nn.Module):
    """Residual MLP for UDE augmentation: maps (z, u) -> correction term of size l."""

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
    """
    MLP replacing the extrinsic coupling terms in the Jansen-Rit UDE.

    Maps (S0, u_t) -> (2*l,) = [forward coupling; backward coupling],
    replacing  C_F @ S0  and  C_B @ S0  in the neuronal dynamics.

    Parameters
    ----------
    l          : number of brain regions
    m          : number of external inputs
    hidden_dim : width of each hidden layer  (default 32)
    u_scale    : divides u before the network to prevent Tanh saturation;
                 set to the maximum expected input amplitude  (e.g. 220.0)
    """

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


class EEGNeuralMLP(nn.Module):
    """
    Neural network modelling the full Jansen-Rit neuronal dynamics.

    Learns  dz/dt = f_θ(z, u_t)  end-to-end, replacing the parametric
    Jansen-Rit equations entirely.  Intended to be used as the neuronal
    component inside EEGNeuralODE (eeg_node.py).

    Parameters
    ----------
    l          : number of brain regions  (state dimension = 9*l)
    m          : number of external inputs
    hidden_dim : width of each hidden layer
    z_scale    : divides z before the network; set to the typical RMS of the
                 state trajectory to keep inputs in the Tanh linear regime
    u_scale    : divides u before the network; set to the maximum input
                 amplitude  (e.g. 220.0 for the standard EEG boxcar)
    """

    def __init__(self, l: int, m: int, hidden_dim: int = 256,
                 z_scale: float = 1.0, u_scale: float = 1.0):
        super().__init__()
        self.l = l
        self.m = m
        self.register_buffer("z_scale", torch.tensor(z_scale, dtype=torch.float32))
        self.register_buffer("u_scale", torch.tensor(u_scale, dtype=torch.float32))

        state_dim = 9 * l
        self.net = nn.Sequential(
            nn.Linear(state_dim + m, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, state_dim)
        )

    def initial_state(self) -> Tensor:
        return torch.zeros(9 * self.l)

    def forward(self, _t: float, z: Tensor, u: Tensor) -> Tensor:
        x = torch.cat([z / self.z_scale, u / self.u_scale], dim=-1)
        return self.net(x)