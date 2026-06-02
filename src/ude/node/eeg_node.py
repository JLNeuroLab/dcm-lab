import torch
import torch.nn as nn
from typing import Callable, Optional

from dcm.models.eeg.lead_field import LeadFieldParametrization
from dcm.simulate.integrators import odeint_euler, odeint_adjoint
from ml.mlp import EEGNeuralMLP


Tensor = torch.Tensor
InputFn = Callable[[float], Tensor]


class EEGNeuralODE(nn.Module):
    """
    Full neural ODE simulation model for EEG DCM.

    Combines an EEGNeuralMLP neuronal model (which learns dz/dt = f_θ(z, u_t))
    with a LeadFieldParametrization observer (which maps pyramidal depolarisation
    x0 to EEG sensors).  Drop-in replacement for EEGCouplingUDE when no
    parametric Jansen-Rit structure is assumed.

    Parameters
    ----------
    neuronal : EEGNeuralMLP
        Neural network modelling the full neuronal state dynamics.
    observer : LeadFieldParametrization
        Linear lead-field mapping from x0 (pyramidal depolarisation) to sensors.
    """

    def __init__(self,
                 neuronal: EEGNeuralMLP,
                 observer: LeadFieldParametrization,
                 adjoint: bool = False,
            ):
        super().__init__()
        self.neuronal = neuronal
        self.observer = observer
        self.adjoint  = adjoint

    def simulate(self, u: InputFn, t_eval: Tensor):
        """
        Integrate the neural ODE and project to EEG sensor space.

        Parameters
        ----------
        u      : callable  t -> (m,)  external input function
        t_eval : (T,) tensor of evaluation time points

        Returns
        -------
        S : (T, 9*l)       full state trajectory
        Y : (T, n_sensors) simulated EEG
        """
        z0 = self.neuronal.initial_state().to(t_eval.device)

        def _f(t, z):
            u_t = torch.as_tensor(u(float(t)), dtype=torch.float32, device=z.device)
            return self.neuronal(t, z, u_t)

        if self.adjoint:
            S = odeint_adjoint(_f, t_eval, z0,
                               adjoint_params=tuple(self.neuronal.parameters()))
        else:
            S = odeint_euler(_f, t_eval, z0)                  # (T, 9*l)
        x0 = S.reshape(-1, 9, self.neuronal.l)[:, 0, :]      # (T, l)
        Y  = (self.observer.K * x0) @ self.observer.L.T      # (T, n_sensors)
        return S, Y
    





    
