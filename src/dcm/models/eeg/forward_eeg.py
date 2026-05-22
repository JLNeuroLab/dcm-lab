from __future__ import annotations
import torch
import torch.nn as nn
from typing import Callable, Optional

from dcm.models.eeg.neuronal_jansen_rit import JansenRitNeuronal
from dcm.models.eeg.lead_field import LeadFieldParametrization
from dcm.simulate.integrators import rk4_integrate_torch

Tensor = torch.Tensor
InputFn = Callable[[float], Tensor]

class EEGForwardModel(nn.Module):

    def __init__(self,
                 neuronal: JansenRitNeuronal,
                 observation: LeadFieldParametrization
                 ):
        super().__init__()
        if neuronal.l != observation.l:
            raise ValueError("neuronal and observation models must have the same l")
        
        self.neuronal = neuronal
        self.observation = observation

    def initial_state(self) -> Tensor:
        return self.neuronal.initial_state()   # (9*l,)
    
    def dynamics(self, t: float, z: Tensor, u_t: Tensor) -> Tensor:
        return self.neuronal.dynamics(t, z, u_t)
    
    def observe(self, z: Tensor) -> Tensor:
        return self.observation(z)
    
    def simulate(
        self,
        u: InputFn,
        t_eval: Tensor,
        z0: Optional[Tensor] = None,
    ) -> tuple[Tensor, Tensor]:
        """
        Returns
        -------
        S : (T, 9*l)       full state trajectory
        Y : (T, n_sensors) observed EEG
        """
        if z0 is None:
            z0 = self.initial_state().to(t_eval.device)

        def f(t: float, z: Tensor) -> Tensor:
            return self.dynamics(t, z, u(t))

        S = rk4_integrate_torch(f, t_eval, z0)
        Y = torch.stack([self.observe(z) for z in S])

        return S, Y


if __name__ == "__main__":
    from dcm.models.eeg.neuronal_jansen_rit import JansenRitParametersTorch

    l, m = 2, 1
    params  = JansenRitParametersTorch.with_defaults(l=l, m=m)
    neuronal    = JansenRitNeuronal(params)
    observation = LeadFieldParametrization(l=l)
    model       = EEGForwardModel(neuronal, observation)

    t_eval = torch.arange(0.0, 1.0, 0.01)
    u_fn   = lambda t: torch.zeros(m)

    S, Y = model.simulate(u=u_fn, t_eval=t_eval)
    print("S:", S.shape)   # (100, 18)
    print("Y:", Y.shape)   # (100, 2)