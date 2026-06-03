import torch
import torch.nn as nn
from typing import Callable, Optional


from dcm.models.eeg.neuronal_jansen_rit import JansenRitNeuronal
from dcm.models.eeg.lead_field import LeadFieldParametrization
from dcm.simulate.integrators import odeint_euler
from ml.mlp import EEGCouplingMLP


Tensor = torch.Tensor
InputFn = Callable[[float], Tensor]

class EEGCouplingUDE(nn.Module):

    def __init__(self,
                 neuronal: JansenRitNeuronal,
                 observer: LeadFieldParametrization,
                 mlp: EEGCouplingMLP,
                 integrator=odeint_euler,
        ):

        super().__init__()

        self.neuronal   = neuronal
        self.observer   = observer
        self.mlp        = mlp
        self.integrator = integrator

        self.l = neuronal.l
        self.m = neuronal.m

    def dynamics(self, t: float, z: Tensor, u_t: Tensor) -> Tensor:

        He    = torch.exp(self.neuronal.log_He)
        Hi    = torch.exp(self.neuronal.log_Hi)
        tau_e = torch.exp(self.neuronal.log_tau_e)
        tau_i = torch.exp(self.neuronal.log_tau_i)

        # unpack: each row is one state variable across all l regions
        X = z.reshape(9, self.l)
        x0, x1, x2, x3 = X[0], X[1], X[2], X[3]
        x4, x5, x6 = X[4], X[5], X[6]
        x7, x8 = X[7], X[8]

        S0 = self.neuronal.sigmoid(x0)   # pyramidal firing rate         (l,)
        S1 = self.neuronal.sigmoid(x1)   # spiny stellate firing rate    (l,)
        S7 = self.neuronal.sigmoid(x7)   # inhibitory interneuron rate   (l,)

        inp = self.neuronal.P @ u_t  # external drive to spiny stellate (l,)

        # MLP: (S0 || u_t) → (2*l,)
        mlp_out    = self.mlp(S0, u_t)
        f_forward  = mlp_out[:self.l]    # replaces (C_F + C_L) @ S0
        f_backward = mlp_out[self.l:]    # replaces (C_B + C_L) @ S0

        # ---- state equations ---

        dx1, dx4 = self.neuronal.synaptic_kernel(
        x1, x4, f_forward  + self.neuronal.gamma_1 * S0 + inp, He, tau_e)

        dx2, dx5 = self.neuronal.synaptic_kernel(
            x2, x5, f_backward + self.neuronal.gamma_2 * S1,        He, tau_e)
        
        dx3, dx6 = self.neuronal.synaptic_kernel(
            x3, x6, self.neuronal.gamma_4 * S7,                      Hi, tau_i)
        
        dx7, dx8 = self.neuronal.synaptic_kernel(
            x7, x8, f_backward + self.neuronal.gamma_3 * S0,         He, tau_e)

        dx0 = x5 - x6

        dX = torch.stack([dx0, dx1, dx2, dx3, dx4, dx5, dx6, dx7, dx8])
        return dX.reshape(-1)
    
    def effective_connectivity(
        self,
        S: torch.Tensor,       # (T, 9*l) full trajectory
        u_fn,                  # input callable
        t_eval: torch.Tensor,  # (T,) time points
        stride: int = 10,      # subsample to avoid recomputing every step
    ) -> tuple[torch.Tensor, torch.Tensor]:
        Js = []
        for i in range(0, len(S), stride):
            X_t  = S[i].reshape(9, self.l)
            S0_t = self.neuronal.sigmoid(X_t[0])
            u_t  = torch.as_tensor(u_fn(float(t_eval[i])),
                                    dtype=torch.float32, device=S.device)
            J = torch.autograd.functional.jacobian(
                    lambda s: self.mlp(s, u_t), S0_t)  # (2l, l)
            Js.append(J)
        J_mean = torch.stack(Js).mean(0)
        return J_mean[:self.l].detach(), J_mean[self.l:].detach()

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
            z0 = self.neuronal.initial_state().to(t_eval.device)

        def f(t: float, z: Tensor) -> Tensor:
            u_t = torch.as_tensor(u(t), device=z.device, dtype=z.dtype)
            return self.dynamics(t, z, u_t)

        S = self.integrator(f, t_eval, z0)
        Y = torch.stack([self.observer(z) for z in S])

        return S, Y