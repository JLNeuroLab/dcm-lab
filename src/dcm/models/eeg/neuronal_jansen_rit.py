from __future__ import annotations # This help defining type hints without python evaluating immediately (e.g def f(self) -> B:)
import torch
import torch.nn as nn
from dataclasses import dataclass # Python decorator that automatically creates classes to store data 
from typing import Optional

# Define parameters class for bilinear equation
Tensor = torch.Tensor

@dataclass
class JansenRitParametersTorch:
    """
    Torch-native parameters for Jansen-Rit neuronal DCM.

    l = number of brain regions
    m = number of inputs
    """
    # --- synaptic gains ---
    He: Tensor      # maximum excitatory PSP amplitude
    Hi: Tensor      # maximum inhibitory PSP amplitude

    # --- time constants ---
    tau_e: Tensor   # excitatory synaptic time constant
    tau_i: Tensor   # inhibitory synaptic time constant

    # --- extrinsic connections ---
    C_F: Tensor     # forward connectivity   (l, l)
    C_B: Tensor     # backward connectivity  (l, l)
    C_L: Tensor     # lateral connectivity   (l, l)

    # --- input coupling ---
    P: Tensor       # input-to-region coupling (l, m)

    # --- sigmoid parameters ---
    e0: Tensor      # half maximum firing rate (Hz)
    v0: Tensor      # firing threshold voltage (mV)
    r: Tensor       # sigmoid steepness (mV^-1)

    # --- within-region connections (fixed, from Jansen & Rit 1995) ---
    # C1=135 is the base constant; ratios give C2=108, C3=C4=33.75
    gamma_1: float = 135.0   # pyramidal -> spiny stellate          (C1)
    gamma_2: float = 108.0   # spiny stellate -> pyramidal          (C2 = 0.8*C1)
    gamma_3: float = 33.75   # pyramidal -> inhibitory interneurons (C3 = 0.25*C1)
    gamma_4: float = 33.75   # inhibitory interneurons -> pyramidal (C4 = 0.25*C1)

    @property
    def l(self) -> int:
        return self.C_F.shape[0]

    @property
    def m(self) -> int:
        return self.P.shape[1]

    @staticmethod
    def with_defaults(l: int, m: int) -> JansenRitParametersTorch:
        """
        Initialize with literature values and uninformative
        starting points for estimated parameters.
        Prior expectations from David et al. 2006 Table 1.
        """
        def _off_diag(val):
            M = torch.full((l, l), val)
            M.fill_diagonal_(0.0)
            return M

        return JansenRitParametersTorch(
            He    = torch.tensor(4.0),
            Hi    = torch.tensor(32.0),
            tau_e = torch.tensor(0.008),   # 8 ms in seconds
            tau_i = torch.tensor(0.016),   # 16 ms in seconds
            C_F   = _off_diag(32.0),
            C_B   = _off_diag(16.0),
            C_L   = _off_diag(4.0),
            P     = torch.ones(l, m),
            e0    = torch.tensor(2.5),
            v0    = torch.tensor(6.0),
            r     = torch.tensor(0.56),
            gamma_1 = 135.0,
            gamma_2 = 108.0,
            gamma_3 = 33.75,
            gamma_4 = 33.75,
        )
    
class JansenRitNeuronal(nn.Module):

    def __init__(self, params: JansenRitParametersTorch):

        super().__init__()

        self.l = params.l
        self.m = params.m

        # log-space for positive-only parameters
        self.log_He    = nn.Parameter(torch.log(params.He.clone()))
        self.log_Hi    = nn.Parameter(torch.log(params.Hi.clone()))
        self.log_tau_e = nn.Parameter(torch.log(params.tau_e.clone()))
        self.log_tau_i = nn.Parameter(torch.log(params.tau_i.clone()))
        self.log_e0    = nn.Parameter(torch.log(params.e0.clone()))
        self.log_r     = nn.Parameter(torch.log(params.r.clone()))

        # unconstrained parameters
        self.v0 = nn.Parameter(params.v0.clone())
        self.C_F = nn.Parameter(params.C_F.clone())
        self.C_B = nn.Parameter(params.C_B.clone())
        self.C_L = nn.Parameter(params.C_L.clone())
        self.P   = nn.Parameter(params.P.clone())

        # fixed within-region connectivity constants
        self.gamma_1 = params.gamma_1
        self.gamma_2 = params.gamma_2
        self.gamma_3 = params.gamma_3
        self.gamma_4 = params.gamma_4

    def set_train_mode(self, mode: str):
        """
        mode:
            - "frozen"             → no gradients on any parameter
            - "trainable"          → all gradients enabled
            - "connections_frozen" → freeze C_F, C_B, C_L, P
            - "synaptic_frozen"    → freeze He, Hi, tau_e, tau_i
        """
        if mode == "frozen":
            for p in self.parameters():
                p.requires_grad = False
        elif mode == "trainable":
            for p in self.parameters():
                p.requires_grad = True
        elif mode == "connections_frozen":
            for p in [self.C_F, self.C_B, self.C_L, self.P]:
                p.requires_grad = False
        elif mode == "synaptic_frozen":
            for p in [self.log_He, self.log_Hi, self.log_tau_e, self.log_tau_i]:
                p.requires_grad = False
        else:
            raise ValueError(f"Unknown mode: {mode}")

    def initial_state(self) -> Tensor:
        return torch.zeros(9 * self.l)
    
    # ---- operators ----
    def sigmoid(self, v: Tensor) -> Tensor:
        e0 = torch.exp(self.log_e0)
        r  = torch.exp(self.log_r)
        return 2 * e0 / (1 + torch.exp(r * (self.v0 - v)))
    
    def synaptic_kernel(self, v, w, f, H, tau):
        dv = w
        dw = (H / tau) * f - (2.0 / tau) * w - (1.0 / tau**2) * v
        return dv, dw
    
    # --- state equations ---
    def dynamics(self, t: float, z: Tensor, u_t: Tensor) -> Tensor:
        """
        9 state variables per region (David et al. 2006, Eq. 2):
            x0: net pyramidal depolarization  (= x2 - x3, source of EEG signal)
            x1: spiny stellate membrane potential
            x2: pyramidal excitatory membrane potential
            x3: pyramidal inhibitory membrane potential
            x4: spiny stellate current
            x5: pyramidal excitatory current
            x6: pyramidal inhibitory current
            x7: inhibitory interneuron membrane potential
            x8: inhibitory interneuron current

        z:   (9*l,)
        u_t: (m,)
        """
        He = torch.exp(self.log_He)
        Hi = torch.exp(self.log_Hi)
        tau_e = torch.exp(self.log_tau_e)
        tau_i = torch.exp(self.log_tau_i)

        # unpack: each row is one state variable across all l regions
        X = z.reshape(9, self.l)
        x0, x1, x2, x3 = X[0], X[1], X[2], X[3]
        x4, x5, x6 = X[4], X[5], X[6]
        x7, x8 = X[7], X[8]

        S0 = self.sigmoid(x0)   # pyramidal firing rate         (l,)
        S1 = self.sigmoid(x1)   # spiny stellate firing rate    (l,)
        S7 = self.sigmoid(x7)   # inhibitory interneuron rate   (l,)

        inp = self.P @ u_t  # external drive to spiny stellate (l,)

        # ---- state equations ---

        # spiny stellate: forward + lateral extrinsic + intrinsic gamma1 + input
        dx1, dx4 = self.synaptic_kernel(
            x1,
            x4,
            (self.C_F + self.C_L) @ S0 + self.gamma_1 * S0 + inp,
            He,
            tau_e
        )
        # pyramidal excitatory: backward + lateral extrinsic + intrinsic gamma_2
        dx2, dx5 = self.synaptic_kernel(
            x2,
            x5,
            (self.C_B + self.C_L) @ S0 + self.gamma_2 * S1,
            He,
            tau_e,
        )

        # pyramidal inhibitory: intrinsic gamma_4 from inhibitory interneurons
        dx3, dx6 = self.synaptic_kernel(
            x3, 
            x6,
            self.gamma_4 * S7,
            Hi, 
            tau_i,
        )

        # inhibitory interneurons: backward + lateral extrinsic + intrinsic gamma_3
        dx7, dx8 = self.synaptic_kernel(
            x7, 
            x8,
            (self.C_B + self.C_L) @ S0 + self.gamma_3 * S0,
            He, 
            tau_e,
        )
        # net pyramidal depolarization (x0 = x2 - x3, so dx0 = dx2 - dx3 = x5 - x6)
        dx0 = x5 - x6

        dX = torch.stack([dx0, dx1, dx2, dx3, dx4, dx5, dx6, dx7, dx8])
        return dX.reshape(-1)

    def forward(self, t: float, z: Tensor, u_t: Tensor) -> Tensor:
        return self.dynamics(t, z, u_t)
    

if __name__ == "__main__":
    params = JansenRitParametersTorch.with_defaults(l=2, m=1)
    model  = JansenRitNeuronal(params)

    z0  = model.initial_state()           # (18,)
    u_t = torch.zeros(1)

    dz = model.dynamics(0.0, z0, u_t)    # should be (18,)
    assert dz.shape == z0.shape, f"shape mismatch: {dz.shape}"

    dz.sum().backward()                   # checks gradients flow
    print("OK — shape:", dz.shape)