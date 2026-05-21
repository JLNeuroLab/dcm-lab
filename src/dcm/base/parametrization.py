from __future__ import annotations
from dataclasses import dataclass
import numpy as np

"""
Neuronal DCM Parameterization
=============================

This module provides utilities for converting the neuronal parameters of a
bilinear Dynamic Causal Model (DCM) between two representations:

1. Structured representation (used by the forward model)
2. Flat vector representation θ (used by optimization and inference)

The neuronal DCM parameters are:

    A : (l, l)
        Baseline effective connectivity between l regions.

    B : (m, l, l)
        Modulatory connectivity matrices. Each experimental input
        can modulate the effective connectivity.

    C : (l, m)
        Direct influence of inputs on neuronal states.

Inference algorithms (optimizers, variational Bayes, MCMC, etc.)
typically operate on a flat parameter vector θ ∈ R^n. However, the
forward DCM simulator expects structured matrices (A, B, C).

This module therefore defines a reversible mapping:

    structured parameters  <->  flat parameter vector θ

The class `NeuronalParameterization` implements:

    pack()   : (A, B, C) -> θ
    unpack() : θ -> (A, B, C)

This separation allows:
- clean model code (working with matrices)
- optimizer-friendly inference (working with vectors)
- easy extension with priors or parameter transforms later.

No priors or parameter constraints are implemented here yet.
"""

Array = np.ndarray

@dataclass(frozen=True)
class NeuronalTheta:
    A: Array          # (l, l)
    B: Array          # (m, l, l)
    C: Array          # (l, m)

class NeuronalParameterization:
    """
    Packs/unpacks neuronal DCM parameters (A,B,C) into a flat vector theta.

    No priors here. No constraints/transforms yet.
    """

    def __init__(self, l: int, m: int):
        
        self.l = l
        self.m = m
        if self.l <= 0 or self.m <= 0:
            raise ValueError("l and m have to be greater than zero")
        
        self.nA = self.l * self.l  # dimensionality of A
        self.nB = self.m * self.l * self.l # dimensionality of B
        self.nC = self.l * self.m # dimensionality of C
        
        i = 0
        self.sl_A = slice(i, i + self.nA); i += self.nA
        self.sl_B = slice(i, i + self.nB); i += self.nB
        self.sl_C = slice(i, i + self.nC); i += self.nC
        
        self.n_params = i

    def unpack(self, theta: Array) -> NeuronalTheta:

        theta = np.asarray(theta, dtype=float)
        if theta.shape != (self.n_params,):
            raise ValueError(f"theta must be shape ({self.n_params},), got {theta.shape}")

        A = theta[self.sl_A].reshape(self.l, self.l)
        B = theta[self.sl_B].reshape(self.m, self.l, self.l)
        C = theta[self.sl_C].reshape(self.l, self.m)

        return NeuronalTheta(A=A, B=B, C=C)
    
    def pack(self, th: NeuronalTheta) -> Array:

        A = np.asarray(th.A, dtype=float)
        B = np.asarray(th.B, dtype=float)
        C = np.asarray(th.C, dtype=float)

        if A.shape != (self.l, self.l):
            raise ValueError(f"A must be ({self.l},{self.l}), got {A.shape}")
        if B.shape != (self.m, self.l, self.l):
            raise ValueError(f"B must be ({self.m},{self.l},{self.l}), got {B.shape}")
        if C.shape != (self.l, self.m):
            raise ValueError(f"C must be ({self.l},{self.m}), got {C.shape}")

        return np.concatenate([A.ravel(), B.ravel(), C.ravel()]).astype(float)
    
    @property
    def names(self) -> list[str]:
        names = []
        for i in range(self.l):
            for j in range(self.l):
                names.append(f"A[{i},{j}]")
        for k in range(self.m):
            for i in range(self.l):
                for j in range(self.l):
                    names.append(f"B[{k},{i},{j}]")
        for i in range(self.l):
            for k in range(self.m):
                names.append(f"C[{i},{k}]")
        return names
    
if __name__ == "__main__":
    l, m = 3, 2
    par = NeuronalParameterization(l, m)

    A = np.random.randn(l, l)
    B = np.random.randn(m, l, l)
    C = np.random.randn(l, m)

    theta = par.pack(NeuronalTheta(A=A, B=B, C=C))
    th2 = par.unpack(theta)
    names = par.names

    assert np.allclose(A, th2.A)
    assert np.allclose(B, th2.B)
    assert np.allclose(C, th2.C)

    print(names)
    print("OK, n_params =", par.n_params)