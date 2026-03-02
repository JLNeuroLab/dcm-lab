# experiments/sim_sanity.py
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

from dcm.simulate.design import make_time_grid, boxcar, stack_inputs, InputDesign
from dcm.models.neuronal_bilinear import BilinearParameters, BilinearNeuronalModel
from dcm.models.hemodynamic_balloon import HemodynamicParameters, HemodynamicBalloonModel
from dcm.models.forward import ForwardModel, simulate_forward


def main():
    # -----------------------
    # 1) Simulation settings
    # -----------------------
    T = 100.0
    dt = 0.1
    t = make_time_grid(T=T, dt=dt)

    # -----------------------
    # 2) Input design (m=2)
    # -----------------------
    u0 = boxcar(t, onsets=[10.0], durations=[20.0], amplitudes=1.0)  # 10-30
    u1 = boxcar(t, onsets=[40.0], durations=[30.0], amplitudes=1.0)  # 40-70
    U = stack_inputs(u0, u1)  # (T, m)
    design = InputDesign(t=t, U=U, names=["u0", "u1"])

    # u(t) callable for solve_ivp
    u = design.callable(kind="zoh")

    # -----------------------
    # 3) Build the model
    # -----------------------
    l = 1
    m = design.m

    A = np.array([[-0.5]], dtype=float)          # (l,l)
    B = np.zeros((m, l, l), dtype=float)         # (m,l,l)
    C = np.array([[1.0, 0.5]], dtype=float)      # (l,m)

    neuronal_params = BilinearParameters(A=A, B=B, C=C)
    neuronal_model = BilinearNeuronalModel(neuronal_params)

    hemo_params = HemodynamicParameters.with_defaults(l)
    hemo_model = HemodynamicBalloonModel(hemo_params)

    model = ForwardModel(neuronal_model, hemo_model)

    # -----------------------
    # 4) Simulate
    # -----------------------
    S, Y = simulate_forward(model, u=u, t_eval=t)

    # -----------------------
    # 5) Unpack
    # -----------------------
    # Joint state: [z, x] where x = [s(0..l-1), f(0..l-1), v(0..l-1), q(0..l-1)]
    z = S[:, :l]     # (T,l)
    x = S[:, l:]     # (T,4l)

    # Now slice the full trajectories consistently:
    s = x[:, 0*l:1*l]  # (T,l)
    f = x[:, 1*l:2*l]  # (T,l)
    v = x[:, 2*l:3*l]  # (T,l)
    q = x[:, 3*l:4*l]  # (T,l)

    bold = Y  # (T,l)

    # -----------------------
    # 6) Quick sanity prints
    # -----------------------
    print("Design: U.shape =", U.shape, "(T,m)")
    print("State:  S.shape =", S.shape, "(T,5l)")
    print("BOLD:   Y.shape =", Y.shape, "(T,l)")

    # Baseline check (at t=0, with default initial hemo state)
    print("\nBaseline (t=0):")
    print("z0:", z[0])
    print("s0:", s[0], "f0:", f[0], "v0:", v[0], "q0:", q[0])
    print("BOLD0:", bold[0])

    print("\nFinal (t=T):")
    print("zT:", z[-1])
    print("BOLDT:", bold[-1])

    # -----------------------
    # 7) Plot
    # -----------------------
    fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True)

    # Inputs
    axes[0].plot(t, U[:, 0], label=design.names[0] if design.names else "u0", lw=2)
    axes[0].plot(t, U[:, 1], label=design.names[1] if design.names else "u1", lw=2)
    axes[0].set_ylabel("u(t)")
    axes[0].set_title("DCM forward model sanity check")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    # Neuronal state
    axes[1].plot(t, z[:, 0], label="z (region 1)", lw=2)
    axes[1].set_ylabel("z(t)")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    # Hemodynamic states (region 1)
    axes[2].plot(t, s[:, 0], label="s", lw=1.5)
    axes[2].plot(t, f[:, 0], label="f", lw=1.5)
    axes[2].plot(t, v[:, 0], label="v", lw=1.5)
    axes[2].plot(t, q[:, 0], label="q", lw=1.5)
    axes[2].set_ylabel("hemo states")
    axes[2].legend(ncol=4)
    axes[2].grid(alpha=0.3)

    # BOLD
    axes[3].plot(t, bold[:, 0], label="BOLD", lw=2)
    axes[3].set_ylabel("y(t)")
    axes[3].set_xlabel("time [s]")
    axes[3].legend()
    axes[3].grid(alpha=0.3)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()