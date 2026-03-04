# experiments/forward_sanity.py
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from dcm.simulate.design import make_time_grid, boxcar, events, stack_inputs, InputDesign
from dcm.models.neuronal_bilinear import BilinearParameters, BilinearNeuronalModel
from dcm.models.hemodynamic_balloon import HemodynamicParameters, HemodynamicBalloonModel
from dcm.models.forward import ForwardModel, simulate_forward

from experiments.lib.io import load_yaml, save_yaml, make_run_dir, save_npz, save_json


def build_design(cfg: dict) -> InputDesign:
    T = float(cfg["simulation"]["T"])
    dt = float(cfg["simulation"]["dt"])
    t = make_time_grid(T=T, dt=dt)

    names = cfg["inputs"]["names"]
    regs = []
    for name in names:
        spec = cfg["inputs"][name]
        typ = spec["type"]

        if typ == "boxcar":
            regs.append(
                boxcar(
                    t,
                    onsets=spec["onsets"],
                    durations=spec["durations"],
                    amplitudes=spec.get("amplitudes", 1.0),
                )
            )
        
        elif typ == "events":
            # Required: onsets
            # Optional: amplitudes (scalar or list), mode ("nearest" or "floor")
            if "durations" in spec:
                raise ValueError(f"events input '{name}' should not define 'durations'")

            regs.append(
                events(
                    t,
                    onsets=spec["onsets"],
                    amplitudes=spec.get("amplitudes", 1.0),
                    mode=spec.get("mode", "nearest")
                )
            )
        
        else:
            raise ValueError(f"Unknown input type '{typ}' for input '{name}'")

    U = stack_inputs(*regs)  # (T, m)
    return InputDesign(t=t, U=U, names=names)


def build_model(cfg: dict) -> ForwardModel:
    l = int(cfg["model"]["l"])
    m = int(cfg["model"]["m"])

    A = np.asarray(cfg["neuronal"]["A"], dtype=float)
    B = np.asarray(cfg["neuronal"]["B"], dtype=float)
    C = np.asarray(cfg["neuronal"]["C"], dtype=float)

    neuronal_params = BilinearParameters(A=A, B=B, C=C)
    if neuronal_params.l != l or neuronal_params.m != m:
        raise ValueError(
            f"Config model(l={l},m={m}) != neuronal params "
            f"(l={neuronal_params.l},m={neuronal_params.m})"
        )
    neuronal_model = BilinearNeuronalModel(neuronal_params)

    if cfg["hemodynamic"].get("use_defaults", True):
        hemo_params = HemodynamicParameters.with_defaults(l)
    else:
        raise NotImplementedError("Explicit hemodynamic params from YAML not implemented yet")

    hemo_model = HemodynamicBalloonModel(hemo_params)
    return ForwardModel(neuronal_model, hemo_model)


def plot_and_save_separate(
    run_dir: Path,
    t: np.ndarray,
    U: np.ndarray,
    z: np.ndarray,
    s: np.ndarray,
    f: np.ndarray,
    v: np.ndarray,
    q: np.ndarray,
    Y: np.ndarray,
    names: list[str] | None = None,
) -> None:
    """Save separate figures (inputs, neuronal, hemo, bold)."""
    fig_dir = run_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    m = U.shape[1]
    if names is None:
        names = [f"u{j}" for j in range(m)]

    # 1) Inputs
    plt.figure(figsize=(10, 3))
    for j in range(m):
        plt.plot(t, U[:, j], lw=2, label=names[j])
    plt.xlabel("time [s]")
    plt.ylabel("u(t)")
    plt.title("Inputs")
    plt.grid(alpha=0.3)
    plt.legend(ncol=min(m, 4))
    plt.tight_layout()
    plt.savefig(fig_dir / "inputs.png", dpi=200)
    plt.close()

    # 2) Neuronal
    plt.figure(figsize=(10, 3))
    for r in range(z.shape[1]):
        plt.plot(t, z[:, r], lw=2, label=f"z{r}")
    plt.xlabel("time [s]")
    plt.ylabel("z(t)")
    plt.title("Neuronal state")
    plt.grid(alpha=0.3)
    plt.legend(ncol=min(z.shape[1], 4))
    plt.tight_layout()
    plt.savefig(fig_dir / "neuronal.png", dpi=200)
    plt.close()

    # 3) Hemodynamics (region 0)
    r0 = 0
    plt.figure(figsize=(10, 3))
    plt.plot(t, s[:, r0], label="s")
    plt.plot(t, f[:, r0], label="f")
    plt.plot(t, v[:, r0], label="v")
    plt.plot(t, q[:, r0], label="q")
    plt.xlabel("time [s]")
    plt.ylabel("state")
    plt.title("Hemodynamic states (region 0)")
    plt.grid(alpha=0.3)
    plt.legend(ncol=4)
    plt.tight_layout()
    plt.savefig(fig_dir / "hemo_states_region0.png", dpi=200)
    plt.close()

    # 4) BOLD
    plt.figure(figsize=(10, 3))
    for r in range(Y.shape[1]):
        plt.plot(t, Y[:, r], lw=2, label=f"BOLD{r}")
    plt.xlabel("time [s]")
    plt.ylabel("BOLD")
    plt.title("BOLD output")
    plt.grid(alpha=0.3)
    plt.legend(ncol=min(Y.shape[1], 4))
    plt.tight_layout()
    plt.savefig(fig_dir / "bold.png", dpi=200)
    plt.close()


def plot_summary(
    run_dir: Path,
    t: np.ndarray,
    U: np.ndarray,
    z: np.ndarray,
    s: np.ndarray,
    f: np.ndarray,
    v: np.ndarray,
    q: np.ndarray,
    Y: np.ndarray,
    names: list[str] | None = None,
) -> None:
    """Save one stacked summary figure with inputs, neuronal, hemo (r0), bold."""
    fig_dir = run_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    m = U.shape[1]
    if names is None:
        names = [f"u{j}" for j in range(m)]

    fig, axes = plt.subplots(
        4, 1,
        figsize=(12, 10),
        sharex=True,
        gridspec_kw={"height_ratios": [1.2, 1.2, 1.5, 1.2]},
    )

    # Inputs
    for j in range(m):
        axes[0].plot(t, U[:, j], lw=2, label=names[j])
    axes[0].set_ylabel("u(t)")
    axes[0].set_title("DCM forward model sanity check (summary)")
    axes[0].grid(alpha=0.3)
    axes[0].legend(ncol=min(m, 4))

    # Neuronal
    for r in range(z.shape[1]):
        axes[1].plot(t, z[:, r], lw=2, label=f"z{r}")
    axes[1].set_ylabel("z(t)")
    axes[1].grid(alpha=0.3)
    axes[1].legend(ncol=min(z.shape[1], 4))

    # Hemodynamics (region 0)
    r0 = 0
    axes[2].plot(t, s[:, r0], label="s")
    axes[2].plot(t, f[:, r0], label="f")
    axes[2].plot(t, v[:, r0], label="v")
    axes[2].plot(t, q[:, r0], label="q")
    axes[2].set_ylabel("hemo (r0)")
    axes[2].grid(alpha=0.3)
    axes[2].legend(ncol=4)

    # BOLD
    for r in range(Y.shape[1]):
        axes[3].plot(t, Y[:, r], lw=2, label=f"BOLD{r}")
    axes[3].set_ylabel("BOLD")
    axes[3].set_xlabel("time [s]")
    axes[3].grid(alpha=0.3)
    axes[3].legend(ncol=min(Y.shape[1], 4))

    plt.tight_layout()
    plt.savefig(fig_dir / "summary.png", dpi=200)
    plt.close(fig)


def main(config_path: str):
    cfg = load_yaml(config_path)
    experiment_name = cfg.get("name", "experiment")

    run_dir = make_run_dir(experiment_name)

    # Save config copy for reproducibility (what produced this run)
    save_yaml(cfg, run_dir / "config.yaml")

    design = build_design(cfg)
    model = build_model(cfg)

    # ensure design matches neuronal model m
    if design.m != model.neuronal.params.m:
        raise ValueError(f"Design m={design.m} != neuronal m={model.neuronal.params.m}")

    solver = cfg["simulation"]["solver"]
    dt = float(cfg["simulation"]["dt"])
    max_step = solver.get("max_step", dt)   # default dt
    u = design.callable(kind="linear")

    S, Y = simulate_forward(
        model=model,
        u=u,
        t_eval=design.t,
        method=solver.get("method", "RK45"),
        max_step=max_step,
        rtol=float(solver.get("rtol", 1e-6)),
        atol=float(solver.get("atol", 1e-9)),
    )

    # Unpack using YOUR convention:
    # joint state: [z, x], with x=[s_all, f_all, v_all, q_all]
    l = model.l
    z = S[:, :l]  # (T,l)
    x = S[:, l:]  # (T,4l)

    s = x[:, 0 * l : 1 * l]
    f = x[:, 1 * l : 2 * l]
    v = x[:, 2 * l : 3 * l]
    q = x[:, 3 * l : 4 * l]

    # Save arrays
    save_npz(
        run_dir / "traces.npz",
        t=design.t,
        U=design.U,
        S=S,
        Y=Y,
        z=z,
        s=s,
        f=f,
        v=v,
        q=q,
    )

    # Save tiny run summary (easy to read without loading npz)
    summary = {
        "U_shape": list(design.U.shape),
        "S_shape": list(S.shape),
        "Y_shape": list(Y.shape),
        "baseline": {
            "z0": z[0].tolist(),
            "s0": s[0].tolist(),
            "f0": f[0].tolist(),
            "v0": v[0].tolist(),
            "q0": q[0].tolist(),
            "Y0": Y[0].tolist(),
        },
        "final": {
            "zT": z[-1].tolist(),
            "YT": Y[-1].tolist(),
        },
    }
    save_json(summary, run_dir / "summary.json")

    # Plots (separate + summary)
    names = list(design.names) if design.names else None
    plot_and_save_separate(run_dir, design.t, design.U, z, s, f, v, q, Y, names=names)
    plot_summary(run_dir, design.t, design.U, z, s, f, v, q, Y, names=names)

    print("Run saved to:", run_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="experiments/configs/forward_sanity.yaml",
        help="Path to YAML config file",
    )
    args = parser.parse_args()
    main(args.config)