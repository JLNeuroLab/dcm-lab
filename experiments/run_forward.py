# experiments/forward_sanity.py
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from experiments.lib.io import load_yaml, save_yaml, make_run_dir, save_npz, save_json
from experiments.lib.utils import build_model, build_design
from experiments.lib.plotting import plot_and_save_separate, plot_summary
from dcm.models.forward import simulate_forward


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