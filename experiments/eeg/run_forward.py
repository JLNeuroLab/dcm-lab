from __future__ import annotations

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt

from experiments.lib.io import load_yaml, save_yaml, make_run_dir, save_npz, save_json
from experiments.lib.utils import build_design_torch, build_eeg_model_torch


def to_numpy(x):
    return x.detach().cpu().numpy() if torch.is_tensor(x) else np.array(x)


def main(config_path: str):

    cfg = load_yaml(config_path)
    device = torch.device("cpu")

    run_dir = make_run_dir(cfg.get("name", "eeg_forward"))
    save_yaml(cfg, run_dir / "config.yaml")

    # ── design ──────────────────────────────────────────────────
    design = build_design_torch(cfg, device=device)
    u_fn   = design.callable()

    # ── model ───────────────────────────────────────────────────
    model = build_eeg_model_torch(cfg, device=device)

    if design.m != model.neuronal.m:
        raise ValueError(f"Design m={design.m} != neuronal m={model.neuronal.m}")

    # ── simulate ─────────────────────────────────────────────────
    with torch.no_grad():
        S, Y = model.simulate(u=u_fn, t_eval=design.t)

    # ── unpack state  S: (T, 9*l) → (T, 9, l) ───────────────────
    l       = model.l
    T_steps = S.shape[0]
    X = to_numpy(S).reshape(T_steps, 9, l)

    x0 = X[:, 0, :]   # pyramidal net depolarization — EEG source
    x1 = X[:, 1, :]   # spiny stellate potential
    x2 = X[:, 2, :]   # pyramidal excitatory potential
    x3 = X[:, 3, :]   # pyramidal inhibitory potential
    x4 = X[:, 4, :]   # spiny stellate current
    x5 = X[:, 5, :]   # pyramidal excitatory current
    x6 = X[:, 6, :]   # pyramidal inhibitory current
    x7 = X[:, 7, :]   # inhibitory interneuron potential
    x8 = X[:, 8, :]   # inhibitory interneuron current

    t_np = to_numpy(design.t)
    U_np = to_numpy(design.U)
    Y_np = to_numpy(Y)

    # ── save ─────────────────────────────────────────────────────
    save_npz(
        run_dir / "traces.npz",
        t=t_np, U=U_np,
        S=to_numpy(S), Y=Y_np,
        x0=x0, x1=x1, x2=x2, x3=x3,
        x4=x4, x5=x5, x6=x6, x7=x7, x8=x8,
    )

    save_json(
        {"S_shape": list(S.shape), "Y_shape": list(Y.shape)},
        run_dir / "summary.json",
    )

    # ── plots ────────────────────────────────────────────────────
    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)

    input_names = list(design.names) if design.names else [f"u{i}" for i in range(design.m)]
    for i, name in enumerate(input_names):
        axes[0].plot(t_np, U_np[:, i], label=name)
    axes[0].set_ylabel("Input u(t)")
    axes[0].set_title("EEG forward simulation")
    axes[0].legend()

    for r in range(l):
        axes[1].plot(t_np, x0[:, r], label=f"region {r}")
    axes[1].set_ylabel("x0 — pyramidal depol. (mV)")
    axes[1].legend()

    for r in range(Y_np.shape[1]):
        axes[2].plot(t_np, Y_np[:, r], label=f"sensor {r}")
    axes[2].set_ylabel("EEG y(t)")
    axes[2].set_xlabel("Time (s)")
    axes[2].legend()

    plt.tight_layout()
    fig.savefig(run_dir / "figures" / "forward.png", dpi=150)
    plt.close(fig)

    print("Run saved to:", run_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="experiments/configs/eeg/forward_2r_eeg.yaml",
    )
    args = parser.parse_args()
    main(args.config)
