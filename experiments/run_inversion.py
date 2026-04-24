from __future__ import annotations

import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from experiments.lib.io import (
    load_yaml,
    save_yaml,
    make_run_dir,
    save_npz,
    save_json,
)

from experiments.lib.utils import build_design_torch, build_model_torch

from dcm.inference.objectives import DCMInferenceModel
from dcm.inference.optim import map_estimation_torch
from dcm.inference.likelihoods import gaussian_log_likelihood_torch
from dcm.inference.priors import gaussian_log_prior_torch


# ============================================================
# CONFIG ADAPTER
# ============================================================

def extract_model_cfg(cfg, key):
    return {
        "model": cfg["model"],
        "neuronal": cfg[key]["neuronal"],
        "hemodynamic": cfg.get("hemodynamic", {"use_defaults": True}),
    }


# ============================================================
# MAIN
# ============================================================

def main(config_path: str):

    cfg = load_yaml(config_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    run_dir = make_run_dir(cfg.get("name", "inversion"))
    save_yaml(cfg, run_dir / "config.yaml")

    # ============================================================
    # DESIGN
    # ============================================================

    design = build_design_torch(cfg, device=device)
    u_fn = design.callable()

    # ============================================================
    # TRUE MODEL + DATA
    # ============================================================

    model_true = build_model_torch(
        extract_model_cfg(cfg, "true_model"),
        device=device
    )

    with torch.no_grad():
        S_true, Y_true = model_true.simulate(
            u=u_fn,
            t_eval=design.t,
        )

    noise_std = torch.tensor(cfg["noise"]["std"], device=device)
    Y_obs = Y_true + noise_std * torch.randn_like(Y_true)

    # ============================================================
    # INFERENCE MODEL
    # ============================================================

    model_inf = build_model_torch(
        extract_model_cfg(cfg, "init_model"),
        device=device
    )

    A0 = torch.tensor(cfg["init_model"]["neuronal"]["A"], device=device)
    B0 = torch.tensor(cfg["init_model"]["neuronal"]["B"], device=device)
    C0 = torch.tensor(cfg["init_model"]["neuronal"]["C"], device=device)

    theta0 = torch.cat([A0.flatten(), B0.flatten(), C0.flatten()])

    mu_theta = theta0.clone()

    sigma_cfg = cfg["priors"]["sigma"]
    l = model_inf.l
    m = model_inf.neuronal.m

    sigma_prior = torch.cat([
        torch.full((l * l,), sigma_cfg["A"], device=device),
        torch.full((l * l * m,), sigma_cfg["B"], device=device),
        torch.full((l * m,), sigma_cfg["C"], device=device),
    ])

    # ============================================================
    # MAP OBJECTIVE
    # ============================================================

    inference_model = DCMInferenceModel(
        forward_model=model_inf,
        likelihood_fn=gaussian_log_likelihood_torch,
        prior_fn=gaussian_log_prior_torch,
        y_obs=Y_obs,
        sigma=noise_std,
        mu=mu_theta,
        sigma_prior=sigma_prior,
        t_eval=design.t,
        u_fn=u_fn,
        z0=torch.zeros(l, device=device),
        x0=model_inf.hemodynamic.initial_state(),
    )

    # ============================================================
    # OPTIMIZATION
    # ============================================================

    theta = theta0.clone().detach().requires_grad_(True)

    theta_est, trace = map_estimation_torch(
        model=inference_model,
        theta=theta,
        n_steps=cfg["optimizer"]["max_iter"],
        lr=float(cfg["optimizer"].get("lr", 1e-2)),
        method=cfg["optimizer"]["method"].lower(),  # FIX: case-safe
        verbose=True,
    )

    # ============================================================
    # POSTERIOR SIMULATION (FIX CRUCIAL)
    # ============================================================

    # IMPORTANT: inject estimated parameters BEFORE simulating
    with torch.no_grad():

        # overwrite model parameters (simple safe approach)
        offset = 0

        l2 = l * l
        lm = l * m

        A_est = theta_est[offset:offset + l2].reshape(l, l)
        offset += l2

        B_est = theta_est[offset:offset + l2 * m].reshape(m, l, l)
        offset += l2 * m

        C_est = theta_est[offset:offset + lm].reshape(l, m)

        model_inf.neuronal.A[:] = A_est
        model_inf.neuronal.B[:] = B_est
        model_inf.neuronal.C[:] = C_est

        S_est, Y_est = model_inf.simulate(
            u=u_fn,
            t_eval=design.t,
        )

    # ============================================================
    # SAVE
    # ============================================================

    save_npz(
        run_dir / "results.npz",
        t=design.t.cpu().numpy(),
        U=design.U.cpu().numpy(),
        Y_true=Y_true.cpu().numpy(),
        Y_obs=Y_obs.cpu().numpy(),
        Y_est=Y_est.cpu().numpy(),
        theta_est=theta_est.detach().cpu().numpy(),
        trace=np.array(trace),
    )

    save_json(
        {
            "final_loss": float(trace[-1]) if len(trace) > 0 else None,
            "n_iterations": len(trace),
        },
        run_dir / "summary.json",
    )

    # ============================================================
    # PLOT (FIXED + MEANINGFUL)
    # ============================================================

    fig_dir = Path(run_dir) / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    t = design.t.cpu().numpy()

    Y_true_np = Y_true.detach().cpu().numpy()
    Y_obs_np = Y_obs.detach().cpu().numpy()
    Y_est_np = Y_est.detach().cpu().numpy()

    # ---------------- INPUTS ----------------
    plt.figure(figsize=(10, 3))
    for i in range(design.U.shape[1]):
        plt.plot(t, design.U[:, i].cpu().numpy())
    plt.title("Inputs")
    plt.grid()
    plt.tight_layout()
    plt.savefig(fig_dir / "inputs.png", dpi=200)
    plt.close()

    # ---------------- BOLD ----------------
    plt.figure(figsize=(10, 4))
    for r in range(Y_true_np.shape[1]):
        plt.plot(t, Y_true_np[:, r], label=f"true {r}")
        plt.plot(t, Y_obs_np[:, r], "--", label=f"obs {r}")
        plt.plot(t, Y_est_np[:, r], ":", label=f"est {r}")

    plt.title("BOLD: true vs obs vs MAP estimate")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig(fig_dir / "bold.png", dpi=200)
    plt.close()

    # ---------------- TRACE ----------------
    plt.figure(figsize=(8, 3))
    plt.plot(trace)
    plt.title("MAP optimization trace (loss)")
    plt.grid()
    plt.tight_layout()
    plt.savefig(fig_dir / "trace.png", dpi=200)
    plt.close()

    print("✔ Inversion finished:", run_dir)


# ============================================================
# ENTRY
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="experiments/configs/inversion_2r_feedforward.yaml",
    )
    args = parser.parse_args()

    main(args.config)