from __future__ import annotations

import argparse
import torch
import numpy as np
from pathlib import Path

from experiments.lib.io import (
    load_yaml,
    save_yaml,
    make_run_dir,
    save_npz,
    save_json,
)

from experiments.lib.utils import build_design_torch, build_model_torch
from experiments.lib.diagnostics.diagnostics_dcm import save_dcm_diagnostics, plot_theta_trajectories

from dcm.inference.objectives import DCMInferenceModel
from dcm.inference.optim import map_estimation_torch
from dcm.inference.likelihoods import gaussian_log_likelihood_torch
from dcm.inference.priors import gaussian_log_prior_torch


# ============================================================
# UTILS SAFE CONVERSION
# ============================================================

def to_numpy(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.array(x)


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

        A_true = model_true.neuronal.A
        B_true = model_true.neuronal.B
        C_true = model_true.neuronal.C

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

    l = model_inf.l
    m = model_inf.neuronal.m

    sigma_cfg = cfg["priors"]["sigma"]

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

    theta_est, trace, theta_trace = map_estimation_torch(
        model=inference_model,
        theta=theta,
        n_steps=cfg["optimizer"]["max_iter"],
        lr=float(cfg["optimizer"].get("lr", 1e-2)),
        method=cfg["optimizer"]["method"].lower(),
        verbose=True,
    )

    theta_est_np = to_numpy(theta_est)

    # ============================================================
    # INJECT ESTIMATED PARAMETERS
    # ============================================================

    with torch.no_grad():

        offset = 0
        l2 = l * l
        lm = l * m

        A_est = theta_est[offset:offset + l2].reshape(l, l)
        A_est_np = to_numpy(A_est)
        eigvals = np.linalg.eigvals(A_est_np)
        stable = bool(np.all(np.real(eigvals) < 0))
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
    # SAVE RESULTS
    # ============================================================

    save_npz(
        run_dir / "results.npz",
        t=design.t.cpu().numpy(),
        U=design.U.cpu().numpy(),
        Y_true=to_numpy(Y_true),
        Y_obs=to_numpy(Y_obs),
        Y_est=to_numpy(Y_est),
        theta_est=theta_est_np,
        trace=np.array(trace),
        A_est=to_numpy(A_est),
        B_est=to_numpy(B_est),
        C_est=to_numpy(C_est),
    )

    # ============================================================
    # DIAGNOSTICS
    # ============================================================

    save_dcm_diagnostics(
        run_dir=run_dir,
        t=to_numpy(design.t),
        U=to_numpy(design.U),
        Y_true=to_numpy(Y_true),
        Y_obs=to_numpy(Y_obs),
        Y_est=to_numpy(Y_est),
        trace=np.array(trace),

        theta_true=np.concatenate([
            to_numpy(A_true).flatten(),
            to_numpy(B_true).flatten(),
            to_numpy(C_true).flatten()
        ]),
        theta_est=theta_est_np,

        A_true=to_numpy(A_true),
        A_est=to_numpy(A_est),
        B_true=to_numpy(B_true),
        B_est=to_numpy(B_est),
        C_true=to_numpy(C_true),
        C_est=to_numpy(C_est),
    )

    plot_theta_trajectories(
        theta_trace=theta_trace,
        run_dir=run_dir,
        l=model_inf.l,
        m=model_inf.neuronal.m,
    )
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