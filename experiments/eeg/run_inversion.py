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

from experiments.lib.utils import build_design_torch, build_eeg_model_torch
from experiments.lib.diagnostics.diagnostics_eeg import save_eeg_diagnostics

from dcm.inference.objectives import EEGInferenceModel
from dcm.inference.optim import map_estimation_torch
from dcm.inference.likelihoods import gaussian_log_likelihood_torch
from dcm.inference.priors import gaussian_log_prior_torch

import os
import time
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

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

def _t(label, t0):
    print(f"  [timing] {label}: {time.perf_counter() - t0:.2f}s")


def extract_model_cfg(cfg, key):
    return {
        "model": cfg["model"],
        "neuronal": cfg[key]["neuronal"],
    }


# ============================================================
# MAIN
# ============================================================

def main(config_path: str):

    t0 = time.perf_counter()

    cfg = load_yaml(config_path)
    #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device("cpu")
    print(f"Device: {device}")

    run_dir = make_run_dir(cfg.get("name", "eeg_inversion"))
    save_yaml(cfg, run_dir / "config.yaml")

    # ── design ──────────────────────────────────────────────────
    design = build_design_torch(cfg, device=device)
    u_fn   = design.callable()
    _t("design", t0); t0 = time.perf_counter()

    # ── true model + data ────────────────────────────────────────
    model_true = build_eeg_model_torch(extract_model_cfg(cfg, "true_model"), device=device)

    with torch.no_grad():
        _, Y_true = model_true.simulate(u=u_fn, t_eval=design.t)

    CF_true = model_true.neuronal.C_F
    CB_true = model_true.neuronal.C_B
    CL_true = model_true.neuronal.C_L
    P_true  = model_true.neuronal.P

    noise_std = torch.tensor(cfg["noise"]["std"], device=device)
    Y_obs = Y_true + noise_std * torch.randn_like(Y_true)
    _t("true model simulate", t0); t0 = time.perf_counter()

    # ── inference model ──────────────────────────────────────────
    model_inf = build_eeg_model_torch(extract_model_cfg(cfg, "init_model"), device=device)

    l = model_inf.l
    m = model_inf.neuronal.m

    init_cfg = cfg["init_model"]["neuronal"]
    CF0 = torch.tensor(init_cfg["C_F"], dtype=torch.float32, device=device).reshape(l, l)
    CB0 = torch.tensor(init_cfg["C_B"], dtype=torch.float32, device=device).reshape(l, l)
    CL0 = torch.tensor(init_cfg["C_L"], dtype=torch.float32, device=device).reshape(l, l)
    P0  = torch.tensor(init_cfg["P"],   dtype=torch.float32, device=device).reshape(l, m)

    theta0    = torch.cat([CF0.flatten(), CB0.flatten(), CL0.flatten(), P0.flatten()])
    mu_theta  = theta0.clone()

    sigma_cfg = cfg["priors"]["sigma"]
    sigma_prior = torch.cat([
        torch.full((l * l,), sigma_cfg["C_F"], device=device),
        torch.full((l * l,), sigma_cfg["C_B"], device=device),
        torch.full((l * l,), sigma_cfg["C_L"], device=device),
        torch.full((l * m,), sigma_cfg["P"],   device=device),
    ])

    # ── MAP objective ────────────────────────────────────────────
    inference_model = EEGInferenceModel(
        forward_model=model_inf,
        likelihood_fn=gaussian_log_likelihood_torch,
        prior_fn=gaussian_log_prior_torch,
        y_obs=Y_obs,
        sigma=noise_std,
        mu=mu_theta,
        sigma_prior=sigma_prior,
        t_eval=design.t,
        u_fn=u_fn,
        z0=torch.zeros(9 * l, device=device),
    )
    _t("inference model setup", t0); t0 = time.perf_counter()

    # ── optimization ─────────────────────────────────────────────
    print(f"Starting MAP ({cfg['optimizer']['method']}, {cfg['optimizer']['max_iter']} steps)...")
    theta = theta0.clone().detach().requires_grad_(True)

    theta_est, trace, _ = map_estimation_torch(
        model=inference_model,
        theta=theta,
        n_steps=cfg["optimizer"]["max_iter"],
        lr=float(cfg["optimizer"].get("lr", 1e-2)),
        method=cfg["optimizer"]["method"].lower(),
        verbose=True,
    )
    _t("MAP estimation", t0); t0 = time.perf_counter()

    # ── inject estimated parameters ──────────────────────────────
    with torch.no_grad():
        CF_est, CB_est, CL_est, P_est = inference_model.unpack_theta(theta_est)

        model_inf.neuronal.C_F.data.copy_(CF_est)
        model_inf.neuronal.C_B.data.copy_(CB_est)
        model_inf.neuronal.C_L.data.copy_(CL_est)
        model_inf.neuronal.P.data.copy_(P_est)

        _, Y_est = model_inf.simulate(u=u_fn, t_eval=design.t)
    _t("final simulation", t0); t0 = time.perf_counter()

    # ── save ─────────────────────────────────────────────────────
    save_npz(
        run_dir / "results.npz",
        t=to_numpy(design.t),
        U=to_numpy(design.U),
        Y_true=to_numpy(Y_true),
        Y_obs=to_numpy(Y_obs),
        Y_est=to_numpy(Y_est),
        trace=np.array(trace),
        theta_est=to_numpy(theta_est),
        CF_true=to_numpy(CF_true), CF_est=to_numpy(CF_est),
        CB_true=to_numpy(CB_true), CB_est=to_numpy(CB_est),
        CL_true=to_numpy(CL_true), CL_est=to_numpy(CL_est),
        P_true=to_numpy(P_true),   P_est=to_numpy(P_est),
    )
    _t("save", t0); t0 = time.perf_counter()

    # ── diagnostics ───────────────────────────────────────────────
    save_eeg_diagnostics(
        run_dir=run_dir,
        t=to_numpy(design.t),
        U=to_numpy(design.U),
        Y_true=to_numpy(Y_true),
        Y_obs=to_numpy(Y_obs),
        Y_est=to_numpy(Y_est),
        trace=np.array(trace),
        CF_true=to_numpy(CF_true), CF_est=to_numpy(CF_est),
        CB_true=to_numpy(CB_true), CB_est=to_numpy(CB_est),
        CL_true=to_numpy(CL_true), CL_est=to_numpy(CL_est),
        P_true=to_numpy(P_true),   P_est=to_numpy(P_est),
    )
    _t("diagnostics", t0)
    print("✔ Inversion finished:", run_dir)


# ============================================================
# ENTRY
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="experiments/configs/eeg/inversion_3r_eeg.yaml",
    )
    args = parser.parse_args()
    main(args.config)