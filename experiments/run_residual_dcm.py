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
from experiments.lib.diagnostics import save_dcm_diagnostics, plot_theta_trajectories

from dcm.inference.objectives import DCMInferenceModel
from dcm.inference.optim import map_estimation_torch
from dcm.inference.likelihoods import gaussian_log_likelihood_torch
from dcm.inference.priors import gaussian_log_prior_torch

from hybrid.residual_dcm import ResidualDCM
from ml.mlp import ResidualMLP

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
    design = build_design_torch(
        cfg=cfg, 
        device=device
    )
    u_fn = design.callable()

    # ============================================================
    # TRUE MODEL + DATA
    # ============================================================

    model_true = build_model_torch(
        cfg=extract_model_cfg(cfg=cfg, key="true_model"),
        device=device
    )
    with torch.no_grad():
        S_true, Y_true = model_true.simulate(
            u=u_fn,
            t_eval=design.t 
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
        cfg=extract_model_cfg(cfg=cfg, key="init_model"),
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
        sigma=sigma_cfg,
        mu=mu_theta,
        sigma_prior=sigma_prior,
        t_eval=design.t,
        u_fn=u_fn,
        z0=torch.zeros(l, device=device),
        x0=model_inf.hemodynamic.initial_state()
    )

    # ============================================================
    # DCM parameter estimation
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

    # ============================================================
    # Hybrid model
    # ============================================================

    bilinear_model = inference_model.forward_model.neuronal
    hemodynamic_model = inference_model.forward_model.hemodynamic
    mlp = ResidualMLP(l, m)

    hybrid_model = ResidualDCM(
        bilinear=bilinear_model,
        hemodynamic=hemodynamic_model
        mlp=mlp,
        alpha=cfg["hybrid"]["alpha"]
    )

    # ============================================================
    # Hybrid model optmization loop, here the mlp is trained
    # ============================================================

    optimizer = torch.optim.Adam(mlp.parameters(), lr=cfg["hybrid"]["lr"])
    
    for epoch in range(cfg["hybrid"]["epochs"]):

        S, Y_pred = hybrid_model.simulate(
            u=u_fn,
            t_eval=design.t,
            z0=torch.zeros(l, device=device),
            x0=hemo.initial_state(),
        )

        loss = ((Y_true - Y_pred)**2).mean()

        optimizer.zero_grad
        loss.backward()
        optimizer.step()

        if epoch % 10 == 0:
            print(f"[{epoch}] loss={loss.item():.6f}")
