from __future__ import annotations

import torch
import numpy as np
import argparse

from experiments.lib.io import (
    load_yaml,
    save_yaml,
    make_run_dir,
    save_npz,
)

from experiments.lib.utils import build_design_torch, build_model_torch
from experiments.lib.diagnostics.diagnostics_residual_dcm import save_hybrid_diagnostics

from dcm.inference.objectives import DCMInferenceModel
from dcm.inference.optim import map_estimation_torch
from dcm.inference.likelihoods import gaussian_log_likelihood_torch
from dcm.inference.priors import gaussian_log_prior_torch

from hybrid.residual_dcm import ResidualDCM
from ml.mlp import ResidualMLP


# ============================================================
# UTILS
# ============================================================

def to_numpy(x):
    return x.detach().cpu().numpy() if torch.is_tensor(x) else np.array(x)


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

    design = build_design_torch(cfg=cfg, device=device)
    u_fn = design.callable()
    t_eval = design.t

    # ============================================================
    # TRUE MODEL
    # ============================================================

    model_true = build_model_torch(
        cfg=extract_model_cfg(cfg, "true_model"),
        device=device
    )

    with torch.no_grad():
        S_true, Y_true = model_true.simulate(u=u_fn, t_eval=t_eval)

    l = model_true.l
    m = model_true.neuronal.m

    noise_std = torch.tensor(cfg["noise"]["std"], device=device)
    Y_obs = Y_true + noise_std * torch.randn_like(Y_true)

    # ============================================================
    # INFERENCE MODEL (DCM MAP)
    # ============================================================

    model_inf = build_model_torch(
        cfg=extract_model_cfg(cfg, "init_model"),
        device=device
    )

    A0 = torch.tensor(cfg["init_model"]["neuronal"]["A"], device=device)
    B0 = torch.tensor(cfg["init_model"]["neuronal"]["B"], device=device)
    C0 = torch.tensor(cfg["init_model"]["neuronal"]["C"], device=device)

    theta0 = torch.cat([A0.flatten(), B0.flatten(), C0.flatten()])
    mu_theta = theta0.clone()

    sigma_cfg = cfg["priors"]["sigma"]

    sigma_prior = torch.cat([
        torch.full((l * l,), sigma_cfg["A"], device=device),
        torch.full((l * l * m,), sigma_cfg["B"], device=device),
        torch.full((l * m,), sigma_cfg["C"], device=device),
    ])

    inference_model = DCMInferenceModel(
        forward_model=model_inf,
        likelihood_fn=gaussian_log_likelihood_torch,
        prior_fn=gaussian_log_prior_torch,
        y_obs=Y_obs,
        sigma=noise_std,
        mu=mu_theta,
        sigma_prior=sigma_prior,
        t_eval=t_eval,
        u_fn=u_fn,
        z0=torch.zeros(l, device=device),
        x0=model_inf.hemodynamic.initial_state()
    )

    # ============================================================
    # MAP ESTIMATION
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

    # update neuronal parameters
    A_est, B_est, C_est = inference_model.unpack_theta(theta_est)

    model_inf.neuronal.A = A_est
    model_inf.neuronal.B = B_est
    model_inf.neuronal.C = C_est

    # ============================================================
    # HYBRID MODEL
    # ============================================================

    mlp = ResidualMLP(l, m)

    hybrid_model = ResidualDCM(
        bilinear=model_inf.neuronal,
        hemodynamic=model_inf.hemodynamic,
        mlp=mlp,
        alpha=cfg["hybrid"]["alpha"]
    )

    # ============================================================
    # BEFORE MLP TRAINING: DCM AFTER MAP OPTIMIZATION
    # ============================================================

    with torch.no_grad():
        S0_dcm, Y_dcm = model_inf.simulate(
            u=u_fn,
            t_eval=t_eval,
            z0=torch.zeros(l, device=device),
            x0=model_inf.hemodynamic.initial_state(),
        )

    # ============================================================
    # DCM DYNAMICS ON PURE DCM TRAJECTORY
    # ============================================================

    dz_dcm_pure = []

    for i in range(len(S0_dcm) - 1):
        z = S0_dcm[i][:l]
        u = u_fn(float(t_eval[i]))
        dz_dcm_pure.append(model_inf.neuronal.dynamics(0.0, z, u))

    dz_dcm_pure = torch.stack(dz_dcm_pure)

    # ============================================================
    # BEFORE MLP TRAINING: DCM AFTER MAP OPTIMIZATION
    # ============================================================
    with torch.no_grad():
        S0_hybrid_0, Y_hybrid_0 = hybrid_model.simulate(
            u=u_fn,
            t_eval=t_eval,
            z0=torch.zeros(l, device=device),
            x0=model_inf.hemodynamic.initial_state(),
        )
    # ============================================================
    # TRAIN MLP
    # ============================================================

    optimizer = torch.optim.Adam(
        mlp.parameters(),
        lr=cfg["hybrid"]["lr"]
    )

    for epoch in range(cfg["hybrid"]["epochs"]):

        S, Y_pred = hybrid_model.simulate(
            u=u_fn,
            t_eval=t_eval,
            z0=torch.zeros(l, device=device),
            x0=model_inf.hemodynamic.initial_state(),
        )

        loss = ((Y_pred - Y_true) ** 2).mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch % 10 == 0:
            print(f"[MLP {epoch}] loss={loss.item():.6f}")

    # ============================================================
    # FINAL SIMULATION
    # ============================================================

    with torch.no_grad():
        S_final, Y_hybrid_T = hybrid_model.simulate(
            u=u_fn,
            t_eval=t_eval,
            z0=torch.zeros(l, device=device),
            x0=model_inf.hemodynamic.initial_state(),
        )

        dz_dcm, dz_res = [], []

        for i in range(len(S_final) - 1):
            z = S_final[i][:l]
            u = u_fn(float(t_eval[i]))

            dz_dcm.append(model_inf.neuronal.dynamics(0.0, z, u))
            dz_res.append(mlp(z, u))

        dz_dcm = torch.stack(dz_dcm)
        dz_res = torch.stack(dz_res)

    # ============================================================
    # SAVE RESULTS
    # ============================================================

    save_npz(
        run_dir / "results.npz",
        t=t_eval.cpu().numpy(),
        Y_true=to_numpy(Y_true),
        Y_obs=to_numpy(Y_obs),
        Y_pred=to_numpy(Y_hybrid_T),
        trace=np.array(trace),
        theta_est=to_numpy(theta_est),
    )

    save_hybrid_diagnostics(
        run_dir=run_dir,

        t=t_eval.detach().cpu().numpy(),
        U=design.U.detach().cpu().numpy(),

        Y_true=to_numpy(Y_true),
        Y_obs=to_numpy(Y_obs),

        Y_dcm=to_numpy(Y_dcm),
        Y_pred=to_numpy(Y_hybrid_T),

        trace=np.array(trace),

        theta_true=to_numpy(theta0),
        theta_est=to_numpy(theta_est),

        A_true=to_numpy(model_true.neuronal.A),
        A_est=to_numpy(A_est),

        B_true=to_numpy(model_true.neuronal.B),
        B_est=to_numpy(B_est),

        C_true=to_numpy(model_true.neuronal.C),
        C_est=to_numpy(C_est),

        dz_dcm=to_numpy(dz_dcm),  # dcm on hybrid trajectory (dcm + initialized mlp)
        dz_res=to_numpy(dz_res),    # mlp trajectory
        dz_dcm_pure=to_numpy(dz_dcm_pure),  # pure dcm trajectory without mlp
    )


# ============================================================
# ENTRY
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="experiments/configs/residual_dcm/inversion_2r_modulatory.yaml",
    )
    args = parser.parse_args()

    main(args.config)