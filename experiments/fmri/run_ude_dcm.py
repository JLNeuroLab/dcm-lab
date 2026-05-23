from __future__ import annotations

import torch
import numpy as np
import argparse

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from experiments.lib.io import (
    load_yaml,
    save_yaml,
    make_run_dir,
    save_npz,
)

from experiments.lib.utils import build_design_torch, build_model_torch
from experiments.lib.diagnostics.diagnostics_residual_dcm import save_hybrid_diagnostics

from ude.hybrid.residual_dcm import ResidualDCM
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
    #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device("cpu")

    run_dir = make_run_dir(cfg.get("name", "inversion"))
    save_yaml(cfg, run_dir/"config.yaml")

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
        S_true, Y_true = model_true.simulate(
            u=u_fn,
            t_eval=t_eval,
        )
    l = model_true.neuronal.l
    m = model_true.neuronal.m

    A_true_t = model_true.neuronal.A.detach()
    B_true_t = model_true.neuronal.B.detach()
    C_true_t = model_true.neuronal.C.detach()
    theta_true = torch.cat([A_true_t.flatten(), B_true_t.flatten(), C_true_t.flatten()])

    noise_std = torch.tensor(cfg["noise"]["std"], device=device)
    Y_obs = Y_true + noise_std * torch.randn_like(Y_true)

    # ============================================================
    # INFERENCE MODEL
    # ============================================================
    model_inf = build_model_torch(
        cfg=extract_model_cfg(cfg, "init_model"),
        device=device
    )

    with torch.no_grad():
        _, Y_init = model_inf.simulate(
            u=u_fn,
            t_eval=t_eval,
            z0=torch.zeros(l, device=device),
            x0=model_inf.hemodynamic.initial_state().to(device),
        )

    # ============================================================
    # HYBRID MODEL
    # ============================================================

    bilinear = model_inf.neuronal
    hemodynamic = model_inf.hemodynamic

    mlp = ResidualMLP(l, m).to(device)
    hybrid_model = ResidualDCM(
        bilinear=bilinear,
        hemodynamic=hemodynamic,
        mlp=mlp,
        alpha=cfg["hybrid"]["alpha"]
    ).to(device)

    # ============================================================
    # JOINT TRAINING
    # ============================================================

    mode = cfg["hybrid"].get("mode", "mlp_only")
    is_joint = mode == "joint"

    if is_joint:
        hybrid_model.bilinear.set_train_mode("trainable")
        lr_dcm = cfg["hybrid"].get("lr_dcm", cfg["hybrid"]["lr"])
        optimizer = torch.optim.Adam([
            {"params": hybrid_model.bilinear.parameters(), "lr": lr_dcm},
            {"params": hybrid_model.mlp.parameters(),      "lr": cfg["hybrid"]["lr"]},
        ])
    else:
        raise ValueError(f"mode {mode} not implemented for this run")
    
    clip_norm       = cfg["hybrid"].get("clip_grad_norm", 1.0)
    stab_weight     = cfg["hybrid"].get("stability_weight", 0.01)

    trace = []

    print(f"Training mode: {mode}")

    for epoch in range(cfg["hybrid"]["epochs"]):

        S, Y_pred = hybrid_model.simulate(
            u=u_fn,
            t_eval=t_eval,
            z0=torch.zeros(l, device=device),
            x0=model_inf.hemodynamic.initial_state().to(device)
        )

        loss = ((Y_pred - Y_obs)**2).mean()

        eigvals = torch.linalg.eigvals(hybrid_model.bilinear.A)
        loss = loss + stab_weight * torch.relu(eigvals.real).sum()

        # optimization step
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(hybrid_model.parameters(), clip_norm) 
        optimizer.step()

        trace.append(loss.item())

        if epoch % 10 == 0:
            print(f"[{mode} {epoch}] loss={loss.item():.6f}")

    A_est = hybrid_model.bilinear.A.detach()
    B_est = hybrid_model.bilinear.B.detach()
    C_est = hybrid_model.bilinear.C.detach()
    theta_est = torch.cat([A_est.flatten(), B_est.flatten(), C_est.flatten()])
    # ============================================================
    # FINAL SIMULATION
    # ============================================================

    with torch.no_grad():
        S_hybrid, Y_hybrid = hybrid_model.simulate(
            u=u_fn,
            t_eval=t_eval,
            z0=torch.zeros(l, device=device),
            x0=hybrid_model.hemodynamic.initial_state().to(device)
        )

        dz_dcm, dz_mlp = [], []

        for i in range(len(S_hybrid) - 1):
            z = S_hybrid[i][:l]
            u = u_fn(t_eval[i].item())
            u = u.to(device)

            dz_dcm.append(hybrid_model.bilinear.dynamics(0.0, z, u))
            dz_mlp.append(mlp(z, u))
        
        dz_dcm = torch.stack(dz_dcm)
        dz_mlp = torch.stack(dz_mlp)
    
    # ============================================================
    # SAVE RESULTS
    # ============================================================

    save_npz(
        run_dir / "results.npz",
        t=t_eval.cpu().numpy(),
        Y_true=to_numpy(Y_true),
        Y_obs=to_numpy(Y_obs),
        Y_pred=to_numpy(Y_hybrid),
        theta_est=to_numpy(theta_est),
    )

    save_hybrid_diagnostics(
        run_dir=run_dir,

        t=t_eval.detach().cpu().numpy(),
        U=design.U.detach().cpu().numpy(),

        Y_true=to_numpy(Y_true),
        Y_obs=to_numpy(Y_obs),

        Y_dcm=to_numpy(Y_init),
        Y_pred=to_numpy(Y_hybrid),

        trace=np.array(trace),

        theta_true=to_numpy(theta_true),
        theta_est=to_numpy(theta_est),

        A_true=to_numpy(model_true.neuronal.A),
        A_est=to_numpy(A_est),

        B_true=to_numpy(model_true.neuronal.B),
        B_est=to_numpy(B_est),

        C_true=to_numpy(model_true.neuronal.C),
        C_est=to_numpy(C_est),

        dz_dcm=to_numpy(dz_dcm),
        dz_res=to_numpy(dz_mlp),
        dz_dcm_pure=None,
    )


# ============================================================
# ENTRY
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="experiments/configs/residual_dcm/inversion_2r_challenging_hybrid.yaml",
    )
    args = parser.parse_args()

    main(args.config)