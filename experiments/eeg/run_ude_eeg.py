from __future__ import annotations

import argparse
import torch
import numpy as np

from experiments.lib.io import load_yaml, save_yaml, make_run_dir, save_npz
from experiments.lib.utils import build_design_torch, build_eeg_model_torch
from experiments.lib.diagnostics.diagnostics_eeg import save_eeg_diagnostics

from dcm.models.eeg.neuronal_jansen_rit import JansenRitParametersTorch, JansenRitNeuronal
from dcm.models.eeg.lead_field import LeadFieldParametrization
from ude.hybrid.eeg_coupling_ude import EEGCouplingUDE
from ml.mlp import EEGCouplingMLP

import os
import time
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


def to_numpy(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.array(x)


def _t(label, t0):
    print(f"  [timing] {label}: {time.perf_counter() - t0:.2f}s")


def extract_model_cfg(cfg, key):
    return {
        "model": cfg["model"],
        "neuronal": cfg[key]["neuronal"],
    }


def effective_connectivity(model: EEGCouplingUDE, S0: torch.Tensor, u_t: torch.Tensor):
    """Linearise MLP at (S0, u_t) → effective C_F (first l rows) and C_B (last l rows)."""
    J = torch.autograd.functional.jacobian(
        lambda s: model.mlp(s, u_t), S0
    )  # (2*l, l)
    l = model.l
    return J[:l].detach(), J[l:].detach()


# ============================================================
# MAIN
# ============================================================

def main(config_path: str):

    t0 = time.perf_counter()

    cfg    = load_yaml(config_path)
    #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device("cpu")
    print(f"Device: {device}")

    run_dir = make_run_dir(cfg.get("name", "ude_eeg"))
    save_yaml(cfg, run_dir / "config.yaml")

    # ── design ──────────────────────────────────────────────────
    design = build_design_torch(cfg, device=device)
    u_fn   = design.callable()
    t_eval = design.t
    _t("design", t0); t0 = time.perf_counter()

    # ── true model + data ────────────────────────────────────────
    model_true = build_eeg_model_torch(extract_model_cfg(cfg, "true_model"), device=device)

    with torch.no_grad():
        S_true, Y_true = model_true.simulate(u=u_fn, t_eval=t_eval)

    CF_true = model_true.neuronal.C_F.detach()
    CB_true = model_true.neuronal.C_B.detach()
    CL_true = model_true.neuronal.C_L.detach()
    P_true  = model_true.neuronal.P.detach()

    noise_std = torch.tensor(cfg["noise"]["std"], device=device)
    Y_obs     = Y_true + noise_std * torch.randn_like(Y_true)
    _t("true model simulate", t0); t0 = time.perf_counter()

    # ── UDE model ────────────────────────────────────────────────
    l = int(cfg["model"]["l"])
    m = int(cfg["model"]["m"])

    params = JansenRitParametersTorch.with_defaults(l=l, m=m)
    if "init" in cfg and "P" in cfg["init"]:
        params.P = torch.tensor(cfg["init"]["P"], dtype=torch.float32).reshape(l, m)
    neuronal = JansenRitNeuronal(params).to(device)
    observer = LeadFieldParametrization(l=l).to(device)
    mlp      = EEGCouplingMLP(l=l, m=m).to(device)

    ude_model = EEGCouplingUDE(
        neuronal=neuronal,
        observer=observer,
        mlp=mlp,
    ).to(device)
    neuronal.set_train_mode("biophysical_frozen")       # fix all JR literature constants
    for p in observer.parameters():
        p.requires_grad = False                         # fix lead field scaling K
    _t("UDE build", t0); t0 = time.perf_counter()

    # ── MLP warm-start ─────────────────────────────────────────────────
    ws_cfg = cfg.get("warmstart", {})
    if ws_cfg.get("enabled", False):
        print(f"MLP warm-start...")

        with torch.no_grad():
            X_true = S_true.reshape(-1, 9, l)          # (T, 9, l)
            S0_true = ude_model.neuronal.sigmoid(X_true[:, 0, :])  # (T, l)
            target_fwd = S0_true @ CF_true.T            # (T, l)
            target_bwd = S0_true @ CB_true.T            # (T, l)
            target = torch.cat([target_fwd, target_bwd], dim=-1)   # (T, 2*l)

        u_zeros = torch.zeros(len(S0_true), m, device=device)

        with torch.no_grad():
            # hidden representations from all layers except the last
            x = torch.cat([S0_true, u_zeros], dim=-1)          # (T, l+m)
            h = mlp.net[:-1](x)                                # (T, hidden_dim)

            # augment with ones for the bias term
            h_aug = torch.cat([h, torch.ones(len(h), 1, device=device)], dim=-1)  # (T, hidden_dim+1)

            # solve h_aug @ [W.T; b] = target  →  least squares
            sol = torch.linalg.lstsq(h_aug, target).solution   # (hidden_dim+1, 2*l)
            mlp.net[-1].weight.data.copy_(sol[:-1].T)           # (2*l, hidden_dim)
            mlp.net[-1].bias.data.copy_(sol[-1])                # (2*l,)

            pred_check = mlp(S0_true, u_zeros)
            print(f"warm-start loss: {(pred_check - target).pow(2).mean():.4f}")

    # ── training ─────────────────────────────────────────────────
    train_cfg = cfg["training"]
    optimizer = torch.optim.Adam(ude_model.parameters(), lr=float(train_cfg["lr"]))
    clip_norm = float(train_cfg.get("clip_grad_norm", 1.0))
    epochs    = int(train_cfg["epochs"])

    trace = []
    print(f"Training UDE ({epochs} epochs)...")

    for epoch in range(epochs):
        _, Y_pred = ude_model.simulate(u=u_fn, t_eval=t_eval)
        loss = ((Y_pred - Y_obs) ** 2).mean()

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(ude_model.parameters(), clip_norm)
        optimizer.step()

        trace.append(loss.item())
        if epoch % 50 == 0:
            print(f"  [{epoch:4d}] loss={loss.item():.6f}")

    P_learned = ude_model.neuronal.P.detach()

    _t("training", t0); t0 = time.perf_counter()

    # ── final simulation ──────────────────────────────────────────
    with torch.no_grad():
        S_final, Y_pred = ude_model.simulate(u=u_fn, t_eval=t_eval)

    # ── effective connectivity via Jacobian ───────────────────────
    X_mean  = S_final.mean(0).reshape(9, l)
    S0_mean = ude_model.neuronal.sigmoid(X_mean[0])
    u_mean  = torch.zeros(m, device=device)
    CF_eff, CB_eff = effective_connectivity(ude_model, S0_mean, u_mean)
    _t("effective connectivity", t0); t0 = time.perf_counter()

    # ── save ─────────────────────────────────────────────────────
    save_npz(
        run_dir / "results.npz",
        t=to_numpy(t_eval),
        U=to_numpy(design.U),
        Y_true=to_numpy(Y_true),
        Y_obs=to_numpy(Y_obs),
        Y_pred=to_numpy(Y_pred),
        trace=np.array(trace),
        CF_true=to_numpy(CF_true), CF_eff=to_numpy(CF_eff),
        CB_true=to_numpy(CB_true), CB_eff=to_numpy(CB_eff),
        P_true=to_numpy(P_true),   P_learned=to_numpy(P_learned),
    )

    # ── diagnostics ───────────────────────────────────────────────
    zeros_ll = np.zeros((l, l))
    zeros_lm = np.zeros((l, m))

    save_eeg_diagnostics(
        run_dir=run_dir,
        t=to_numpy(t_eval),
        U=to_numpy(design.U),
        Y_true=to_numpy(Y_true),
        Y_obs=to_numpy(Y_obs),
        Y_est=to_numpy(Y_pred),
        trace=trace,
        CF_true=to_numpy(CF_true), CF_est=to_numpy(CF_eff),
        CB_true=to_numpy(CB_true), CB_est=to_numpy(CB_eff),
        CL_true=to_numpy(CL_true), CL_est=zeros_ll,
        P_true=to_numpy(P_true),   P_est=to_numpy(P_learned),
    )
    _t("diagnostics", t0)
    print("Done:", run_dir)


# ============================================================
# ENTRY
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="experiments/configs/eeg/ude_3r_eeg.yaml",
    )
    args = parser.parse_args()
    main(args.config)
