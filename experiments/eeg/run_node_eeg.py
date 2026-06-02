from __future__ import annotations

import argparse
import torch
import numpy as np

from experiments.lib.io import load_yaml, save_yaml, make_run_dir, save_npz
from experiments.lib.utils import build_design_torch, build_eeg_model_torch
from experiments.lib.diagnostics.diagnostics_eeg import save_eeg_diagnostics

from dcm.models.eeg.lead_field import LeadFieldParametrization
from ude.node.eeg_node import EEGNeuralODE
from ml.mlp import EEGNeuralMLP

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


# ============================================================
# MAIN
# ============================================================

def main(config_path: str):

    t0 = time.perf_counter()

    cfg    = load_yaml(config_path)
    _dev   = cfg.get("device", "cpu")
    device = torch.device("cuda" if _dev == "cuda" and torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    run_dir = make_run_dir(cfg.get("name", "node_eeg"))
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

    l = int(cfg["model"]["l"])
    m = int(cfg["model"]["m"])

    # ── NODE model ───────────────────────────────────────────────
    hidden_dim = int(cfg.get("mlp", {}).get("hidden_dim", 256))
    u_scale    = float(cfg.get("mlp", {}).get("u_scale", 1.0))
    z_scale    = float(cfg.get("mlp", {}).get("z_scale", 1.0))

    # auto z_scale: RMS of the true trajectory if not set in config
    if z_scale == 1.0:
        z_scale = float(S_true.std())
        print(f"  auto z_scale = {z_scale:.4f}")

    use_adjoint = bool(cfg.get("adjoint", False))

    neuronal   = EEGNeuralMLP(l=l, m=m, hidden_dim=hidden_dim,
                               z_scale=z_scale, u_scale=u_scale).to(device)
    observer   = LeadFieldParametrization(l=l).to(device)
    for p in observer.parameters():
        p.requires_grad = False
    node_model = EEGNeuralODE(neuronal=neuronal, observer=observer,
                               adjoint=use_adjoint).to(device)
    print(f"  NODE: hidden_dim={hidden_dim}, z_scale={z_scale:.4f}, u_scale={u_scale}, adjoint={use_adjoint}")
    _t("NODE build", t0); t0 = time.perf_counter()

    # ── warm-start: supervised regression on JR vector field ─────
    ws_cfg = cfg.get("warmstart", {})
    if ws_cfg.get("enabled", False):
        print("NODE warm-start (fitting JR vector field)...")

        with torch.no_grad():
            u_vals = torch.stack([
                torch.as_tensor(u_fn(float(t)), dtype=torch.float32, device=device)
                for t in t_eval
            ])  # (T, m)
            dz_targets = torch.stack([
                model_true.neuronal.dynamics(float(t_eval[i]), S_true[i], u_vals[i])
                for i in range(len(t_eval))
            ]).detach()  # (T, 9*l)

        dz_scale = float(dz_targets.std())
        dz_norm  = dz_targets / dz_scale
        print(f"  dz_scale = {dz_scale:.2f}")

        ws_steps = int(ws_cfg.get("steps", 1000))
        ws_lr    = float(ws_cfg.get("lr", 1e-3))
        ws_opt   = torch.optim.Adam(neuronal.parameters(), lr=ws_lr)

        for step in range(ws_steps):
            dz_pred = neuronal(0.0, S_true, u_vals)
            ws_loss = (dz_pred - dz_norm).pow(2).mean()
            ws_opt.zero_grad()
            ws_loss.backward()
            ws_opt.step()
            if step % 100 == 0:
                print(f"  [ws {step:4d}] loss={ws_loss.item():.6f}")

        with torch.no_grad():
            neuronal.net[-1].weight.data *= dz_scale
            neuronal.net[-1].bias.data   *= dz_scale
        print(f"  warm-start done — final loss: {ws_loss.item():.6f}")
        _t("warm-start", t0); t0 = time.perf_counter()

    # ── training ─────────────────────────────────────────────────
    train_cfg = cfg["training"]
    optimizer = torch.optim.Adam(node_model.parameters(), lr=float(train_cfg["lr"]))
    clip_norm = float(train_cfg.get("clip_grad_norm", 1.0))
    epochs    = int(train_cfg["epochs"])
    norm_loss = train_cfg.get("normalized_loss", False)

    sched_cfg  = train_cfg.get("lr_schedule", {})
    scheduler  = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        factor   = float(sched_cfg.get("factor",   0.5)),
        patience = int(sched_cfg.get("patience",   50)),
        min_lr   = float(sched_cfg.get("min_lr",   1e-6)),
    )

    if norm_loss:
        sensor_var = Y_obs.var(dim=0, keepdim=True).detach() + 1e-8
        print("  Using sensor-normalized loss")

    trace    = []
    lr_trace = []
    print(f"Training NODE ({epochs} epochs)...")

    for epoch in range(epochs):
        _, Y_pred = node_model.simulate(u=u_fn, t_eval=t_eval)
        if norm_loss:
            loss = ((Y_pred - Y_obs) ** 2 / sensor_var).mean()
        else:
            loss = ((Y_pred - Y_obs) ** 2).mean()

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(node_model.parameters(), clip_norm)
        optimizer.step()
        scheduler.step(loss.item())

        current_lr = optimizer.param_groups[0]["lr"]
        trace.append(loss.item())
        lr_trace.append(current_lr)
        if epoch % 50 == 0:
            print(f"  [{epoch:4d}] loss={loss.item():.6f}  lr={current_lr:.2e}")

    _t("training", t0); t0 = time.perf_counter()

    # ── final simulation ──────────────────────────────────────────
    with torch.no_grad():
        _, Y_pred = node_model.simulate(u=u_fn, t_eval=t_eval)

    # ── save + diagnostics ────────────────────────────────────────
    zeros_ll = np.zeros((l, l))
    zeros_lm = np.zeros((l, m))

    save_npz(
        run_dir / "results.npz",
        t=to_numpy(t_eval),
        U=to_numpy(design.U),
        Y_true=to_numpy(Y_true),
        Y_obs=to_numpy(Y_obs),
        Y_pred=to_numpy(Y_pred),
        trace=np.array(trace),
        lr_trace=np.array(lr_trace),
        CF_true=to_numpy(CF_true),
        CB_true=to_numpy(CB_true),
        P_true=to_numpy(P_true),
    )

    save_eeg_diagnostics(
        run_dir=run_dir,
        t=to_numpy(t_eval),
        U=to_numpy(design.U),
        Y_true=to_numpy(Y_true),
        Y_obs=to_numpy(Y_obs),
        Y_est=to_numpy(Y_pred),
        trace=trace,
        CF_true=to_numpy(CF_true), CF_est=zeros_ll,
        CB_true=to_numpy(CB_true), CB_est=zeros_ll,
        CL_true=to_numpy(CL_true), CL_est=zeros_ll,
        P_true=to_numpy(P_true),   P_est=zeros_lm,
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
        default="experiments/configs/eeg/node_6r_eeg.yaml",
    )
    args = parser.parse_args()
    main(args.config)