from __future__ import annotations

import argparse
import json
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

def _build_ude(cfg, P_true, seed, device):
    torch.manual_seed(seed)
    np.random.seed(seed)

    l, m = cfg["model"]["l"], cfg["model"]["m"]

    params = JansenRitParametersTorch.with_defaults(l=l, m=m)
    params.P = P_true.clone().cpu()
    neuronal = JansenRitNeuronal(params=params).to(device)

    #----- random initialization of extrinsic couplings
    with torch.no_grad():
        neuronal.C_F.data = torch.rand(l, l, device=device) * cfg["multistart"]["init"]["C_F_scale"]
        neuronal.C_B.data = torch.rand(l, l, device=device) * cfg["multistart"]["init"]["C_B_scale"]
    #----- freeze biophysical variables, only extrinsic coupling matrices are optimized
    neuronal.set_train_mode("biophysical_frozen")
    neuronal.P.requires_grad = False
    #----- Lead-field parametrization is kept fixed and not optimized
    observer = LeadFieldParametrization(l=l).to(device)
    for p in observer.parameters():
        p.requires_grad = False
    
    mlp = EEGCouplingMLP(l=l, m=m, hidden_dim=cfg["mlp"]["hidden_dim"], u_scale=cfg["mlp"]["u_scale"]).to(device)
    ude_model = EEGCouplingUDE(neuronal=neuronal, observer=observer, mlp=mlp).to(device)

    return ude_model

# ------ main ---------
def main(config_path: str):
    
    t0 = time.perf_counter()

    cfg    = load_yaml(config_path)
    _dev   = cfg.get("device", "cpu")
    device = torch.device("cuda" if _dev == "cuda" and torch.cuda.is_available() else "cpu")
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

    l = int(cfg["model"]["l"])
    m = int(cfg["model"]["m"])

    # ----------------- multi-start phase -------------------
    ms_cfg = cfg["multistart"]
    results = []

    sensor_var = Y_obs.var(dim=0, keepdim=True).detach() + 1e-8
    norm_loss  = ms_cfg.get("normalized_loss", False)
    if norm_loss:
        print("  Using sensor-normalized loss")
        
    n_restarts = int(ms_cfg["n_restarts"])
    for seed in range(n_restarts):
        print(f"\n--- Restart {seed+1}/{n_restarts} (seed={seed}) ---")
        ude = _build_ude(cfg, P_true, seed, device)
        optimizer = torch.optim.Adam(ude.parameters(), lr=ms_cfg["lr"])
        trace = []

        for epoch in range(ms_cfg["epochs_per_restart"]):
            _, Y_pred = ude.simulate(u=u_fn, t_eval=t_eval)
            loss = ((Y_obs - Y_pred)**2 / sensor_var).mean()
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(ude.parameters(), ms_cfg["clip_grad_norm"])
            optimizer.step()
            trace.append(loss.item())
            if epoch % 50 == 0:
                print(f"  [{epoch:4d}] loss={loss.item():.6f}")

        with torch.no_grad():
            _, Y_pred = ude.simulate(u=u_fn, t_eval=t_eval)
        rmse = ((Y_pred - Y_true) ** 2).mean().sqrt().item()

        results.append({"seed": seed, "final_loss": trace[-1],
                        "min_loss": min(trace), "rmse": rmse})
        print(f"  => final={trace[-1]:.4f}  min={min(trace):.4f}  rmse={rmse:.4f}")

    # ── summary ───────────────────────────────────────────────────
    best = min(results, key=lambda r: r["final_loss"])
    print(f"\nBest: seed={best['seed']}, final_loss={best['final_loss']:.4f}, rmse={best['rmse']:.4f}")

    save_npz(
        run_dir / "multistart_summary.npz",
        seeds        = np.array([r["seed"]       for r in results]),
        final_losses = np.array([r["final_loss"]  for r in results]),
        min_losses   = np.array([r["min_loss"]    for r in results]),
        rmses        = np.array([r["rmse"]        for r in results]),
        best_seed    = np.array(best["seed"]),
    )

    summary_json = {
        "best_seed": int(best["seed"]),
        "restarts": [
            {
                "seed":       int(r["seed"]),
                "final_loss": round(float(r["final_loss"]), 6),
                "min_loss":   round(float(r["min_loss"]),   6),
                "rmse":       round(float(r["rmse"]),       6),
            }
            for r in sorted(results, key=lambda r: r["final_loss"])
        ],
    }
    with open(run_dir / "multistart_summary.json", "w") as f:
        json.dump(summary_json, f, indent=2)
    _t("multi-start", t0); t0 = time.perf_counter()

    # ── full training on best seed ────────────────────────────────
    train_cfg   = cfg["training"]
    full_epochs = int(train_cfg["epochs"])
    full_lr     = float(train_cfg["lr"])
    full_clip   = float(train_cfg.get("clip_grad_norm", 1.0))

    print(f"\nFull training on best seed={best['seed']} ({full_epochs} epochs)...")
    ude_best  = _build_ude(cfg, P_true, best["seed"], device)
    optimizer = torch.optim.Adam(ude_best.parameters(), lr=full_lr)

    sched_cfg = train_cfg.get("lr_schedule", {})
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        factor   = float(sched_cfg.get("factor",   0.5)),
        patience = int(sched_cfg.get("patience",   50)),
        min_lr   = float(sched_cfg.get("min_lr",   1e-6)),
    )

    full_trace    = []
    full_lr_trace = []

    for epoch in range(full_epochs):
        _, Y_pred = ude_best.simulate(u=u_fn, t_eval=t_eval)
        loss = ((Y_pred - Y_obs) ** 2 / sensor_var).mean()
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(ude_best.parameters(), full_clip)
        optimizer.step()
        scheduler.step(loss.item())

        current_lr = optimizer.param_groups[0]["lr"]
        full_trace.append(loss.item())
        full_lr_trace.append(current_lr)
        if epoch % 100 == 0:
            print(f"  [{epoch:4d}] loss={loss.item():.6f}  lr={current_lr:.2e}")

    _t("full training", t0); t0 = time.perf_counter()

    # ── final simulation ──────────────────────────────────────────
    with torch.no_grad():
        _, Y_final = ude_best.simulate(u=u_fn, t_eval=t_eval)

    CF_est = ude_best.neuronal.C_F.detach()
    CB_est = ude_best.neuronal.C_B.detach()
    P_est  = ude_best.neuronal.P.detach()
    zeros_ll = np.zeros((l, l))

    # ── save results ─────────────────────────────────────────────
    save_npz(
        run_dir / "results.npz",
        t          = to_numpy(t_eval),
        U          = to_numpy(design.U),
        Y_true     = to_numpy(Y_true),
        Y_obs      = to_numpy(Y_obs),
        Y_pred     = to_numpy(Y_final),
        trace      = np.array(full_trace),
        lr_trace   = np.array(full_lr_trace),
        CF_true    = to_numpy(CF_true),  CF_est = to_numpy(CF_est),
        CB_true    = to_numpy(CB_true),  CB_est = to_numpy(CB_est),
        P_true     = to_numpy(P_true),   P_est  = to_numpy(P_est),
        best_seed  = np.array(best["seed"]),
    )

    # ── diagnostics ───────────────────────────────────────────────
    save_eeg_diagnostics(
        run_dir    = run_dir,
        t          = to_numpy(t_eval),
        U          = to_numpy(design.U),
        Y_true     = to_numpy(Y_true),
        Y_obs      = to_numpy(Y_obs),
        Y_est      = to_numpy(Y_final),
        trace      = full_trace,
        CF_true    = to_numpy(CF_true),  CF_est = to_numpy(CF_est),
        CB_true    = to_numpy(CB_true),  CB_est = to_numpy(CB_est),
        CL_true    = to_numpy(CL_true),  CL_est = zeros_ll,
        P_true     = to_numpy(P_true),   P_est  = to_numpy(P_est),
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
        default="experiments/configs/eeg/multistart_ude_6r_linear.yaml",
    )
    args = parser.parse_args()
    main(args.config)