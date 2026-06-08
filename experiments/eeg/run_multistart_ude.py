from __future__ import annotations

import argparse
import json
import torch
import numpy as np


import os, sys
sys.path.append(os.path.abspath("/home/student/r/rofritzsche/projects/dcm-lab"))
sys.path.append(os.path.abspath("/home/student/r/rofritzsche/projects/dcm-lab/src"))

from experiments.lib.io import load_yaml, save_yaml, make_run_dir, save_npz
from experiments.lib.utils import build_design_torch, build_eeg_model_torch
from experiments.lib.diagnostics.diagnostics_eeg import save_eeg_diagnostics

from dcm.models.eeg.neuronal_jansen_rit import JansenRitParametersTorch, JansenRitNeuronal
from dcm.models.eeg.lead_field import LeadFieldParametrization
from dcm.simulate.integrators import get_integrator
from ude.hybrid.eeg_coupling_ude import EEGCouplingUDE
from ml.mlp import EEGCouplingMLP

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

def _make_optimizer(params, opt_cfg: dict):
    method = opt_cfg.get("method", "adam").lower()
    if method == "lbfgs":
        return torch.optim.LBFGS(
            params,
            lr       = float(opt_cfg.get("lr",       1.0)),
            max_iter = int(opt_cfg.get("max_iter",   20)),
            line_search_fn='strong_wolfe'
        )
    return torch.optim.Adam(params, lr=float(opt_cfg.get("lr", 1e-3)))


def _jac_penalty(ude, jac_lambda, device):
    """Jacobian sparsity penalty at a fixed operating point (sigmoid midpoint)."""
    S0_fixed = torch.full((ude.l,), 0.5, device=device)
    u_fixed  = torch.zeros(ude.m, device=device)
    J = torch.autograd.functional.jacobian(
            lambda s: ude.mlp(s, u_fixed), S0_fixed)
    return jac_lambda * J.abs().mean()


def _train_step(ude, optimizer, Y_obs, t_eval, u_fn, sensor_var, clip_norm, is_lbfgs, l1_lambda=0.0, jac_lambda=0.0):
    if is_lbfgs:
        def closure():
            optimizer.zero_grad()
            _, Y_pred = ude.simulate(u=u_fn, t_eval=t_eval)
            loss = ((Y_pred - Y_obs) ** 2 / sensor_var).mean()
            if l1_lambda > 0.0:
                loss = loss + l1_lambda * (
                    ude.neuronal.C_F.abs().mean() +
                    ude.neuronal.C_B.abs().mean()
                )
            if jac_lambda > 0.0:
                loss = loss + _jac_penalty(ude, jac_lambda, Y_obs.device)
            loss.backward()
            return loss
        return optimizer.step(closure).item()
    else:
        _, Y_pred = ude.simulate(u=u_fn, t_eval=t_eval)
        loss = ((Y_pred - Y_obs) ** 2 / sensor_var).mean()
        if l1_lambda > 0.0:
            loss = loss + l1_lambda * (
                ude.neuronal.C_F.abs().mean() +
                ude.neuronal.C_B.abs().mean()
            )
        if jac_lambda > 0.0:
            loss = loss + _jac_penalty(ude, jac_lambda, Y_obs.device)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(ude.parameters(), clip_norm)
        optimizer.step()
        return loss.item()

def _build_ude(cfg, P_init, seed, device):
    torch.manual_seed(seed)
    np.random.seed(seed)

    l, m = cfg["model"]["l"], cfg["model"]["m"]

    params = JansenRitParametersTorch.with_defaults(l=l, m=m)
    params.P = P_init.clone().cpu()
    neuronal = JansenRitNeuronal(params=params).to(device)

    #----- random initialization of extrinsic couplings
    with torch.no_grad():
        neuronal.C_F.data = torch.rand(l, l, device=device) * cfg["multistart"]["init"]["C_F_scale"]
        neuronal.C_B.data = torch.rand(l, l, device=device) * cfg["multistart"]["init"]["C_B_scale"]
    #----- freeze biophysical variables, only extrinsic coupling matrices are optimized
    neuronal.set_train_mode("biophysical_frozen")
    freeze_P = cfg.get("freeze_P", False)
    neuronal.P.requires_grad = not freeze_P
    #----- Lead-field parametrization is kept fixed and not optimized
    observer = LeadFieldParametrization(l=l).to(device)
    for p in observer.parameters():
        p.requires_grad = False
    
    mlp = EEGCouplingMLP(l=l, m=m, hidden_dim=cfg["mlp"]["hidden_dim"],
                         u_scale=cfg["mlp"]["u_scale"],
                         output_scale=cfg["mlp"].get("output_scale", 1.0)).to(device)

    integrator_name = cfg.get("simulation", {}).get("integrator", "euler")
    print(f"integrator: {integrator_name}")
    integrator = get_integrator(integrator_name)
    is_adjoint = "adjoint" in integrator_name
    ude_model = EEGCouplingUDE(neuronal=neuronal, observer=observer, mlp=mlp,
                               integrator=integrator, is_adjoint=is_adjoint).to(device)

    return ude_model

# ------ main ---------
def main(config_path: str, best_seed_override: int | None = None, checkpoint_path: str | None = None):
    
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

    _P_init_cfg = cfg.get("init", {}).get("P", None)
    if _P_init_cfg is not None:
        P_init = torch.tensor(_P_init_cfg, dtype=torch.float32, device=device)
        print(f"Using P_init from config['init']['P']")
    else:
        P_init = P_true
        print("Using P_init = P_true")
    noise_std = torch.tensor(cfg["noise"]["std"], device=device)
    Y_obs     = Y_true + noise_std * torch.randn_like(Y_true)
    _t("true model simulate", t0); t0 = time.perf_counter()

    l = int(cfg["model"]["l"])
    m = int(cfg["model"]["m"])

    # ----------------- multi-start phase -------------------
    ms_cfg = cfg["multistart"]

    sensor_var = Y_obs.var(dim=0, keepdim=True).detach() + 1e-8
    norm_loss  = ms_cfg.get("normalized_loss", False)
    if norm_loss:
        print("  Using sensor-normalized loss")

    if checkpoint_path is not None:
        print(f"\nSkipping multi-start — loading checkpoint: {checkpoint_path}")
        best_state_dict = torch.load(checkpoint_path, map_location=device, weights_only=True)
        best = {"seed": 0, "final_loss": float("nan"), "rmse": float("nan")}
    elif best_seed_override is not None:
        print(f"\nSkipping multi-start — using seed={best_seed_override} directly.")
        best_state_dict = None
        best = {"seed": best_seed_override, "final_loss": float("nan"), "rmse": float("nan")}
    else:
        results = []
        ms_opt_cfg = ms_cfg.get("optimizer", {"method": "adam", "lr": ms_cfg.get("lr", 1e-3)})
        ms_clip    = float(ms_cfg.get("clip_grad_norm", 1.0))
        is_lbfgs   = ms_opt_cfg.get("method", "adam").lower() == "lbfgs"
        print(f"  Optimizer: {ms_opt_cfg.get('method', 'adam').upper()}")

        n_restarts = int(ms_cfg["n_restarts"])
        state_dicts = []
        for seed in range(n_restarts):
            print(f"\n--- Restart {seed+1}/{n_restarts} (seed={seed}) ---")
            ude       = _build_ude(cfg, P_init, seed, device)
            optimizer = _make_optimizer(ude.parameters(), ms_opt_cfg)
            trace     = []

            for epoch in range(ms_cfg["epochs_per_restart"]):
                loss_val = _train_step(ude, optimizer, Y_obs, t_eval, u_fn,
                                       sensor_var, ms_clip, is_lbfgs)
                if not np.isfinite(loss_val):
                    print(f"  [{epoch:4d}] NaN/Inf — aborting restart")
                    break
                trace.append(loss_val)
                if epoch % 50 == 0:
                    print(f"  [{epoch:4d}] loss={loss_val:.6f}")

            with torch.no_grad():
                _, Y_pred = ude.simulate(u=u_fn, t_eval=t_eval)
            rmse = ((Y_pred - Y_true) ** 2).mean().sqrt().item()

            results.append({"seed": seed, "final_loss": trace[-1] if trace else float("nan"),
                            "min_loss": min(trace) if trace else float("nan"), "rmse": rmse})
            state_dicts.append({k: v.detach().clone() for k, v in ude.state_dict().items()})
            print(f"  => final={results[-1]['final_loss']:.4f}  min={results[-1]['min_loss']:.4f}  rmse={rmse:.4f}")

        best_idx = min(range(len(results)), key=lambda i: results[i]["final_loss"])
        best = results[best_idx]
        best_state_dict = state_dicts[best_idx]
        print(f"\nBest: seed={best['seed']}, final_loss={best['final_loss']:.4f}, rmse={best['rmse']:.4f}")

    if best_seed_override is None and checkpoint_path is None:
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
        torch.save(best_state_dict, run_dir / "best_multistart_checkpoint.pt")
        print(f"  Checkpoint saved: {run_dir / 'best_multistart_checkpoint.pt'}")
    _t("multi-start", t0); t0 = time.perf_counter()

    # ── full training on best seed ────────────────────────────────
    train_cfg   = cfg["training"]
    full_epochs = int(train_cfg["epochs"])
    full_clip   = float(train_cfg.get("clip_grad_norm", 1.0))

    ft_opt_cfg   = train_cfg.get("optimizer", {"method": "adam", "lr": 1e-3})
    l1_lambda  = float(train_cfg.get("l1_lambda",  0.0))
    jac_lambda = float(train_cfg.get("jac_lambda", 0.0))
    ft_is_lbfgs  = ft_opt_cfg.get("method", "adam").lower() == "lbfgs"
    print(f"\nFull training on best seed={best['seed']} ({full_epochs} epochs, {ft_opt_cfg.get('method','adam').upper()})...")
    ude_best  = _build_ude(cfg, P_init, best["seed"], device)
    if best_state_dict is not None:
        ude_best.load_state_dict(best_state_dict)
    optimizer = _make_optimizer(ude_best.parameters(), ft_opt_cfg)

    sched_cfg = train_cfg.get("lr_schedule", {})
    scheduler = None
    if not ft_is_lbfgs:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            factor   = float(sched_cfg.get("factor",   0.5)),
            patience = int(sched_cfg.get("patience",   50)),
            min_lr   = float(sched_cfg.get("min_lr",   1e-6)),
        )

    full_trace    = []
    full_lr_trace = []

    for epoch in range(full_epochs):
        loss_val = _train_step(ude_best, optimizer, Y_obs, t_eval, u_fn,
                               sensor_var, full_clip, ft_is_lbfgs, l1_lambda, jac_lambda)
        if not np.isfinite(loss_val):
            print(f"  [{epoch:4d}] NaN/Inf — aborting full training")
            break
        if scheduler is not None:
            scheduler.step(loss_val)

        current_lr = optimizer.param_groups[0]["lr"]
        full_trace.append(loss_val)
        full_lr_trace.append(current_lr)
        if epoch % 100 == 0:
            print(f"  [{epoch:4d}] loss={loss_val:.6f}  lr={current_lr:.2e}")

    _t("full training", t0); t0 = time.perf_counter()

    # ── final simulation ──────────────────────────────────────────
    with torch.no_grad():
        S_final, Y_final = ude_best.simulate(u=u_fn, t_eval=t_eval)

    CF_est, CB_est = ude_best.effective_connectivity(S_final, u_fn, t_eval)
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
    parser.add_argument(
        "--best-seed",
        type=int,
        default=None,
        help="Skip multi-start and run full training on this seed directly.",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to best_multistart_checkpoint.pt — skip multi-start and load saved weights.",
    )
    args = parser.parse_args()
    main(args.config, best_seed_override=args.best_seed, checkpoint_path=args.checkpoint)

# python -m experiments.eeg.run_multistart_ude `
 # --config experiments/configs/eeg/multistart_ude_6r_adjoint.yaml `
 # --checkpoint results/runs/2026-06-07_13-09-57_multistart_ude_6r_adjoint/best_multistart_checkpoint.pt
