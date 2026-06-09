from __future__ import annotations

import argparse
import json
import torch
import numpy as np

import os, sys
sys.path.append(os.path.abspath("/home/student/r/rofritzsche/projects/dcm-lab"))
sys.path.append(os.path.abspath("/home/student/r/rofritzsche/projects/dcm-lab/src"))
sys.path.append(os.path.abspath("/home/student/r/rofritzsche/projects/dcm-lab/experiments"))

from experiments.lib.io import load_yaml, save_yaml, make_run_dir, save_npz
from experiments.lib.utils import build_design_torch, build_eeg_model_torch
from experiments.lib.diagnostics.diagnostics_eeg import save_eeg_diagnostics

from dcm.models.eeg.lead_field import LeadFieldParametrization
from ude.node.eeg_node import EEGNeuralODE
from ml.mlp import EEGNeuralMLP



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


def _build_node(cfg, z_scale, seed, device):
    torch.manual_seed(seed)
    np.random.seed(seed)
    l          = int(cfg["model"]["l"])
    m          = int(cfg["model"]["m"])
    hidden_dim = int(cfg.get("mlp", {}).get("hidden_dim", 256))
    u_scale    = float(cfg.get("mlp", {}).get("u_scale", 1.0))
    integrator_name = cfg.get("simulation", {}).get("integrator", "euler")
    use_adjoint = "adjoint" in integrator_name
    neuronal   = EEGNeuralMLP(l=l, m=m, hidden_dim=hidden_dim,
                               z_scale=z_scale, u_scale=u_scale).to(device)
    observer   = LeadFieldParametrization(l=l).to(device)
    for p in observer.parameters():
        p.requires_grad = False
    return EEGNeuralODE(neuronal=neuronal, observer=observer, adjoint=use_adjoint).to(device)


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
# ============================================================
# MAIN
# ============================================================

def main(config_path: str, best_seed_override: int | None = None, checkpoint_path: str | None = None):

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

    # ── auto z_scale ─────────────────────────────────────────────
    z_scale = float(cfg.get("mlp", {}).get("z_scale", 1.0))
    if z_scale == 1.0:
        z_scale = float(S_true.std())
        print(f"  auto z_scale = {z_scale:.4f}")

    integrator_name = cfg.get("simulation", {}).get("integrator", "euler")
    print(f"  integrator={integrator_name}")

    ms_cfg = cfg.get("multistart", None)

    # ── multi-start phase ─────────────────────────────────────────
    if ms_cfg is not None:
        norm_loss  = ms_cfg.get("normalized_loss", False)
        sensor_var = Y_obs.var(dim=0, keepdim=True).detach() + 1e-8 if norm_loss else 1.0
        if norm_loss:
            print("  Using sensor-normalized loss (multistart)")

        if checkpoint_path is not None:
            print(f"\nSkipping multi-start — loading checkpoint: {checkpoint_path}")
            best_state_dict = torch.load(checkpoint_path, map_location=device, weights_only=True)
            best = {"seed": 0, "final_loss": float("nan"), "rmse": float("nan")}
        elif best_seed_override is not None:
            print(f"\nSkipping multi-start — using seed={best_seed_override} directly.")
            best_state_dict = None
            best = {"seed": best_seed_override, "final_loss": float("nan"), "rmse": float("nan")}
        else:
            ms_opt_cfg = ms_cfg.get("optimizer", {"method": "adam", "lr": 1e-3})
            ms_clip    = float(ms_cfg.get("clip_grad_norm", 1.0))
            is_lbfgs   = ms_opt_cfg.get("method", "adam").lower() == "lbfgs"
            n_restarts = int(ms_cfg["n_restarts"])
            print(f"  Optimizer: {ms_opt_cfg.get('method', 'adam').upper()}, restarts: {n_restarts}")

            results     = []
            state_dicts = []
            for seed in range(n_restarts):
                print(f"\n--- Restart {seed+1}/{n_restarts} (seed={seed}) ---")
                node   = _build_node(cfg, z_scale, seed, device)
                opt    = _make_optimizer(node.parameters(), ms_opt_cfg)
                trace  = []

                for epoch in range(ms_cfg["epochs_per_restart"]):
                    loss_val = _train_step(node, opt, Y_obs, t_eval, u_fn,
                                           sensor_var, ms_clip, is_lbfgs)
                    if not np.isfinite(loss_val):
                        print(f"  [{epoch:4d}] NaN/Inf — aborting restart")
                        break
                    trace.append(loss_val)
                    if epoch % 10 == 0:
                        print(f"  [{epoch:4d}] loss={loss_val:.6f}")

                with torch.no_grad():
                    _, Y_pred = node.simulate(u=u_fn, t_eval=t_eval)
                rmse = ((Y_pred - Y_true) ** 2).mean().sqrt().item()

                results.append({"seed": seed,
                                "final_loss": trace[-1] if trace else float("nan"),
                                "min_loss":   min(trace) if trace else float("nan"),
                                "rmse": rmse})
                state_dicts.append({k: v.detach().clone() for k, v in node.state_dict().items()})
                print(f"  => final={results[-1]['final_loss']:.4f}  min={results[-1]['min_loss']:.4f}  rmse={rmse:.4f}")

            best_idx        = min(range(len(results)), key=lambda i: results[i]["final_loss"])
            best            = results[best_idx]
            best_state_dict = state_dicts[best_idx]
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
            torch.save(best_state_dict, run_dir / "best_multistart_checkpoint.pt")
            print(f"  Checkpoint saved: {run_dir / 'best_multistart_checkpoint.pt'}")

        _t("multi-start", t0); t0 = time.perf_counter()

        # build best node and load weights
        node_model = _build_node(cfg, z_scale, best["seed"], device)
        if best_state_dict is not None:
            node_model.load_state_dict(best_state_dict)

    else:
        # ── single-run (no multistart section in config) ──────────
        hidden_dim = int(cfg.get("mlp", {}).get("hidden_dim", 256))
        u_scale    = float(cfg.get("mlp", {}).get("u_scale", 1.0))
        use_adjoint = "adjoint" in integrator_name
        neuronal   = EEGNeuralMLP(l=l, m=m, hidden_dim=hidden_dim,
                                   z_scale=z_scale, u_scale=u_scale).to(device)
        observer   = LeadFieldParametrization(l=l).to(device)
        for p in observer.parameters():
            p.requires_grad = False
        node_model = EEGNeuralODE(neuronal=neuronal, observer=observer,
                                   adjoint=use_adjoint).to(device)
        print(f"  NODE: hidden_dim={hidden_dim}, z_scale={z_scale:.4f}, u_scale={u_scale}")
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
        ws_opt   = torch.optim.Adam(node_model.neuronal.parameters(), lr=ws_lr)

        for step in range(ws_steps):
            dz_pred = node_model.neuronal(0.0, S_true, u_vals)
            ws_loss = (dz_pred - dz_norm).pow(2).mean()
            ws_opt.zero_grad()
            ws_loss.backward()
            ws_opt.step()
            if step % 100 == 0:
                print(f"  [ws {step:4d}] loss={ws_loss.item():.6f}")

        with torch.no_grad():
            node_model.neuronal.net[-1].weight.data *= dz_scale
            node_model.neuronal.net[-1].bias.data   *= dz_scale
        print(f"  warm-start done — final loss: {ws_loss.item():.6f}")
        _t("warm-start", t0); t0 = time.perf_counter()

    # ── training ─────────────────────────────────────────────────
    train_cfg  = cfg["training"]
    ft_opt_cfg = train_cfg.get("optimizer", {"method": "adam", "lr": train_cfg.get("lr", 1e-3)})
    optimizer  = _make_optimizer(node_model.parameters(), ft_opt_cfg)
    is_lbfgs   = ft_opt_cfg.get("method", "adam").lower() == "lbfgs"
    clip_norm  = float(train_cfg.get("clip_grad_norm", 1.0))
    epochs     = int(train_cfg["epochs"])
    jac_lambda = float(train_cfg.get("jac_lambda", 0.0))
    l1_lambda  = float(train_cfg.get("l1_lambda",  0.0))
    norm_loss  = train_cfg.get("normalized_loss", False)

    sched_cfg  = train_cfg.get("lr_schedule", {})
    scheduler  = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        factor   = float(sched_cfg.get("factor",   0.5)),
        patience = int(sched_cfg.get("patience",   80)),
        min_lr   = float(sched_cfg.get("min_lr",   1e-6)),
    )

    if norm_loss:
        sensor_var = Y_obs.var(dim=0, keepdim=True).detach() + 1e-8
        print("  Using sensor-normalized loss")
    else:
        sensor_var = 1.0

    trace    = []
    lr_trace = []
    print(f"Training NODE ({epochs} epochs)...")

    for epoch in range(epochs):
        loss_val = _train_step(node_model, optimizer, Y_obs, t_eval, u_fn,
                               sensor_var, clip_norm, is_lbfgs, l1_lambda, jac_lambda)
        scheduler.step(loss_val)
        current_lr = optimizer.param_groups[0]["lr"]
        trace.append(loss_val)
        lr_trace.append(current_lr)
        if epoch % 50 == 0:
            print(f"  [{epoch:4d}] loss={loss_val:.6f}  lr={current_lr:.2e}")

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
