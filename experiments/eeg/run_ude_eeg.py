from __future__ import annotations

import argparse
import torch
import numpy as np

from experiments.lib.io import load_yaml, save_yaml, make_run_dir, save_npz
from experiments.lib.utils import build_design_torch, build_eeg_model_torch
from experiments.lib.diagnostics.diagnostics_eeg import save_eeg_diagnostics

from dcm.models.eeg.neuronal_jansen_rit import JansenRitParametersTorch, JansenRitNeuronal
from dcm.models.eeg.lead_field import LeadFieldParametrization
from dcm.inference.objectives import EEGInferenceModel
from dcm.inference.optim import map_estimation_torch
from dcm.inference.likelihoods import gaussian_log_likelihood_torch
from dcm.inference.priors import gaussian_log_prior_torch
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

    # ── MAP pre-estimation ────────────────────────────────────────
    map_cfg     = cfg.get("map_init", {})
    map_enabled = map_cfg.get("enabled", False)
    CF_map = CB_map = CL_map = P_map = None

    if map_enabled:
        print("Running MAP pre-estimation...")
        init_n = map_cfg["init_model"]["neuronal"]
        CF0 = torch.tensor(init_n["C_F"], dtype=torch.float32, device=device).reshape(l, l)
        CB0 = torch.tensor(init_n["C_B"], dtype=torch.float32, device=device).reshape(l, l)
        CL0 = torch.tensor(init_n["C_L"], dtype=torch.float32, device=device).reshape(l, l)
        P0  = torch.tensor(init_n["P"],   dtype=torch.float32, device=device).reshape(l, m)

        theta0   = torch.cat([CF0.flatten(), CB0.flatten(), CL0.flatten(), P0.flatten()])
        mu_theta = theta0.clone()

        sc = map_cfg["priors"]["sigma"]
        sigma_prior = torch.cat([
            torch.full((l * l,), sc["C_F"], device=device),
            torch.full((l * l,), sc["C_B"], device=device),
            torch.full((l * l,), sc["C_L"], device=device),
            torch.full((l * m,), sc["P"],   device=device),
        ])

        model_map = build_eeg_model_torch({"model": cfg["model"], "neuronal": init_n}, device=device)

        inference_model = EEGInferenceModel(
            forward_model=model_map,
            likelihood_fn=gaussian_log_likelihood_torch,
            prior_fn=gaussian_log_prior_torch,
            y_obs=Y_obs,
            sigma=noise_std,
            mu=mu_theta,
            sigma_prior=sigma_prior,
            t_eval=design.t,
            u_fn=u_fn,
            z0=torch.zeros(9*l, device=device)
        )

        opt_cfg = map_cfg["optimizer"]

        theta_est, _, _ = map_estimation_torch(
            model=inference_model,
            theta=theta0.clone().detach().requires_grad_(True),
            n_steps=int(opt_cfg["max_iter"]),
            lr=float(opt_cfg.get("lr", 1.0)),
            method=opt_cfg["method"].lower(),
            verbose=True
        )
        with torch.no_grad():
            CF_map, CB_map, CL_map, P_map = inference_model.unpack_theta(theta_est)
            model_map.neuronal.C_F.data.copy_(CF_map)
            model_map.neuronal.C_B.data.copy_(CB_map)
            model_map.neuronal.C_L.data.copy_(CL_map)
            model_map.neuronal.P.data.copy_(P_map)
            S_map, Y_map = model_map.simulate(u=u_fn, t_eval=t_eval)

        print(f"  MAP done — P_map:\n{P_map}")
        _t("MAP pre-estimation", t0); t0 = time.perf_counter()
    else:
        S_map = Y_map = None

    # ── UDE model ───────────────────────────────────────────────

    params = JansenRitParametersTorch.with_defaults(l=l, m=m)
    if map_enabled and P_map is not None:
        params.P = P_map.clone().cpu()
    elif "init" in cfg and "P" in cfg["init"]:
        params.P = torch.tensor(cfg["init"]["P"], dtype=torch.float32).reshape(l, m)
    neuronal   = JansenRitNeuronal(params).to(device)
    observer   = LeadFieldParametrization(l=l).to(device)
    hidden_dim = int(cfg.get("mlp", {}).get("hidden_dim", 32))
    mlp        = EEGCouplingMLP(l=l, m=m, hidden_dim=hidden_dim).to(device)

    ude_model = EEGCouplingUDE(
        neuronal=neuronal,
        observer=observer,
        mlp=mlp,
    ).to(device)
    neuronal.set_train_mode("biophysical_frozen")       # fix all JR literature constants
    for p in observer.parameters():
        p.requires_grad = False                         # fix lead field scaling K
    if map_enabled and map_cfg.get("freeze_P_after_map", False):
        ude_model.neuronal.P.requires_grad = False
        print("  P frozen at MAP estimate — only MLP trains")
    _t("UDE build", t0); t0 = time.perf_counter()

    # ── MLP warm-start ─────────────────────────────────────────────────
    ws_cfg = cfg.get("warmstart", {})
    if ws_cfg.get("enabled", False):
        print(f"MLP warm-start...")

        with torch.no_grad():
            if map_enabled:
                S_ws = S_map    # reuse already-simulated MAP trajectory
            else:
                S_ws = S_true   # oracle fallback when MAP is off

            X_ws  = S_ws.reshape(-1, 9, l)
            S0_ws = ude_model.neuronal.sigmoid(X_ws[:, 0, :])   # (T, l)

            # actual inputs over the trajectory
            u_vals = torch.stack([
                torch.as_tensor(u_fn(float(t)), dtype=torch.float32, device=device)
                for t in t_eval
            ])  # (T, m)

            CF_ws = CF_map if (map_enabled and CF_map is not None) else CF_true
            CB_ws = CB_map if (map_enabled and CB_map is not None) else CB_true

            # use true model's coupling function when available (captures nonlinear gain)
            true_neuronal = model_true.neuronal
            coupling_S0 = torch.stack([
                true_neuronal.coupling_fn(S0_ws[i], u_vals[i])
                for i in range(len(S0_ws))
            ])  # (T, l)
            target = torch.cat([coupling_S0 @ CF_ws.T, coupling_S0 @ CB_ws.T], dim=-1)  # (T, 2l)

        ws_steps = int(ws_cfg.get("steps", 500))
    ws_lr    = float(ws_cfg.get("lr", 1e-3))
    ws_opt   = torch.optim.Adam(mlp.parameters(), lr=ws_lr)
    for step in range(ws_steps):
        ws_pred = mlp(S0_ws, u_vals)
        ws_loss = (ws_pred - target).pow(2).mean()
        ws_opt.zero_grad()
        ws_loss.backward()
        ws_opt.step()
        if step % 100 == 0:
            print(f"  [ws {step:4d}] loss={ws_loss.item():.6f}")
    print(f"  warm-start done — final loss: {ws_loss.item():.6f}")

    # ── training ─────────────────────────────────────────────────
    train_cfg     = cfg["training"]
    if train_cfg.get("freeze_P", False):
        ude_model.neuronal.P.requires_grad = False
        print("  P frozen at initial value — only MLP trains")
    optimizer     = torch.optim.Adam(ude_model.parameters(), lr=float(train_cfg["lr"]))
    clip_norm     = float(train_cfg.get("clip_grad_norm", 1.0))
    epochs        = int(train_cfg["epochs"])
    norm_loss     = train_cfg.get("normalized_loss", False)

    if norm_loss:
        sensor_var = Y_obs.var(dim=0, keepdim=True).detach() + 1e-8
        print("  Using sensor-normalized loss")

    trace = []
    print(f"Training UDE ({epochs} epochs)...")

    for epoch in range(epochs):
        _, Y_pred = ude_model.simulate(u=u_fn, t_eval=t_eval)
        if norm_loss:
            loss = ((Y_pred - Y_obs) ** 2 / sensor_var).mean()
        else:
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

    # ── dynamics contribution (only when MAP was run) ─────────────
    dz_mlp = dz_map_on_ude = dz_map_on_map = None
    if map_enabled:
        with torch.no_grad():
            X_ude      = S_final.reshape(-1, 9, l)
            S0_ude_traj = ude_model.neuronal.sigmoid(X_ude[:, 0, :])       # (T, l)

            X_map_traj  = S_map.reshape(-1, 9, l)
            S0_map_traj = ude_model.neuronal.sigmoid(X_map_traj[:, 0, :])  # (T, l)

            u_z = torch.zeros(1, m, device=device)
            mlp_list, map_ude_list, map_map_list = [], [], []
            for s_ude, s_map in zip(S0_ude_traj, S0_map_traj):
                mlp_list.append(ude_model.mlp(s_ude.unsqueeze(0), u_z).squeeze(0))
                map_ude_list.append(torch.cat([CF_map @ s_ude, CB_map @ s_ude]))
                map_map_list.append(torch.cat([CF_map @ s_map, CB_map @ s_map]))

            dz_mlp        = torch.stack(mlp_list)      # (T, 2l) — MLP on UDE traj
            dz_map_on_ude = torch.stack(map_ude_list)  # (T, 2l) — MAP coupling on UDE traj
            dz_map_on_map = torch.stack(map_map_list)  # (T, 2l) — MAP coupling on MAP traj

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
        Y_map=to_numpy(Y_map) if Y_map is not None else None,
        dz_mlp=to_numpy(dz_mlp) if dz_mlp is not None else None,
        dz_map_on_ude=to_numpy(dz_map_on_ude) if dz_map_on_ude is not None else None,
        dz_map_on_map=to_numpy(dz_map_on_map) if dz_map_on_map is not None else None,
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
