from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from experiments.lib.io import save_json
from experiments.lib.utils import _to_np, autocorr, _plot_matrix, _normalize


def save_eeg_diagnostics(
    run_dir,
    t,
    U,
    Y_true,
    Y_obs,
    Y_est,
    trace,
    CF_true, CF_est,
    CB_true, CB_est,
    CL_true, CL_est,
    P_true,  P_est,
    Y_map=None,
    dz_mlp=None,
    dz_map_on_ude=None,
    dz_map_on_map=None,
):
    run_dir = Path(run_dir)
    fig_dir = run_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    CF_true, CF_est = map(_to_np, (CF_true, CF_est))
    CB_true, CB_est = map(_to_np, (CB_true, CB_est))
    CL_true, CL_est = map(_to_np, (CL_true, CL_est))
    P_true,  P_est  = map(_to_np, (P_true,  P_est))
    t = np.array(t)

    # ── connectivity matrices ─────────────────────────────────────
    fig, axes = plt.subplots(4, 2, figsize=(10, 14))

    _plot_matrix(axes[0, 0], _normalize(CF_true), "C_F true")
    _plot_matrix(axes[0, 1], _normalize(CF_est),  "C_F est")
    _plot_matrix(axes[1, 0], _normalize(CB_true), "C_B true")
    _plot_matrix(axes[1, 1], _normalize(CB_est),  "C_B est")
    _plot_matrix(axes[2, 0], _normalize(CL_true), "C_L true")
    _plot_matrix(axes[2, 1], _normalize(CL_est),  "C_L est")
    _plot_matrix(axes[3, 0], _normalize(P_true),  "P true")
    _plot_matrix(axes[3, 1], _normalize(P_est),   "P est")

    plt.tight_layout()
    plt.savefig(fig_dir / "matrices.png", dpi=200)
    plt.close()

    # ── inputs ────────────────────────────────────────────────────
    U = np.array(U)
    n_inputs = U.shape[1] if U.ndim == 2 else 1
    if U.ndim == 1:
        U = U[:, None]

    fig, axes = plt.subplots(n_inputs, 1, figsize=(12, 2 * n_inputs), sharex=True)
    if n_inputs == 1:
        axes = [axes]

    for i, ax in enumerate(axes):
        ax.plot(t, U[:, i], linewidth=1.5)
        ax.set_ylabel(f"u{i}")
        ax.grid(alpha=0.3)

    axes[-1].set_xlabel("Time (s)")
    plt.suptitle("Inputs")
    plt.tight_layout()
    plt.savefig(fig_dir / "inputs.png", dpi=200)
    plt.close()

    # ── loss trace ────────────────────────────────────────────────
    plt.figure(figsize=(7, 3))
    plt.plot(trace)
    plt.title("Loss trace")
    plt.xlabel("Iteration")
    plt.grid()
    plt.tight_layout()
    plt.savefig(fig_dir / "trace.png", dpi=200)
    plt.close()

    # ── EEG signal: true vs obs vs est ───────────────────────────
    n_sensors = Y_true.shape[1]
    fig, axes = plt.subplots(n_sensors, 1, figsize=(12, 3 * n_sensors), sharex=True)
    if n_sensors == 1:
        axes = [axes]

    for i, ax in enumerate(axes):
        ax.plot(t, Y_true[:, i], label="true",       linewidth=1.5)
        ax.plot(t, Y_obs[:, i],  label="obs+noise",  alpha=0.5, linewidth=0.8)
        if Y_map is not None:
            ax.plot(t, Y_map[:, i],  label="MAP init",    linestyle=":", linewidth=1.5, color="orange")
        est_label = "UDE" if Y_map is not None else "estimated"
        ax.plot(t, Y_est[:, i],  label=est_label,     linestyle="--", linewidth=1.5, color="green")
        ax.set_ylabel(f"sensor {i} (mV)")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

    axes[-1].set_xlabel("Time (s)")
    plt.suptitle("EEG: true vs MAP init vs UDE" if Y_map is not None else "EEG: true vs estimated")
    plt.tight_layout()
    plt.savefig(fig_dir / "eeg_fit.png", dpi=200)
    plt.close()

    # ── residuals ─────────────────────────────────────────────────
    residual = Y_obs - Y_est

    plt.figure(figsize=(12, 3))
    for i in range(n_sensors):
        plt.plot(t, residual[:, i], label=f"sensor {i}", alpha=0.8)
    plt.title("Residuals (obs - est)")
    plt.xlabel("Time (s)")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(fig_dir / "residuals.png", dpi=200)
    plt.close()

    # ── residual autocorrelation ──────────────────────────────────
    plt.figure(figsize=(6, 3))
    for i in range(n_sensors):
        ac = autocorr(residual[:, i])
        plt.plot(ac[:200], label=f"sensor {i}")
    plt.title("Residual autocorrelation")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(fig_dir / "residual_autocorr.png", dpi=200)
    plt.close()

    # ── UDE vs MAP contribution (difference in sensor space) ─────
    if Y_map is not None:
        diff = Y_est - Y_map

        fig, axes = plt.subplots(n_sensors, 1, figsize=(12, 3 * n_sensors), sharex=True)
        if n_sensors == 1:
            axes = [axes]

        for i, ax in enumerate(axes):
            ax.plot(t, diff[:, i], linewidth=1.2)
            ax.axhline(0, color="k", linewidth=0.6, linestyle="--")
            ax.set_ylabel(f"Δ sensor {i}")
            ax.grid(alpha=0.3)

        axes[-1].set_xlabel("Time (s)")
        plt.suptitle("Difference: UDE − MAP init")
        plt.tight_layout()
        plt.savefig(fig_dir / "difference_ude_vs_map.png", dpi=200)
        plt.close()

    # ── dynamics contribution ─────────────────────────────────────
    mlp_contrib = None
    if dz_mlp is not None and dz_map_on_ude is not None:
        dz_mlp        = np.array(dz_mlp)
        dz_map_on_ude = np.array(dz_map_on_ude)

        mlp_contrib = float(np.mean(np.abs(dz_mlp)) / (np.mean(np.abs(dz_map_on_ude)) + 1e-8))

        plt.figure(figsize=(8, 3))
        plt.plot(np.mean(np.abs(dz_map_on_ude), axis=1), label="MAP coupling")
        plt.plot(np.mean(np.abs(dz_mlp),         axis=1), label="MLP coupling")
        plt.legend()
        plt.title(f"Dynamics contribution (MLP/MAP ratio = {mlp_contrib:.3f})")
        plt.xlabel("Time step")
        plt.grid()
        plt.tight_layout()
        plt.savefig(fig_dir / "dynamics_contribution.png", dpi=200)
        plt.close()

    # ── dynamics comparison ───────────────────────────────────────
    if dz_mlp is not None and dz_map_on_ude is not None and dz_map_on_map is not None:
        dz_map_on_map = np.array(dz_map_on_map)

        plt.figure(figsize=(8, 3))
        plt.plot(np.mean(np.abs(dz_map_on_map), axis=1), label="MAP coupling (MAP traj)")
        plt.plot(np.mean(np.abs(dz_map_on_ude), axis=1), label="MAP coupling (UDE traj)")
        plt.plot(np.mean(np.abs(dz_mlp),         axis=1), label="MLP coupling (UDE traj)")
        plt.legend()
        plt.title("Dynamics: MAP trajectory vs UDE trajectory decomposition")
        plt.xlabel("Time step")
        plt.grid()
        plt.tight_layout()
        plt.savefig(fig_dir / "dynamics_comparison.png", dpi=200)
        plt.close()

    # ── metrics ───────────────────────────────────────────────────
    theta_true = np.concatenate([CF_true.flatten(), CB_true.flatten(),
                                  CL_true.flatten(), P_true.flatten()])
    theta_est  = np.concatenate([CF_est.flatten(),  CB_est.flatten(),
                                  CL_est.flatten(),  P_est.flatten()])

    param_error = np.linalg.norm(theta_true - theta_est) / (np.linalg.norm(theta_true) + 1e-8)
    rmse        = float(np.sqrt(np.mean((Y_obs - Y_est) ** 2)))
    corrs       = [float(np.corrcoef(Y_obs[:, i], Y_est[:, i])[0, 1]) for i in range(n_sensors)]

    metrics = {
        "param_error":      float(param_error),
        "rmse":             rmse,
        "mean_correlation": float(np.mean(corrs)),
        "per_sensor_corr":  corrs,
        "final_loss":       float(trace[-1]) if len(trace) > 0 else None,
    }

    if Y_map is not None:
        rmse_map  = float(np.sqrt(np.mean((Y_obs - Y_map) ** 2)))
        corrs_map = [float(np.corrcoef(Y_obs[:, i], Y_map[:, i])[0, 1]) for i in range(n_sensors)]
        ude_contribution = float(np.mean(np.abs(Y_est - Y_map)) / (np.mean(np.abs(Y_map)) + 1e-8))
        metrics.update({
            "rmse_map":                rmse_map,
            "rmse_ude":                rmse,
            "rmse_improvement":        rmse_map - rmse,
            "mean_correlation_map":    float(np.mean(corrs_map)),
            "per_sensor_corr_map":     corrs_map,
            "ude_contribution_ratio":  ude_contribution,
            "mlp_map_coupling_ratio":  mlp_contrib,
        })

    save_json(metrics, run_dir / "metrics.json")
    print("✔ EEG diagnostics saved in:", run_dir)
