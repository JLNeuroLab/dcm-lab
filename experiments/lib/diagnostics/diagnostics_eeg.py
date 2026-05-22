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
        ax.plot(t, Y_true[:, i], label="true",      linewidth=1.5)
        ax.plot(t, Y_obs[:, i],  label="obs+noise",  alpha=0.5, linewidth=0.8)
        ax.plot(t, Y_est[:, i],  label="estimated",  linestyle="--", linewidth=1.5)
        ax.set_ylabel(f"sensor {i} (mV)")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

    axes[-1].set_xlabel("Time (s)")
    plt.suptitle("EEG: true vs estimated")
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

    # ── metrics ───────────────────────────────────────────────────
    theta_true = np.concatenate([CF_true.flatten(), CB_true.flatten(),
                                  CL_true.flatten(), P_true.flatten()])
    theta_est  = np.concatenate([CF_est.flatten(),  CB_est.flatten(),
                                  CL_est.flatten(),  P_est.flatten()])

    param_error = np.linalg.norm(theta_true - theta_est) / (np.linalg.norm(theta_true) + 1e-8)
    rmse        = float(np.sqrt(np.mean((Y_obs - Y_est) ** 2)))
    corrs       = [float(np.corrcoef(Y_obs[:, i], Y_est[:, i])[0, 1]) for i in range(n_sensors)]

    metrics = {
        "param_error":       float(param_error),
        "rmse":              rmse,
        "mean_correlation":  float(np.mean(corrs)),
        "per_sensor_corr":   corrs,
        "final_loss":        float(trace[-1]) if len(trace) > 0 else None,
    }

    save_json(metrics, run_dir / "metrics.json")
    print("✔ EEG diagnostics saved in:", run_dir)
