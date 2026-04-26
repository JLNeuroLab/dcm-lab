# experiments/lib/diagnostics/hybrid_diagnostics.py

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from experiments.lib.io import save_json
from experiments.lib.utils import _to_np


def save_hybrid_diagnostics(
    run_dir,
    t,
    U,
    Y_true,
    Y_obs,
    Y_hybrid_0, 
    Y_pred,     
    trace,
    theta_true,
    theta_est,
    A_true, A_est,
    B_true, B_est,
    C_true, C_est,
    dz_dcm=None,
    dz_res=None,
):
    run_dir = Path(run_dir)
    fig_dir = run_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    # ============================================================
    # CONVERSION
    # ============================================================
    A_true, A_est = map(_to_np, (A_true, A_est))
    B_true, B_est = map(_to_np, (B_true, B_est))
    C_true, C_est = map(_to_np, (C_true, C_est))

    t = np.array(t)

    # ============================================================
    # BOLD COMPARISON
    # ============================================================
    Y_hybrid_T = Y_pred

    plt.figure(figsize=(10, 4))

    for i in range(min(3, Y_true.shape[1])):

        plt.plot(t, Y_true[:, i], label=f"true {i}")

        plt.plot(t, Y_hybrid_0[:, i], "--", label=f"hybrid init {i}")

        plt.plot(t, Y_hybrid_T[:, i], "-.", label=f"hybrid final {i}")

    plt.title("BOLD: DCM vs Hybrid (before/after MLP)")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig(fig_dir / "bold_before_after_mlp.png", dpi=200)
    plt.close()

    # ============================================================
    # RESIDUALS
    # ============================================================
    residual = Y_obs - Y_hybrid_T

    plt.figure(figsize=(10, 4))
    for i in range(residual.shape[1]):
        plt.plot(t, residual[:, i])

    plt.title("Residuals (final hybrid)")
    plt.grid()
    plt.tight_layout()
    plt.savefig(fig_dir / "residuals_hybrid.png", dpi=200)
    plt.close()

    # ============================================================
    # DYNAMICS CONTRIBUTION
    # ============================================================
    mlp_contrib = None

    if dz_dcm is not None and dz_res is not None:

        dz_dcm = _to_np(dz_dcm)
        dz_res = _to_np(dz_res)

        mlp_contrib = np.mean(np.abs(dz_res)) / (np.mean(np.abs(dz_dcm)) + 1e-8)

        plt.figure(figsize=(8, 3))
        plt.plot(np.mean(np.abs(dz_dcm), axis=1), label="DCM")
        plt.plot(np.mean(np.abs(dz_res), axis=1), label="MLP")
        plt.legend()
        plt.title(f"Dynamics contribution (ratio={mlp_contrib:.3f})")
        plt.grid()
        plt.tight_layout()
        plt.savefig(fig_dir / "dynamics_contribution.png", dpi=200)
        plt.close()

    # ============================================================
    # TRACE
    # ============================================================
    plt.figure(figsize=(6, 3))
    plt.plot(trace)
    plt.title("Loss trace")
    plt.grid()
    plt.tight_layout()
    plt.savefig(fig_dir / "trace.png", dpi=200)
    plt.close()

    # ============================================================
    # METRICS
    # ============================================================
    rmse = np.sqrt(np.mean((Y_obs - Y_hybrid_T) ** 2))

    param_error = np.linalg.norm(theta_true - theta_est) / (
        np.linalg.norm(theta_true) + 1e-8
    )

    corrs = [
        np.corrcoef(Y_obs[:, i], Y_hybrid_T[:, i])[0, 1]
        for i in range(Y_obs.shape[1])
    ]

    eigvals = np.linalg.eigvals(A_est)
    stable = bool(np.all(np.real(eigvals) < 0))

    metrics = {
        "rmse": float(rmse),
        "param_error": float(param_error),
        "mean_corr": float(np.mean(corrs)),
        "stable": stable,
        "mlp_contribution_ratio": float(mlp_contrib) if mlp_contrib is not None else None,
        "final_loss": float(trace[-1]) if len(trace) > 0 else None,
    }

    save_json(metrics, run_dir / "metrics_hybrid.json")

    print(f"✔ Hybrid diagnostics saved in: {run_dir}")