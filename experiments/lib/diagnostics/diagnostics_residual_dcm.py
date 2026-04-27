# experiments/lib/diagnostics/hybrid_diagnostics.py

import numpy as np
from scipy.signal import correlate
import matplotlib.pyplot as plt
from pathlib import Path

from experiments.lib.io import save_json
from experiments.lib.utils import _to_np, _plot_matrix, _normalize


def save_hybrid_diagnostics(
    run_dir,
    t,
    U,
    Y_true,
    Y_obs,
    Y_dcm, 
    Y_pred,     
    trace,
    theta_true,
    theta_est,
    A_true, A_est,
    B_true, B_est,
    C_true, C_est,
    dz_dcm=None,
    dz_res=None,
    dz_dcm_pure=None
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
    # INPUTS (DESIGN MATRIX)
    # ============================================================

    plt.figure(figsize=(10, 4))

    for i in range(U.shape[1]):
        plt.plot(t, U[:, i], label=f"u{i}")

    plt.title("Inputs (design matrix)")
    plt.xlabel("time")
    plt.ylabel("u(t)")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig(fig_dir / "inputs.png", dpi=200)
    plt.close()
    # ============================================================
    # BOLD COMPARISON
    # ============================================================
    Y_hybrid_T = Y_pred

    plt.figure(figsize=(10, 4))

    for i in range(min(3, Y_true.shape[1])):

        plt.plot(t, Y_true[:, i], label=f"true {i}")

        plt.plot(t, Y_dcm[:, i], "--", label=f"DCM {i}")

        plt.plot(t, Y_hybrid_T[:, i], "-.", label=f"DCM+MLP {i}")

    plt.title("BOLD: DCM vs Hybrid (trained)")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig(fig_dir / "bold_dcm_vs_hybrid.png", dpi=200)
    plt.close()

    # ============================================================
    # MATRICES
    # ============================================================

    fig, axes = plt.subplots(3, 2, figsize=(12, 10))

    _plot_matrix(axes[0, 0], _normalize(A_true), "A true")
    _plot_matrix(axes[0, 1], _normalize(A_est), "A est")

    _plot_matrix(axes[1, 0], _normalize(np.mean(B_true, axis=0)), "B true")
    _plot_matrix(axes[1, 1], _normalize(np.mean(B_est, axis=0)), "B est")

    _plot_matrix(axes[2, 0], _normalize(C_true), "C true")
    _plot_matrix(axes[2, 1], _normalize(C_est), "C est")

    plt.tight_layout()
    plt.savefig(fig_dir / "matrices.png", dpi=200)
    plt.close()

    # ============================================================
    # DIFFERENCE: HYBRID - DCM
    # ============================================================

    plt.figure(figsize=(10, 4))

    diff = Y_hybrid_T - Y_dcm

    for i in range(diff.shape[1]):
        plt.plot(t, diff[:, i], label=f"Δ {i}")

    plt.title("Difference (Hybrid - DCM)")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig(fig_dir / "difference_hybrid_vs_dcm.png", dpi=200)
    plt.close()
    # ============================================================
    # RESIDUALS
    # ============================================================
    plt.figure(figsize=(10, 4))

    res_dcm = Y_obs - Y_dcm
    res_hybrid = Y_obs - Y_hybrid_T

    for i in range(res_dcm.shape[1]):
        plt.plot(t, res_dcm[:, i], "--", alpha=0.6, label=f"DCM {i}" if i == 0 else None)
        plt.plot(t, res_hybrid[:, i], "-", alpha=0.8, label=f"Hybrid {i}" if i == 0 else None)

    plt.title("Residuals: DCM vs Hybrid")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig(fig_dir / "residuals_dcm_vs_hybrid.png", dpi=200)
    plt.close()

    # ============================================================
    # RESIDUAL AUTOCORRELATION (HYBRID)
    # ============================================================

    plt.figure(figsize=(10, 4))

    max_lag = 50  # adjust if needed

    for i in range(res_hybrid.shape[1]):

        r = res_hybrid[:, i]
        r = r - np.mean(r)

        ac = correlate(r, r, mode="full")
        ac = ac[len(ac)//2:]
        ac = ac / (ac[0] + 1e-8)

        plt.plot(ac[:max_lag], label=f"region {i}")

    plt.title("Residual autocorrelation (Hybrid)")
    plt.xlabel("lag")
    plt.ylabel("autocorr")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig(fig_dir / "residual_autocorr_hybrid.png", dpi=200)
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
    # DYNAMICS: PURE DCM vs HYBRID
    # ============================================================

    if dz_dcm is not None and dz_res is not None and dz_dcm_pure is not None:

        dz_dcm = _to_np(dz_dcm)
        dz_res = _to_np(dz_res)
        dz_dcm_pure = _to_np(dz_dcm_pure)

        plt.figure(figsize=(8, 3))

        plt.plot(np.mean(np.abs(dz_dcm_pure), axis=1), label="DCM (pure traj)")
        plt.plot(np.mean(np.abs(dz_dcm), axis=1), label="DCM (hybrid traj)")
        plt.plot(np.mean(np.abs(dz_res), axis=1), label="MLP")

        plt.legend()
        plt.title("Dynamics: pure DCM vs hybrid decomposition")
        plt.grid()
        plt.tight_layout()
        plt.savefig(fig_dir / "dynamics_comparison.png", dpi=200)
        plt.close()

    # ============================================================
    # INPUT – RESIDUAL COUPLING (LAGGED)
    # ============================================================

    plt.figure(figsize=(10, 4))

    U_np = _to_np(U)
    res_np = _to_np(res_hybrid)

    lags = 10  # adjust depending on TR / sampling

    for i in range(res_np.shape[1]):

        corr_lags = []

        for lag in range(lags):

            if lag == 0:
                u_lag = U_np[:, :]
                r_lag = res_np[:, i]
            else:
                u_lag = U_np[:-lag, :]
                r_lag = res_np[lag:, i]

            # average over input channels
            c = np.mean([
                np.corrcoef(u_lag[:, j], r_lag)[0, 1]
                for j in range(u_lag.shape[1])
            ])

            corr_lags.append(c)

        plt.plot(corr_lags, label=f"region {i}")

    plt.title("Lagged input–residual coupling")
    plt.xlabel("lag (time steps)")
    plt.ylabel("correlation")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig(fig_dir / "input_residual_lagged.png", dpi=200)
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
    rmse_dcm = np.sqrt(np.mean((Y_obs - Y_dcm) ** 2))
    rmse_hybrid = np.sqrt(np.mean((Y_obs - Y_hybrid_T) ** 2))

    param_error = np.linalg.norm(theta_true - theta_est) / (
        np.linalg.norm(theta_true) + 1e-8
    )

    corrs_dcm = [
        np.corrcoef(Y_obs[:, i], Y_dcm[:, i])[0, 1]
        for i in range(Y_obs.shape[1])
    ]

    corrs_hybrid = [
        np.corrcoef(Y_obs[:, i], Y_hybrid_T[:, i])[0, 1]
        for i in range(Y_obs.shape[1])
    ]

    eigvals = np.linalg.eigvals(A_est)
    stable = bool(np.all(np.real(eigvals) < 0))

    metrics = {
        "rmse_dcm": float(rmse_dcm),
        "rmse_hybrid": float(rmse_hybrid),
        "rmse_improvement": float(rmse_dcm - rmse_hybrid),

        "param_error": float(param_error),

        "mean_corr_dcm": float(np.mean(corrs_dcm)),
        "mean_corr_hybrid": float(np.mean(corrs_hybrid)),

        "stable": stable,

        "mlp_contribution_ratio": float(mlp_contrib) if mlp_contrib is not None else None,

        "final_loss": float(trace[-1]) if len(trace) > 0 else None,
    }

    save_json(metrics, run_dir / "metrics_hybrid.json")

    print(f"✔ Hybrid diagnostics saved in: {run_dir}")