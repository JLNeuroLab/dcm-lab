import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json
from scipy.signal import correlate

from experiments.lib.io import save_json
from experiments.lib.utils import _to_np, autocorr, _plot_matrix, _normalize


def plot_theta_trajectories(theta_trace, run_dir, l, m, max_params=5):
    import numpy as np
    import matplotlib.pyplot as plt
    from pathlib import Path

    theta_trace = np.stack([t.numpy() for t in theta_trace])  # [T, dim]

    fig_dir = Path(run_dir) / "figures"

    # ============================================================
    # A PARAMETERS
    # ============================================================
    A_dim = l * l
    A_trace = theta_trace[:, :A_dim]

    plt.figure(figsize=(10, 3))
    for i in range(min(max_params, A_dim)):
        plt.plot(A_trace[:, i], label=f"A{i}")

    plt.title("A parameters trajectory")
    plt.grid()
    plt.legend()
    plt.tight_layout()
    plt.savefig(fig_dir / "theta_A.png", dpi=200)
    plt.close()

    # ============================================================
    # B PARAMETERS
    # ============================================================
    B_start = A_dim
    B_dim = l * l * m
    B_trace = theta_trace[:, B_start:B_start + B_dim]

    plt.figure(figsize=(10, 3))
    for i in range(min(max_params, B_dim)):
        plt.plot(B_trace[:, i], label=f"B{i}")

    plt.title("B parameters trajectory")
    plt.grid()
    plt.legend()
    plt.tight_layout()
    plt.savefig(fig_dir / "theta_B.png", dpi=200)
    plt.close()

    # ============================================================
    # C PARAMETERS
    # ============================================================
    C_start = B_start + B_dim
    C_trace = theta_trace[:, C_start:]

    plt.figure(figsize=(10, 3))
    for i in range(min(max_params, C_trace.shape[1])):
        plt.plot(C_trace[:, i], label=f"C{i}")

    plt.title("C parameters trajectory")
    plt.grid()
    plt.legend()
    plt.tight_layout()
    plt.savefig(fig_dir / "theta_C.png", dpi=200)
    plt.close()

    print("✔ theta trajectory plots saved")

# ============================================================
# MAIN DIAGNOSTICS
# ============================================================

def save_dcm_diagnostics(
        run_dir,
        t,
        U,
        Y_true,
        Y_obs,
        Y_est,
        trace,
        theta_true,
        theta_est,
        A_true, A_est,
        B_true, B_est,
        C_true, C_est,
):
    run_dir = Path(run_dir)
    fig_dir = run_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    # ============================================================
    # NUMPY CONVERSION
    # ============================================================

    A_true, A_est = map(_to_np, (A_true, A_est))
    B_true, B_est = map(_to_np, (B_true, B_est))
    C_true, C_est = map(_to_np, (C_true, C_est))

    t = np.array(t)

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
    # TRACE
    # ============================================================

    plt.figure(figsize=(7, 3))
    plt.plot(trace)
    plt.title("Loss trace")
    plt.grid()
    plt.tight_layout()
    plt.savefig(fig_dir / "trace.png", dpi=200)
    plt.close()

    # ============================================================
    # BOLD
    # ============================================================

    plt.figure(figsize=(10, 4))
    for i in range(min(3, Y_true.shape[1])):
        plt.plot(t, Y_true[:, i], label=f"true {i}")
        plt.plot(t, Y_est[:, i], "--", label=f"est {i}")

    plt.title("BOLD true vs est")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig(fig_dir / "bold.png", dpi=200)
    plt.close()

    # ============================================================
    # RESIDUALS
    # ============================================================

    residual = Y_obs - Y_est

    plt.figure(figsize=(10, 4))
    for i in range(residual.shape[1]):
        plt.plot(t, residual[:, i], label=f"res {i}")

    plt.title("Residuals")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig(fig_dir / "residuals.png", dpi=200)
    plt.close()

    # ============================================================
    # AUTOCORRELATION
    # ============================================================

    plt.figure(figsize=(6, 3))
    for i in range(min(3, residual.shape[1])):
        ac = autocorr(residual[:, i])
        plt.plot(ac, label=f"region {i}")

    plt.title("Residual autocorr")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig(fig_dir / "residual_autocorr.png", dpi=200)
    plt.close()

    # ============================================================
    # METRICS 
    # ============================================================

    # parameter error
    param_error = np.linalg.norm(theta_true - theta_est) / (
        np.linalg.norm(theta_true) + 1e-8
    )

    # RMSE
    rmse = np.sqrt(np.mean((Y_obs - Y_est) ** 2))

    # correlation
    corrs = [
        np.corrcoef(Y_obs[:, i], Y_est[:, i])[0, 1]
        for i in range(Y_obs.shape[1])
    ]
    mean_corr = float(np.mean(corrs))

    # stability (A)
    eigvals = np.linalg.eigvals(A_est)
    stable = bool(np.all(np.real(eigvals) < 0))

    # energy metrics
    input_energy = np.mean(U ** 2, axis=0)
    output_energy = np.mean(Y_est ** 2, axis=0)
    gain = (output_energy / (input_energy + 1e-8)).tolist()

    energy_ratio = float(
        np.mean(Y_est ** 2) / (np.mean(Y_true ** 2) + 1e-8)
    )

    # coupling residual-input
    residual_input_corr = [
        np.corrcoef(residual[:, i], U[:, min(i, U.shape[1]-1)])[0, 1]
        for i in range(residual.shape[1])
    ]

    # ============================================================
    # SAVE JSON (SINGLE SOURCE OF TRUTH)
    # ============================================================

    metrics = {
        "param_error": float(param_error),
        "rmse": float(rmse),
        "mean_correlation": float(mean_corr),
        "stable": stable,
        "eigvals_real": np.real(eigvals).tolist(),
        "gain": gain,
        "energy_ratio": energy_ratio,
        "residual_input_corr": residual_input_corr,
        "final_loss": float(trace[-1]) if len(trace) > 0 else None,
    }

    save_json(metrics, run_dir / "metrics.json")

    # ============================================================
    # DONE
    # ============================================================

    print("✔ Diagnostics saved in:", run_dir)