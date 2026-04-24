import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


# ============================================================
# UTILS
# ============================================================

def _to_np(x):
    return np.array(x)


def _normalize(M):
    return M / (np.max(np.abs(M)) + 1e-8)


def _plot_matrix(ax, M, title):
    im = ax.imshow(M, cmap="coolwarm", aspect="auto", vmin=-1, vmax=1)
    ax.set_title(title)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)


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

        A_true,
        A_est,
        B_true,
        B_est,
        C_true,
        C_est,
):
    run_dir = Path(run_dir)
    fig_dir = run_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    A_true = _to_np(A_true)
    A_est = _to_np(A_est)

    B_true = _to_np(B_true)
    B_est = _to_np(B_est)

    C_true = _to_np(C_true)
    C_est = _to_np(C_est)

    # ============================================================
    # MATRIX FIGURE
    # ============================================================

    fig, axes = plt.subplots(3, 2, figsize=(12, 10))

    # ---------------- A ----------------
    _plot_matrix(axes[0, 0], _normalize(A_true), "A true (norm)")
    _plot_matrix(axes[0, 1], _normalize(A_est), "A estimated (norm)")

    # ---------------- B (mean over inputs) ----------------
    B_true_mean = np.mean(B_true, axis=0)
    B_est_mean = np.mean(B_est, axis=0)

    _plot_matrix(axes[1, 0], _normalize(B_true_mean), "B true (mean, norm)")
    _plot_matrix(axes[1, 1], _normalize(B_est_mean), "B estimated (mean, norm)")

    # ---------------- C ----------------
    _plot_matrix(axes[2, 0], _normalize(C_true), "C true (norm)")
    _plot_matrix(axes[2, 1], _normalize(C_est), "C estimated (norm)")

    plt.tight_layout()
    plt.savefig(fig_dir / "matrices.png", dpi=200)
    plt.close()

    # ============================================================
    # TRACE
    # ============================================================

    plt.figure(figsize=(7, 3))
    plt.plot(trace)
    plt.title("MAP optimization trace")
    plt.xlabel("iteration")
    plt.ylabel("loss")
    plt.grid()
    plt.tight_layout()
    plt.savefig(fig_dir / "trace.png", dpi=200)
    plt.close()

    # ============================================================
    # TIME SERIES (optional but useful)
    # ============================================================

    t = np.array(t)

    plt.figure(figsize=(10, 4))
    for i in range(min(3, Y_true.shape[1])):
        plt.plot(t, Y_true[:, i], label=f"true {i}")
        plt.plot(t, Y_est[:, i], "--", label=f"est {i}")
    plt.title("BOLD: true vs estimated")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig(fig_dir / "bold.png", dpi=200)
    plt.close()

    # ============================================================
    # PARAMETER ERROR SUMMARY
    # ============================================================

    error = np.linalg.norm(theta_true - theta_est) / (np.linalg.norm(theta_true) + 1e-8)

    with open(run_dir / "summary.txt", "w") as f:
        f.write(f"Relative parameter error: {error:.6f}\n")
        f.write(f"Final loss: {trace[-1] if len(trace) > 0 else None}\n")

    print("✔ Diagnostics saved in:", run_dir)