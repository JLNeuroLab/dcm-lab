import numpy as np
import matplotlib.pyplot as plt
from  pathlib import Path


def plot_and_save_separate(
    run_dir: Path,
    t: np.ndarray,
    U: np.ndarray,
    z: np.ndarray,
    s: np.ndarray,
    f: np.ndarray,
    v: np.ndarray,
    q: np.ndarray,
    Y: np.ndarray,
    names: list[str] | None = None,
) -> None:
    """Save separate figures (inputs, neuronal, hemo, bold)."""
    fig_dir = run_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    m = U.shape[1]
    if names is None:
        names = [f"u{j}" for j in range(m)]

    # 1) Inputs
    plt.figure(figsize=(10, 3))
    for j in range(m):
        plt.plot(t, U[:, j], lw=2, label=names[j])
    plt.xlabel("time [s]")
    plt.ylabel("u(t)")
    plt.title("Inputs")
    plt.grid(alpha=0.3)
    plt.legend(ncol=min(m, 4))
    plt.tight_layout()
    plt.savefig(fig_dir / "inputs.png", dpi=200)
    plt.close()

    # 2) Neuronal
    plt.figure(figsize=(10, 3))
    for r in range(z.shape[1]):
        plt.plot(t, z[:, r], lw=2, label=f"z{r}")
    plt.xlabel("time [s]")
    plt.ylabel("z(t)")
    plt.title("Neuronal state")
    plt.grid(alpha=0.3)
    plt.legend(ncol=min(z.shape[1], 4))
    plt.tight_layout()
    plt.savefig(fig_dir / "neuronal.png", dpi=200)
    plt.close()

    # 3) Hemodynamics (region 0)
    r0 = 0
    plt.figure(figsize=(10, 3))
    plt.plot(t, s[:, r0], label="s")
    plt.plot(t, f[:, r0], label="f")
    plt.plot(t, v[:, r0], label="v")
    plt.plot(t, q[:, r0], label="q")
    plt.xlabel("time [s]")
    plt.ylabel("state")
    plt.title("Hemodynamic states (region 0)")
    plt.grid(alpha=0.3)
    plt.legend(ncol=4)
    plt.tight_layout()
    plt.savefig(fig_dir / "hemo_states_region0.png", dpi=200)
    plt.close()

    # 4) BOLD
    plt.figure(figsize=(10, 3))
    for r in range(Y.shape[1]):
        plt.plot(t, Y[:, r], lw=2, label=f"BOLD{r}")
    plt.xlabel("time [s]")
    plt.ylabel("BOLD")
    plt.title("BOLD output")
    plt.grid(alpha=0.3)
    plt.legend(ncol=min(Y.shape[1], 4))
    plt.tight_layout()
    plt.savefig(fig_dir / "bold.png", dpi=200)
    plt.close()


def plot_summary(
    run_dir: Path,
    t: np.ndarray,
    U: np.ndarray,
    z: np.ndarray,
    s: np.ndarray,
    f: np.ndarray,
    v: np.ndarray,
    q: np.ndarray,
    Y: np.ndarray,
    names: list[str] | None = None,
) -> None:
    """Save one stacked summary figure with inputs, neuronal, hemo (r0), bold."""
    fig_dir = run_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    m = U.shape[1]
    if names is None:
        names = [f"u{j}" for j in range(m)]

    fig, axes = plt.subplots(
        4, 1,
        figsize=(12, 10),
        sharex=True,
        gridspec_kw={"height_ratios": [1.2, 1.2, 1.5, 1.2]},
    )

    # Inputs
    for j in range(m):
        axes[0].plot(t, U[:, j], lw=2, label=names[j])
    axes[0].set_ylabel("u(t)")
    axes[0].set_title("DCM forward model sanity check (summary)")
    axes[0].grid(alpha=0.3)
    axes[0].legend(ncol=min(m, 4))

    # Neuronal
    for r in range(z.shape[1]):
        axes[1].plot(t, z[:, r], lw=2, label=f"z{r}")
    axes[1].set_ylabel("z(t)")
    axes[1].grid(alpha=0.3)
    axes[1].legend(ncol=min(z.shape[1], 4))

    # Hemodynamics (region 0)
    r0 = 0
    axes[2].plot(t, s[:, r0], label="s")
    axes[2].plot(t, f[:, r0], label="f")
    axes[2].plot(t, v[:, r0], label="v")
    axes[2].plot(t, q[:, r0], label="q")
    axes[2].set_ylabel("hemo (r0)")
    axes[2].grid(alpha=0.3)
    axes[2].legend(ncol=4)

    # BOLD
    for r in range(Y.shape[1]):
        axes[3].plot(t, Y[:, r], lw=2, label=f"BOLD{r}")
    axes[3].set_ylabel("BOLD")
    axes[3].set_xlabel("time [s]")
    axes[3].grid(alpha=0.3)
    axes[3].legend(ncol=min(Y.shape[1], 4))

    plt.tight_layout()
    plt.savefig(fig_dir / "summary.png", dpi=200)
    plt.close(fig)