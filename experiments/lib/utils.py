import numpy as np
import torch
import matplotlib.pyplot as plt
from scipy.signal import correlate

# ============================================================
# IMPORT DESIGN
# ============================================================

from dcm.simulate.design import (
    make_time_grid as make_time_grid_np,
    boxcar as boxcar_np,
    events as events_np,
    stack_inputs as stack_np,
    InputDesign as InputDesign,
)

from dcm.simulate.design_torch import (
    make_time_grid as make_time_grid_torch,
    boxcar as boxcar_torch,
    events as events_torch,
    stack_inputs as stack_torch,
    InputDesignTorch,
)

# ============================================================
# MODELS (NUMPY)
# ============================================================

from dcm.models.neuronal_bilinear import (
    BilinearParameters,
    BilinearNeuronalModel,
)

from dcm.models.hemodynamic_balloon import (
    HemodynamicParameters,
    HemodynamicBalloonModel,
)

from dcm.models.forward import ForwardModel


# ============================================================
# MODELS (TORCH)
# ============================================================

from dcm.torch.neuronal_torch import (
    BilinearNeuronalTorch,
    BilinearParametersTorch,
)

from dcm.torch.hemodynamic_torch import (
    HemodynamicBalloonTorch,
    HemodynamicParametersTorch,
)

from dcm.torch.forward_torch import ForwardModelTorch


def _to_np(x):
    return np.array(x)


def _normalize(M):
    return M / (np.max(np.abs(M)) + 1e-8)


def _plot_matrix(ax, M, title):
    im = ax.imshow(M, cmap="coolwarm", aspect="auto", vmin=-1, vmax=1)
    ax.set_title(title)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)


def autocorr(x):
    x = x - np.mean(x)
    ac = correlate(x, x, mode="full")
    ac = ac[len(ac)//2:]
    return ac / (ac[0] + 1e-8)
# ============================================================
# DESIGN BUILDERS
# ============================================================

def build_design_numpy(cfg: dict):
    T = float(cfg["simulation"]["T"])
    dt = float(cfg["simulation"]["dt"])

    t = make_time_grid_np(T, dt)

    names = cfg["inputs"]["names"]
    regs = []

    for name in names:
        spec = cfg["inputs"][name]
        typ = spec["type"]

        if typ == "boxcar":
            regs.append(
                boxcar_np(
                    t,
                    spec["onsets"],
                    spec["durations"],
                    spec.get("amplitudes", 1.0),
                )
            )

        elif typ == "events":
            regs.append(
                events_np(
                    t,
                    spec["onsets"],
                    spec.get("amplitudes", 1.0),
                    spec.get("mode", "nearest"),
                )
            )

        else:
            raise ValueError(f"Unknown input type: {typ}")

    U = stack_np(*regs)
    return InputDesign(t=t, U=U, names=names)


def build_design_torch(cfg: dict, device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    T = float(cfg["simulation"]["T"])
    dt = float(cfg["simulation"]["dt"])

    t = make_time_grid_torch(T, dt, device=device)

    names = cfg["inputs"]["names"]
    regs = []

    for name in names:
        spec = cfg["inputs"][name]
        typ = spec["type"]

        if typ == "boxcar":
            regs.append(
                boxcar_torch(
                    t,
                    spec["onsets"],
                    spec["durations"],
                    spec.get("amplitudes", 1.0),
                    device=device,
                )
            )

        elif typ == "events":
            regs.append(
                events_torch(
                    t,
                    spec["onsets"],
                    spec.get("amplitudes", 1.0),
                    spec.get("mode", "nearest"),
                    device=device,
                )
            )

        else:
            raise ValueError(f"Unknown input type: {typ}")

    U = stack_torch(*regs, device=device)
    return InputDesignTorch(t=t, U=U, names=names)


# ============================================================
# MODEL BUILDERS
# ============================================================

def build_model_numpy(cfg: dict, param_key="neuronal"):
    l = int(cfg["model"]["l"])
    m = int(cfg["model"]["m"])

    A = np.asarray(cfg[param_key]["A"], dtype=float)
    B = np.asarray(cfg[param_key]["B"], dtype=float)
    C = np.asarray(cfg[param_key]["C"], dtype=float)

    params = BilinearParameters(A=A, B=B, C=C)

    if params.l != l or params.m != m:
        raise ValueError("Model dimension mismatch")

    neuronal = BilinearNeuronalModel(params)

    hemo = HemodynamicBalloonModel(
        HemodynamicParameters.with_defaults(l)
    )

    return ForwardModel(neuronal, hemo)


def build_model_torch(cfg: dict, param_key="neuronal", device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    l = int(cfg["model"]["l"])
    m = int(cfg["model"]["m"])

    A = torch.tensor(cfg[param_key]["A"], dtype=torch.float32, device=device)
    B = torch.tensor(cfg[param_key]["B"], dtype=torch.float32, device=device)
    C = torch.tensor(cfg[param_key]["C"], dtype=torch.float32, device=device)

    params = BilinearParametersTorch(A=A, B=B, C=C)

    if params.l != l or params.m != m:
        raise ValueError("Model dimension mismatch")

    neuronal = BilinearNeuronalTorch(params).to(device)

    hemo = HemodynamicBalloonTorch(
        HemodynamicParametersTorch.with_defaults(l, device=device)
    ).to(device)

    return ForwardModelTorch(neuronal, hemo).to(device)