import torch
from dataclasses import dataclass
from typing import Sequence, Optional, Callable

Tensor = torch.Tensor


# ============================================================
# WRAPPERS TORCH (DIRECT)
# ============================================================

def make_time_grid(T: float, dt: float, device=None) -> Tensor:
    n = int(round(T / dt))
    t = torch.arange(n + 1, device=device, dtype=torch.float32) * dt
    return t


def boxcar(t, onsets, durations, amplitudes=1.0, device=None) -> Tensor:
    t = torch.as_tensor(t, device=device)

    onsets = torch.as_tensor(onsets, device=device)
    durations = torch.as_tensor(durations, device=device)

    if isinstance(amplitudes, (int, float)):
        amplitudes = torch.full((len(onsets),), float(amplitudes), device=device)
    else:
        amplitudes = torch.as_tensor(amplitudes, device=device)

    u = torch.zeros_like(t)

    for onset, dur, amp in zip(onsets, durations, amplitudes):
        mask = (t >= onset) & (t < onset + dur)
        u = u + mask.float() * amp

    return u


def events(t, onsets, amplitudes=1.0, mode="nearest", device=None) -> Tensor:
    t = torch.as_tensor(t, device=device)
    onsets = torch.as_tensor(onsets, device=device)

    # ----------------------------
    # amplitudes handling (FIX)
    # ----------------------------
    if isinstance(amplitudes, (int, float)):
        amplitudes = torch.full((len(onsets),), float(amplitudes), device=device)
    else:
        amplitudes = torch.as_tensor(amplitudes, device=device)

    u = torch.zeros_like(t)

    # ----------------------------
    # event loop
    # ----------------------------
    for onset, amp in zip(onsets, amplitudes):

        if mode == "nearest":
            idx = torch.argmin(torch.abs(t - onset))

        elif mode == "floor":
            idx = torch.searchsorted(t, onset, right=True) - 1
            idx = torch.clamp(idx, 0, len(t) - 1)

        else:
            raise ValueError("mode must be 'nearest' or 'floor'")

        u[idx] += amp

    return u

def stack_inputs(*u_list, device=None) -> Tensor:
    u_list = [torch.as_tensor(u, device=device).ravel() for u in u_list]
    return torch.stack(u_list, dim=1)


# ============================================================
# INPUT DESIGN TORCH
# ============================================================

@dataclass(frozen=True)
class InputDesignTorch:
    t: torch.Tensor
    U: torch.Tensor
    names: Optional[Sequence[str]] = None

    @property
    def m(self) -> int:
        return self.U.shape[1]

    @property
    def T(self) -> int:
        return self.U.shape[0]

    # ------------------------------------------------------------
    # ZOH CALLABLE
    # ------------------------------------------------------------

    def callable(self) -> Callable[[torch.Tensor], torch.Tensor]:
        t = self.t
        U = self.U
        dt = t[1] - t[0]

        def u(ts):
            ts = torch.as_tensor(ts, device=U.device)

            idx = ((ts - t[0]) / dt).long()
            idx = torch.clamp(idx, 0, self.T - 1)

            return U[idx]

        return u


# ============================================================
# TEST BLOCK
# ============================================================

if __name__ == "__main__":
    import matplotlib.pyplot as plt

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("device:", device)

    # ============================================================
    # TIME GRID
    # ============================================================

    t = make_time_grid(T=100.0, dt=0.1, device=device)

    print("t shape:", t.shape)
    print("t range:", t[0].item(), "->", t[-1].item())

    # ============================================================
    # INPUTS
    # ============================================================

    u0 = boxcar(t, [10.0], [20.0], 1.0, device=device)
    u1 = boxcar(t, [40.0], [30.0], 1.0, device=device)

    e0 = events(
        t,
        onsets=[5.0, 50.0, 80.0],
        amplitudes=[1.0, 0.5, 1.0],
        mode="nearest",
        device=device
    )

    # ============================================================
    # STACK
    # ============================================================

    U = stack_inputs(u0, u1, e0, device=device)

    design = InputDesignTorch(t=t, U=U, names=["u0", "u1", "e0"])

    print("\nU shape:", design.U.shape)
    print("m:", design.m)

    # ============================================================
    # CALLABLE TEST
    # ============================================================

    u = design.callable()

    test_times = torch.tensor(
        [0.0, 9.9, 10.0, 10.1, 30.0, 40.0, 50.0, 80.0, 100.0],
        device=device
    )

    print("\nCallable checks:")
    for ts in test_times:
        print(ts.item(), "->", u(ts))

    # ============================================================
    # PLOT
    # ============================================================

    t_np = t.cpu().numpy()
    U_np = U.cpu().numpy()

    import matplotlib.pyplot as plt

    plt.figure(figsize=(10, 3))
    plt.plot(t_np, U_np[:, 0], label="u0")
    plt.plot(t_np, U_np[:, 1], label="u1")
    plt.plot(t_np, U_np[:, 2], label="events")

    plt.title("design_torch sanity check")
    plt.legend()
    plt.grid(alpha=0.3)

    plt.tight_layout()
    plt.show()