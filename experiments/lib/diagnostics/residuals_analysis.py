import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def plot_mlp_norm(dz_res, run_dir):

    dz_res = np.array(dz_res)

    plt.figure()
    plt.plot(np.linalg.norm(dz_res, axis=1))
    plt.title("MLP residual norm over time")
    plt.grid()

    Path(run_dir, "figures").mkdir(exist_ok=True)
    plt.savefig(Path(run_dir) / "figures/mlp_norm.png")
    plt.close()