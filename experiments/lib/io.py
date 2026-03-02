from __future__ import annotations

from pathlib import Path
import datetime as dt
import json
import yaml
import numpy as np


def load_yaml(path: str | Path) -> dict:
    path = Path(path)
    with path.open("r") as f:
        return yaml.safe_load(f)


def save_yaml(cfg: dict, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)


def save_json(obj: dict, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(obj, f, indent=2)


def make_run_dir(experiment: str, root: str | Path = "results/runs") -> Path:
    """
    Create a unique run folder:
      results/runs/YYYY-mm-dd_HH-MM-SS_experiment/
        figures/
    """
    root = Path(root)
    ts = dt.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = root / f"{ts}_{experiment}"
    (run_dir / "figures").mkdir(parents=True, exist_ok=False)
    return run_dir


def save_npz(path: str | Path, **arrays) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)