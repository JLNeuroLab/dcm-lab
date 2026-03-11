from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from dcm.models.forward import simulate_forward
from dcm.models.parametrization import NeuronalParameterization, NeuronalTheta
from dcm.inference.forward_adapter import ForwardAdapter
from dcm.inference.objectives import gaussian_log_posterior
from dcm.inference.optim import map_estimation

from experiments.lib.io import load_yaml, save_yaml, make_run_dir, save_npz, save_json
from experiments.run_forward import build_design, build_model

# Helper function to build NeuronalTheta with initial parameter values
def theta_from_config(cfg_section):

    A = np.asarray(cfg_section["A"], dtype=float)
    B = np.asarray(cfg_section["B"], dtype=float)
    C = np.asarray(cfg_section["C"], dtype=float)

    return NeuronalTheta(A=A, B=B, C=C)

def simulate_data(cfg, model, design):
    """
    Simulate dataset using true parameters and add Gaussian noise.
    """
    u = design.callable()

    S, Y = simulate_forward(
        model=model,
        u=u,
        t_eval=design.t
    )

    noise_std = cfg["noise"]["std"]
    Y_obs = Y + noise_std * np.random.randn(*Y.shape)

    return S, Y, Y_obs

def main(config_path: str):

    cfg = load_yaml(config_path)
    run_dir = make_run_dir(cfg["name"])
    save_yaml(cfg, run_dir / "config.yaml")

    model_true = build_model(cfg, param_key="true_parameters")
    design = build_design(cfg)

    # --- Simulate dataset ---
    S, Y_true, Y_obs = simulate_data(cfg, model_true, design)

    model_inference = build_model(cfg, "initialization")
    parametrization = NeuronalParameterization(
        l=model_inference.l,
        m=model_inference.neuronal.params.m
    )

    adapter = ForwardAdapter(
        forward_model=model_inference,
        parametrization=parametrization,
        design=design,
    )
    # --- Pack initial guess ---
    th0 = theta_from_config(cfg["initialization"])
    theta0 = parametrization.pack(th0)

    # --- Pack priors ---
    mu_theta = parametrization.pack(theta_from_config(cfg["priors"]["mu"]))

    sigma_cfg = cfg["priors"]["sigma"]
    sigma_prior = np.concatenate([
        np.full(parametrization.l*parametrization.l, sigma_cfg["A"]),
        np.full(parametrization.m*parametrization.l*parametrization.l, sigma_cfg["B"]),
        np.full(parametrization.l*parametrization.m, sigma_cfg["C"])
    ])

    result, trace = map_estimation(
        theta0=theta0,
        y_obs=Y_obs,
        adapter=adapter,
        sigma=cfg["noise"]["std"],
        mu=mu_theta,
        sigma_prior=sigma_prior,
        method=cfg["optimizer"]["method"]
    )

    theta_est = result.x

    # --- Save results ---
    save_npz(
        run_dir / "results.npz",
        Y_true=Y_true,
        Y_obs=Y_obs,
        theta_est=theta_est,
        trace=trace
    )

    summary = {
        "success": result.success,
        "message": result.message,
        "n_iterations": result.nit,
        "final_loss": result.fun,
        "theta_est_shape": theta_est.shape,
    }
    save_json(summary, run_dir / "summary.json")

    print("Inversion run saved to:", run_dir)
    print("Optimizer message:", result.message)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="experiments/configs/inversion_2r_feedforward.yaml"
    )
    args = parser.parse_args()
    main(args.config)


