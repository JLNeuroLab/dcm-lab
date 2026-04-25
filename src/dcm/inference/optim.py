import numpy as np
from scipy.optimize import minimize
from dcm.inference.objectives import gaussian_log_posterior

# ================================================================
# NUMPY MAP OPTIMIZATION (SCIPY)
# ================================================================

def map_estimation(
        theta0,
        y_obs,
        adapter,
        sigma,
        mu,
        sigma_prior,
        method="L-BFGS-B",
        verbose=False,
        cache_eval=True
):
    """
    MAP estimation using SciPy optimizer (NumPy version).

    Optimizes the negative log-posterior using black-box optimization.
    Includes optional caching to avoid repeated forward evaluations.
    """
    trace = []
    cache = {} if cache_eval else None

    # ------------------------------------------------------------
    # Build objective function
    # ------------------------------------------------------------
    def negative_log_posterior(theta):
        if cache_eval:
            key = tuple(np.round(theta, 8))
            if key in cache:
                val = cache[key]
            else:
                val = -gaussian_log_posterior(theta, y_obs, adapter, sigma, mu, sigma_prior)
                cache[key] = val
        else:
            val = -gaussian_log_posterior(theta, y_obs, adapter, sigma, mu, sigma_prior)

        trace.append(val)
        if verbose:
            print(f"Eval {len(trace)}: {val:.4f}")
        return val

    # ------------------------------------------------------------
    # Run optimization
    # ------------------------------------------------------------
    result = minimize(
        fun=negative_log_posterior,
        x0=theta0,
        method=method,
        options={"maxiter": 500, "disp": verbose}
    )

    return result, np.array(trace)

# ================================================================
# PYTORCH MAP OPTIMIZATION (DIFFERENTIABLE)
# ================================================================

import torch

def map_estimation_torch(
        model,
        theta,
        n_steps: int = 50,
        lr: float = 1e-2,
        method: str = "lbfgs",
        verbose: bool = True,
    ):
    """
    MAP estimation using PyTorch optimizers.

    The model is a differentiable objective returning log-posterior.
    Supports LBFGS (second-order) and Adam (first-order).
    """

    trace = []
    theta_trace = []
    # ------------------------------------------------------------
    # LBFGS (requires closure)
    # ------------------------------------------------------------
    if method.lower() == "lbfgs":
        
        optimizer = torch.optim.LBFGS(
            [theta],
            lr=lr,
            max_iter=n_steps,
            line_search_fn="strong_wolfe"
        )

        counter = {"i": 0}

        def closure():
            optimizer.zero_grad()

            loss = -model(theta)

            loss.backward()

            trace.append(loss.item())
            theta_trace.append(theta.detach().cpu().clone())

            counter["i"] += 1 
            
            if verbose and counter["i"] % 5 == 0:
                grad_norm = theta.grad.norm().item() if theta.grad is not None else 0.0
                print(f"[LBFGS {counter['i']}] loss={loss.item():.6f} | grad_norm={grad_norm:.6f}")

            return loss

        optimizer.step(closure)

        return theta, torch.tensor(trace), theta_trace

    # ------------------------------------------------------------
    # Adam (standard gradient descent)
    # ------------------------------------------------------------
    elif method.lower() == "adam":

        optimizer = torch.optim.Adam([theta], lr=lr)

        for i in range(n_steps):

            optimizer.zero_grad()

            loss = -model(theta)

            loss.backward()

            optimizer.step()

            trace.append(loss.item())
            theta_trace.append(theta.detach().cpu().clone())

            if verbose and i % 10 == 0:
                grad_norm = theta.grad.norm().item() if theta.grad is not None else 0.0
                print(f"[ADAM {i}] loss={loss.item():.6f} | grad_norm={grad_norm:.6f}")


        return theta, torch.tensor(trace), theta_trace

    else:
        raise ValueError(f"Unknown method: {method}")

        

