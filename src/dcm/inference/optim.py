import numpy as np
from scipy.optimize import minimize
from dcm.inference.objectives import gaussian_log_posterior

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
    trace = []
    cache = {} if cache_eval else None

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

    result = minimize(
        fun=negative_log_posterior,
        x0=theta0,
        method=method,
        options={"maxiter": 500, "disp": verbose}
    )

    return result, np.array(trace)

import torch

def map_estimation_torch(
        model,
        theta,
        n_steps: int = 50,
        lr: float = 1e-2,
        method: str = "lbfgs",
        verbose: bool = True,
    ):

    trace = []
    if method.lower() == "lbfgs":
        
        optimizer = torch.optim.LBFGS(
            [theta],
            lr=lr,
            max_iter=n_steps,
            line_search_fn="strong_wolfe"
        )

        def closure():
            optimizer.zero_grad()

            loss = -model(theta)

            loss.backward()

            trace.append(loss.item())

            if verbose:
                print(f"[LBFGS] loss: {loss.item():.6f}")

        optimizer.step(closure)

        return theta, torch.tensor(trace)

    elif method.lower() == "adam":
        
        optimizer = torch.optim.Adam([theta], lr=lr)

        for i in range(n_steps):

            optimizer.zero_grad()

            loss = -model(theta)

            loss.backward()

            optimizer.step()

            trace.append(loss.item())

            if verbose:
                print(f"[ADAM] step {i} loss: {loss.item():.6f}")

        return theta, torch.tensor(trace)

    else:
        raise ValueError(f"Unknown method: {method}")

        

