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