import numpy as np
import pytest

from dcm.models.hemodynamic_balloon import (
    HemodynamicParameters,
    HemodynamicBalloonModel,
)
from dcm.simulate.adapters import hemodynamic_rhs_factory
from dcm.simulate.integrators import euler_integrate


# ------------------------------------------------------------------
# Helper functions to isolate test features
# ------------------------------------------------------------------

def baseline_state(l: int) -> np.ndarray:
    """x = [s,f,v,q] flattened; baseline is s=0, f=1, v=1, q=1."""
    s0 = np.zeros(l)
    f0 = np.ones(l)
    v0 = np.ones(l)
    q0 = np.ones(l)
    return np.concatenate([s0, f0, v0, q0])


def unpack_for_tests(x: np.ndarray, l: int):
    """Mirror your model.unpack, but kept local for tests."""
    return x[:l], x[l:2*l], x[2*l:3*l], x[3*l:4*l]


# ------------------------------------------------------------------
# Pytest fixtures
# ------------------------------------------------------------------

@pytest.fixture(scope="module")
def l():
    """Default number of regions for most unit tests."""
    return 3


@pytest.fixture(scope="module")
def params(l):
    """
    Default parameters used across tests.
    Keep tests stable: we set V0 explicitly here.
    """
    p = HemodynamicParameters.with_defaults(l)
    return HemodynamicParameters(
        l=p.l,
        kappa=p.kappa,
        gamma=p.gamma,
        tau=p.tau,
        alpha=p.alpha,
        rho=p.rho,
        V0=0.02,
    )


@pytest.fixture
def model(params):
    """Fresh model instance per test (safe if you later add state)."""
    return HemodynamicBalloonModel(params)


@pytest.fixture
def x0(l):
    """Baseline hemodynamic state vector."""
    return baseline_state(l)


# ------------------------------------------------------------------
# TESTS
# ------------------------------------------------------------------

def test_baseline_is_fixed_point_and_bold_zero(model, x0, l):

    z0 = np.zeros(l)

    f = hemodynamic_rhs_factory(model, lambda t: z0)

    xdot = f(0.0, x0)

    assert np.allclose(xdot, 0.0, atol=1e-12)

    y0 = model.bold(x0)
    assert y0.shape == (l,)
    assert np.allclose(y0, 0.0, atol=1e-12)


def test_invariant_f_dot_equals_s(model, l):

    rng = np.random.default_rng(0)

    s = rng.normal(0, 0.1, size=l)
    f = np.clip(1.0 + rng.normal(0, 0.1, size=l), 0.2, None)
    v = np.clip(1.0 + rng.normal(0, 0.1, size=l), 0.2, None)
    q = np.clip(1.0 + rng.normal(0, 0.1, size=l), 0.2, None)

    x = np.concatenate([s, f, v, q])
    z_t = rng.normal(0, 0.1, size=l)

    f_rhs = hemodynamic_rhs_factory(model, lambda t: z_t)

    xdot = f_rhs(0.0, x)

    s_dot, f_dot, v_dot, q_dot = unpack_for_tests(xdot, l)

    assert np.allclose(f_dot, s, atol=1e-12), "Expected f_dot == s exactly"


def test_positive_drive_increases_signal_derivative_at_baseline(model, x0, l):

    z_t = np.full(l, 0.1)

    f = hemodynamic_rhs_factory(model, lambda t: z_t)

    xdot = f(0.0, x0)

    s_dot, f_dot, v_dot, q_dot = unpack_for_tests(xdot, l)

    assert np.all(s_dot > 0.0)


def test_brief_pulse_gives_lagged_bold_peak():

    l = 1
    p1 = HemodynamicParameters.with_defaults(l)
    p1 = HemodynamicParameters(
        l=1,
        kappa=p1.kappa,
        gamma=p1.gamma,
        tau=p1.tau,
        alpha=p1.alpha,
        rho=p1.rho,
        V0=0.02,
    )
    model = HemodynamicBalloonModel(p1)

    t = np.linspace(0.0, 40.0, 401)
    x0 = baseline_state(1)

    def z_fn(tt: float):
        return np.array([1.0 if 2.0 <= tt < 3.0 else 0.0], dtype=float)

    f = hemodynamic_rhs_factory(model, z_fn)

    X = euler_integrate(f, t, x0)
    Y = np.array([model.bold(Xk)[0] for Xk in X])

    t_peak = t[int(np.argmax(Y))]

    assert 4.0 < t_peak < 10.0
    assert abs(Y[-1]) < abs(Y.max()) * 0.5


def test_simulation_stays_finite_and_positive():

    l = 1
    p = HemodynamicParameters.with_defaults(l)
    model = HemodynamicBalloonModel(p)

    def z(t: float):
        return np.array([0.5 if 0 <= t < 20 else 0.0])

    t = np.linspace(0.0, 60.0, 601)

    f = hemodynamic_rhs_factory(model, z)
    X = euler_integrate(f, t, baseline_state(1))

    Y = np.array([model.bold(x) for x in X])

    assert np.all(np.isfinite(X))
    assert np.all(np.isfinite(Y))

    s, f_, v, q = X[:, 0], X[:, 1], X[:, 2], X[:, 3]

    assert np.min(f_) > 0.0
    assert np.min(v) > 0.0