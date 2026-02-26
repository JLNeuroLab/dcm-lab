import numpy as np
import pytest

from dcm.models.neuronal_bilinear import BilinearNeuronalModel, BilinearParameters, simulate_neuronal

def _stable_A(l: int, rate: float = 0.5) -> np.ndarray:
    """Simple stable A with negative diagonal."""
    return -rate * np.eye(l, dtype=float)

def test_zero_input_behaviour_decays_to_zero():
    """
    Zero input: u(t)=0 => z_dot = A z.
    With stable A and nonzero z0, z(t) should decay towards 0.
    """
    l, m = 3, 2

    A = _stable_A(l, rate=0.8)
    B = np.zeros((m, l, l), dtype=float)
    C = np.zeros((l, m), dtype=float)

    params = BilinearParameters(A, B, C)
    model = BilinearNeuronalModel(params)

    def u(t: float) -> np.ndarray:
        return np.zeros((m,), dtype=float)
    
    t = np.linspace(0.0, 10.0, 501)
    z0 = np.array([1.0, -2.0, 0.5], dtype=float)

    Z = simulate_neuronal(
        model=model,
        u=u,
        t_eval=t,
        z0=z0
    )
    # Norms should decay towards the end
    n0 = np.linalg.norm(Z[0])
    nT = np.linalg.norm(Z[-1])

    assert nT < 0.1 * n0
    print("Zero input behavior test passed")

def test_pure_driving_input():
    l, m = 3, 1

    A = _stable_A(l, rate=0.8)        # simple stable decay
    B = np.zeros((m, l, l))     # no modulatory coupling

    C = np.zeros((l, m))
    C[0, 0] = 1.0               # only region 0 is driven

    params = BilinearParameters(A=A, B=B, C=C)
    model = BilinearNeuronalModel(params)

    def u_step(t: float):
        return np.array([1.0 if t > 5.0 else 0.0])

    t = np.linspace(0.0, 40.0, 400)
    Z = simulate_neuronal(model, u_step, t)

    # Region 0 should activate
    assert np.max(Z[:, 0]) > 0.1

    # Other regions should remain near zero
    assert np.allclose(Z[:, 1:], 0.0, atol=1e-4)

    print("Pure driving input test passed")

def test_modulatory_coupling():

    l, m = 2, 1

    A = np.array([
        [-0.5,  0.2],
        [ 0.0, -0.5],
    ])
    B = np.zeros((m, l, l), dtype=float)
    B[0, 1, 0] = 1.0  # modulation increases 0→1 coupling

    C = np.zeros((l, m), dtype=float)
    C[0, 0] = 1.0  # drive region 0

    params = BilinearParameters(A, B, C)
    model = BilinearNeuronalModel(params)

    def u_off(t: float):
        return np.array([0.0])
    
    def u_on(t: float):
        return np.array([1.0])
    
    t = np.linspace(0.0, 30.0, 300)

    Z_off = simulate_neuronal(model, u_off, t)
    Z_on = simulate_neuronal(model, u_on, t)

    peak_off = np.max(Z_off[:, 1])
    peak_on = np.max(Z_on[:, 1])

    assert peak_off < peak_on