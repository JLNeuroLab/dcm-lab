import numpy as np
import pytest

from dcm.models.neuronal_bilinear import BilinearNeuronalModel, BilinearParameters, simulate_neuronal
from dcm.models.hemodynamic_balloon import HemodynamicParameters, HemodynamicBalloonModel, simulate_hemodynamic
from dcm.models.forward import ForwardModel, simulate_forward

@pytest.fixture
def dcm_model():
    l, m = 1, 1
    A = np.array([[-0.5]])
    B = np.zeros((m, l, l))
    C = np.array([[1.0]])

    neuronal_params = BilinearParameters(A, B, C)
    hemodynamic_params = HemodynamicParameters.with_defaults(l)

    neuronal_model = BilinearNeuronalModel(neuronal_params)
    hemodynamic_model = HemodynamicBalloonModel(hemodynamic_params)

    model = ForwardModel(neuronal_model, hemodynamic_model)
    t_eval = np.linspace(0.0, 60, 601)

    return model, t_eval, l, m

def test_zero_input_behaviour(dcm_model):

    model, t_eval, l, m = dcm_model

    def u_zero(t):
        return np.zeros(m, dtype=float)
    
    S, Y = simulate_forward(model, u=u_zero, t_eval=t_eval)

    # Neuronal state should remain near zero
    assert np.allclose(S[:, :l], 0.0, atol=1e-8), "neuronal states should be zero, got {}"
    # BOLD should remain near zero
    assert np.allclose(Y, 0.0, atol=1e-8)

def test_step_input_behaviour_strict(dcm_model):
    model, t_eval, l, m = dcm_model

    def u_step(t):
        return np.array([1.0 if 10.0 <= t <= 20.0 else 0.0])

    S, Y = simulate_forward(model, u=u_step, t_eval=t_eval)

    # Neuronal activity rises
    neuronal_max = np.max(S[:, :l])
    assert neuronal_max > 0.0, "Neuronal activity did not respond"

    # BOLD response follows the trend
    neuronal_diff = np.diff(S[:, :l], axis=0)
    bold_diff = np.diff(Y, axis=0)

    # When neuronal rises, BOLD mostly rises too
    rising_period = neuronal_diff > 0
    assert np.all(bold_diff[rising_period] >= -1e-8), "BOLD should rise when neuronal rises"

    # When neuronal falls, BOLD mostly falls too
    falling_period = neuronal_diff < 0
    assert np.all(bold_diff[falling_period] <= 1e-8), "BOLD should fall when neuronal falls"


def test_step_input_behaviour_relaxed(dcm_model):
    model, t_eval, l, m = dcm_model

    def u_step(t):
        return np.array([1.0 if 10.0 <= t <= 20.0 else 0.0])

    S, Y = simulate_forward(model, u=u_step, t_eval=t_eval)

    # Calculate maximum neuronal activity over all time points
    neuronal_max = np.max(S[:, :l])
    assert neuronal_max > 0.0, "Neuronal activity did not respond"

    # Compare BOLD at neuronal start vs end of rising phase
    # Get the index corresponding to the time point where neuronal activity starts rising
    # In this case when it reaches 1% of its maximum value previously calculated
    rise_start_idx = np.where(S[:, :l] > 0.01 * neuronal_max)[0][0]
    rise_end_idx = np.argmax(S[:, :l])

    bold_start = Y[rise_start_idx, 0]
    bold_end = Y[rise_end_idx, 0]

    print("BOLD start:", bold_start, "BOLD end:", bold_end)
    assert bold_end > bold_start, "BOLD should increase overall during neuronal rise"