import numpy as np
import pytest

from dcm.models.neuronal_bilinear import BilinearNeuronalModel, BilinearParameters
from dcm.models.hemodynamic_balloon import HemodynamicBalloonModel, HemodynamicParameters
from dcm.models.forward import ForwardModel
from dcm.models.parametrization import NeuronalParameterization
from dcm.inference.forward_adapter import ForwardAdapter
from dcm.simulate.design import make_time_grid, boxcar, InputDesign

def make_test_adapter():
     # small toy system
    l = 2
    m = 1

    # connectivity matrices
    A = np.array([
        [-0.5, 0.2],
        [0.1, -0.4],
    ])

    B = np.zeros((m, l, l))

    C = np.array([
        [1.0],
        [0.0],
    ])

    neuronal = BilinearNeuronalModel(
        BilinearParameters(A, B, C)
    )

    hemodynamic = HemodynamicBalloonModel(
        HemodynamicParameters.with_defaults(l)
    )
    model = ForwardModel(
        neuronal_model=neuronal,
        hemodynamic_model=hemodynamic
    )

    t = make_time_grid(T=10, dt=0.05)
    u = boxcar(
        t=t,
        onsets=10,
        durations=20,
    )

    design = InputDesign(t=t, U=u)

    param = NeuronalParameterization(l, m)

    adapter = ForwardAdapter(
        forward_model=model,
        parametrization=param,
        design=design,
    )

    return adapter

def test_forward_adapter_output_shape():

    adapter = make_test_adapter()

    theta = np.random.randn(adapter.parametrization.n_params)

    y = adapter(theta)

    assert y.ndim == 2
    assert y.shape[0] == adapter.design.time.size

def test_forward_adapter_deterministic():
    adapter = make_test_adapter()

    theta = np.random.randn(adapter.parametrization.n_params)

    y1 = adapter(theta)
    y2 = adapter(theta)

    assert np.allclose(y1, y2)

def test_forward_adapter_custom_initial_state():
    adapter = make_test_adapter()
    theta = np.random.randn(adapter.parametrization.n_params)

    z0 = np.ones(adapter.forward_model.l)
    x0 = np.zeros(4 * adapter.forward_model.l)

    adapter.z0 = z0
    adapter.x0 = x0

    y = adapter(theta)
    assert y.shape[0] == adapter.design.t.size

def test_forward_adapter_different_design():
    adapter = make_test_adapter()
    theta = np.random.randn(adapter.parametrization.n_params)

    # modifico la griglia temporale
    t_new = np.linspace(0, 30, 301)
    adapter.design = InputDesign(t=t_new, U=np.zeros((301, 1)))

    y = adapter(theta)
    assert y.shape[0] == 301

def test_forward_adapter_invalid_theta():
    adapter = make_test_adapter()
    theta = np.random.randn(adapter.parametrization.n_params + 1) 
    with pytest.raises(ValueError):
        adapter(theta)