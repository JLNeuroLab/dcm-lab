import numpy as np
import pytest

from .helpers import make_test_adapter
from dcm.simulate.design import InputDesign

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