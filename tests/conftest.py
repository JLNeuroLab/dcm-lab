import pytest
import numpy as np
from dcm.inference.forward_adapter import ForwardAdapter
from .helpers import make_test_adapter

@pytest.fixture
def adapter():
    """Return a small ForwardAdapter for testing."""
    return make_test_adapter()

@pytest.fixture
def synthetic_data(adapter, theta_init):
    return adapter(theta_init)

@pytest.fixture
def dummy_data(adapter):
    T = adapter.design.t.size
    l = adapter.forward_model.l
    return np.zeros((T, l)) # simple zeros, easy to test

@pytest.fixture
def theta_init(adapter):
    """Return initial theta vector for testing."""
    return np.zeros(adapter.parametrization.n_params)