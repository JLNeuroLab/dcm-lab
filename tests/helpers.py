import numpy as np

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
        onsets=[10],
        durations=[20],
    )

    design = InputDesign(t=t, U=u)

    param = NeuronalParameterization(l, m)

    adapter = ForwardAdapter(
        forward_model=model,
        parametrization=param,
        design=design,
    )

    return adapter