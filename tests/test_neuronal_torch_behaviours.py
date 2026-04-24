import numpy as np
import torch

from dcm.models.neuronal_bilinear import BilinearNeuronalModel, BilinearParameters
from dcm.torch.neuronal_torch import BilinearNeuronalTorch, BilinearParametersTorch
from dcm.simulate.adapters import neuronal_rhs_factory_torch, neuronal_rhs_factory
from dcm.simulate.integrators import rk4_integrate_torch, rk4_integrate


def test_numpy_torch_equivalence():

        l, m = 3, 2

        A = np.array([
            [-0.6,  0.2,  0.0],
            [ 0.1, -0.5,  0.3],
            [ 0.0,  0.2, -0.4],
        ])

        B = np.zeros((m, l, l))
        B[0, 1, 0] = 0.5
        B[1, 2, 1] = 0.4

        C = np.zeros((l, m))
        C[0, 0] = 1.0
        C[1, 1] = 0.8

        # numpy model
        model_np = BilinearNeuronalModel(BilinearParameters(A, B, C))

        # torch model
        model_torch = BilinearNeuronalTorch(
            BilinearParametersTorch(
                A=torch.tensor(A, dtype=torch.float32),
                B=torch.tensor(B, dtype=torch.float32),
                C=torch.tensor(C, dtype=torch.float32),
            )
        )

        def u(t):
            return np.array([1.0 if t > 5 else 0.0,
                            0.5 if t > 10 else 0.0])

        def u_torch(t):
            return torch.tensor(u(t), dtype=torch.float32)

        t = torch.linspace(0, 20, 200, dtype=torch.float32)
        z0 = torch.tensor([1.0, -1.0, 0.5], dtype=torch.float32)

        f_torch = neuronal_rhs_factory_torch(model=model_torch, input_fn=u_torch)
        Z_torch = rk4_integrate_torch(f=f_torch, t_eval=t, z0=z0)

        f_np = neuronal_rhs_factory(model=model_np, input_fn=u)
        Z_np = rk4_integrate(f=f_np, t_eval=t, z0=z0)
        assert np.allclose(Z_np, Z_torch, atol=1e-5)

def test_neuronal_torch_matches_numpy():

    l, m = 3, 2

    A = np.random.randn(l, l) * 0.1
    B = np.random.randn(m, l, l) * 0.1
    C = np.random.randn(l, m) * 0.1

    z = np.random.randn(l)
    u = np.random.randn(m)

    # NumPy
    np_model = BilinearNeuronalModel(BilinearParameters(A, B, C))
    dz_np = np_model.dynamics(0.0, z, u)

    # Torch
    torch_model = BilinearNeuronalTorch(
        BilinearParametersTorch(
            A=torch.tensor(A, dtype=torch.float32),
            B=torch.tensor(B, dtype=torch.float32),
            C=torch.tensor(C, dtype=torch.float32),
        )
    )

    dz_torch = torch_model.dynamics(
        0.0,
        torch.tensor(z, dtype=torch.float32),
        torch.tensor(u, dtype=torch.float32),
    ).detach().numpy()

    assert np.allclose(dz_np, dz_torch, atol=1e-5)

def test_neuronal_autograd():

    l, m = 3, 2

    A = torch.randn(l, l)
    B = torch.randn(m, l, l)
    C = torch.randn(l, m)
    model = BilinearNeuronalTorch(
        BilinearParametersTorch(A=A, B=B, C=C)
    )

    z = torch.randn(l, requires_grad=True)
    u = torch.randn(m)

    dz = model.dynamics(0.0, z, u)
    loss = dz.sum()
    loss.backward()

    assert z.grad is not None
    assert A.requires_grad is False
    assert B.requires_grad is False
    assert C.requires_grad is False

def test_neuronal_torch_finite():
    import torch
    import numpy as np

    l, m = 3, 2

    A = torch.randn(l, l) * 0.1
    B = torch.randn(m, l, l) * 0.1
    C = torch.randn(l, m) * 0.1

    model = BilinearNeuronalTorch(
        BilinearParametersTorch(A=A, B=B, C=C)
    )

    z = torch.randn(l)
    u = torch.randn(m)

    dz = model.dynamics(0.0, z, u)

    assert torch.all(torch.isfinite(dz))
    assert dz.shape == (l,)