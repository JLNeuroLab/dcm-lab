import numpy as np
import torch

from dcm.models.neuronal_bilinear import BilinearNeuronalModel, BilinearParameters
from dcm.models.torch.neuronal_torch import NeuronalBilinearTorch, BilinearParametersTorch
from dcm.simulation.numpy import simulate_neuronal as sim_np
from dcm.simulation.torch import simulate_neuronal as sim_torch


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
        model_torch = NeuronalBilinearTorch(
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

        t = np.linspace(0, 20, 200)

        z0 = np.array([1.0, -1.0, 0.5])

        Z_np = sim_np(model_np, u, t, z0=z0)
        Z_torch = sim_torch(model_torch, u_torch, t, z0=torch.tensor(z0)).detach().numpy()

        assert np.allclose(Z_np, Z_torch, atol=1e-5)

    def test_stability_decay_to_zero():

        l, m = 3, 2

        A = np.diag([-0.8, -0.6, -0.5])
        B = np.zeros((m, l, l))
        C = np.zeros((l, m))

        model = NeuronalBilinearTorch(
            BilinearParametersTorch(
                A=torch.tensor(A, dtype=torch.float32),
                B=torch.tensor(B, dtype=torch.float32),
                C=torch.tensor(C, dtype=torch.float32),
            )
        )

        def u(t):
            return torch.zeros(m)

        t = np.linspace(0, 10, 200)

        z0 = torch.tensor([1.0, -2.0, 0.5])

        Z = simulate_torch(model, u, t, z0=z0)

        assert torch.norm(Z[-1]) < 0.1 * torch.norm(Z[0])

        def test_gradient_flow():

        l, m = 3, 2

        A = torch.randn(l, l, requires_grad=True)
        B = torch.zeros(m, l, l, requires_grad=True)
        C = torch.zeros(l, m, requires_grad=True)

        model = NeuronalBilinearTorch(
            BilinearParametersTorch(A=A, B=B, C=C)
        )

        def u(t):
            return torch.ones(m)

        t = np.linspace(0, 5, 50)

        z0 = torch.zeros(l, requires_grad=False)

        Z = simulate_torch(model, u, t, z0=z0)

        loss = Z.pow(2).mean()

        loss.backward()

        assert A.grad is not None
        assert not torch.isnan(A.grad).any()
        assert torch.isfinite(A.grad).all()

    def test_long_rollout_numerical_stability():

        l, m = 3, 2

        A = np.diag([-0.5, -0.6, -0.7])
        B = np.zeros((m, l, l))
        C = np.zeros((l, m))

        model = NeuronalBilinearTorch(
            BilinearParametersTorch(
                A=torch.tensor(A, dtype=torch.float32),
                B=torch.tensor(B, dtype=torch.float32),
                C=torch.tensor(C, dtype=torch.float32),
            )
        )

        def u(t):
            return torch.sin(torch.tensor([t, t]))

        t = np.linspace(0, 200, 5000)

        z0 = torch.randn(l)

        Z = simulate_torch(model, u, t, z0=z0)

        assert torch.isfinite(Z).all()
        assert not torch.isnan(Z).any()