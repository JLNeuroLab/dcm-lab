from __future__ import annotations
import numpy as np

from dcm.models.forward import ForwardModel, simulate_forward
from dcm.models.parametrization import NeuronalParameterization
from dcm.models.neuronal_bilinear import BilinearParameters

Array = np.ndarray


class ForwardAdapter:
    """
    Adapter mapping flat parameter vector θ to predicted BOLD signal.

    This allows inference algorithms to treat the full DCM simulator as
    a function:

        θ → ŷ(θ)

    Internally the adapter:
        1. Unpacks θ into structured neuronal parameters
        2. Updates the neuronal model parameters
        3. Runs the forward simulation
        4. Returns predicted BOLD
    """

    def __init__(
        self,
        forward_model: ForwardModel,
        parametrization: NeuronalParameterization,
        design,
        z0: Array | None = None,
        x0: Array | None = None,
    ):
        self.forward_model = forward_model
        self.parametrization = parametrization
        self.design = design

        self.z0 = z0
        self.x0 = x0

    # ------------------------------------------------------------------
    # Parameter handling
    # ------------------------------------------------------------------

    def unpack(self, theta: Array):
        """Convert flat θ into neuronal parameter matrices."""
        return self.parametrization.unpack(theta)

    def _update_model_parameters(self, theta: Array):
        """
        Update neuronal model parameters inside ForwardModel.
        """
        th = self.unpack(theta)

        new_params = BilinearParameters(
            A=th.A,
            B=th.B,
            C=th.C,
        )

        self.forward_model.neuronal.params = new_params

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def predict(self, theta: Array) -> Array:
        """
        Run forward DCM simulation.

        Returns
        -------
        Y_hat : (T, l)
            Predicted BOLD time series
        """

        # update neuronal parameters
        self._update_model_parameters(theta)

        u = self.design.callable()

        S, Y = simulate_forward(
            self.forward_model,
            u=u,
            t_eval=self.design.t,
            z0=self.z0,
            x0=self.x0,
        )

        return Y

    def __call__(self, theta: Array) -> Array:
        return self.predict(theta)