"""
O'Hara-Rudy 2011 Human Ventricular Myocyte Model

Digital-Twin-of-heart
"""

from dataclasses import dataclass
import numpy as np


@dataclass
class ORdParameters:
    """Parameters used by the O'Hara-Rudy ventricular model."""

    # Membrane properties
    Cm: float = 1.0  # membrane capacitance (uF/cm^2)

    # Temperature
    temperature: float = 310.0  # Kelvin

    # Gas constant
    R: float = 8314.0  # J/(kmol*K)

    # Faraday constant
    F: float = 96485.0  # C/mol


class ORdModel:
    """
    O'Hara-Rudy 2011 human ventricular myocyte model.

    The model will be built incrementally:
        1. Parameters
        2. State variables
        3. Ionic currents
        4. Calcium handling
        5. Differential equations
    """

    def __init__(self, parameters=None):
        self.parameters = parameters or ORdParameters()

        # Model state will be added here as implementation progresses.
        self.state = None

    def derivatives(self, t, state):
        """
        Calculate derivatives of all ORd state variables.

        Parameters
        ----------
        t : float
            Simulation time.
        state : numpy.ndarray
            Current model state.

        Returns
        -------
        numpy.ndarray
            Time derivatives of the model state.
        """

        raise NotImplementedError(
            "ORd equations have not been implemented yet."
        )