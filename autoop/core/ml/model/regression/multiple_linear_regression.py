""".

File: multiple_linear_regression.py
Authors: Oscar Schuyl(s5576954, o.schuyl@student.rug.nl)
         & Santos Bril (s5529875, s.l.bril@student.rug.nl)
Description: a file that implements;
a class for a multiple linear regression model.
"""

from copy import deepcopy

import numpy as np
from pydantic import PrivateAttr

from autoop.core.ml.model.model import Model


class MultipleLinearRegression(Model):
    """A model for multiple linear regression."""

    _slope: np.ndarray = PrivateAttr(default=np.ndarray)

    @property
    def get_slope(self) -> np.ndarray:
        return deepcopy(self._slope)

    def fit(self, observations: np.ndarray, ground_truth: np.ndarray) -> None:
        """
        Uses the observations and ground truth
        to create a slope for prediction.
        """
        super().fit(observations, ground_truth)
        self._observations = observations
        self._ground_truth = ground_truth
        observations_with_constants = (
            MultipleLinearRegression._add_row_of_ones(
                observations
            )
        )
        # reduce amount of math necessary as this happens twice
        transposed_observations = observations_with_constants.T

        first_part_equation = np.linalg.inv(
            transposed_observations @ observations_with_constants
        )
        second_part_equation = transposed_observations @ ground_truth
        self._slope = first_part_equation @ second_part_equation
        self._parameters.update({"slope": self._slope})

    def predict(self, observations: np.ndarray) -> np.ndarray:
        """
        Using the slope and observations to calculate
        and return the ground truth.
        """
        super().predict(observations)
        if type(self._slope) is not np.ndarray:
            raise TypeError(
                f"The variable self._slope is type {type(self._slope)},"
                + "self._slope should be np.ndarray."
            )
        if observations.shape[1] is not len(self._slope) - 1:
            raise ValueError(
                "Unexpected amount of columns in observations,"
                + f"expected{len(self._slope) - 1}"
                + f"got{observations.shape[1]}. "
            )

        observations_with_constants = (
            MultipleLinearRegression._add_row_of_ones(
                observations
            )
        )
        return observations_with_constants @ self._slope

    def _add_row_of_ones(matrix: np.ndarray) -> np.ndarray:
        """
        adds a row of ones to the matrix
        """
        if type(matrix) is not np.ndarray:
            raise TypeError(
                f"The variable matrix is type {type(matrix)},"
                + "matrix should be np.ndarray."
            )

        matrix_with_rows_of_one = np.append(
            matrix, np.array([[1] for _ in range(matrix.shape[0])]), axis=1
        )
        return matrix_with_rows_of_one
