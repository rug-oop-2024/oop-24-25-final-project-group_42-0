"""
file: k_nearest_neighbors.py
Authors: Oscar Schuyl(s5576954, o.schuyl@student.rug.nl)
         & Santos Bril (s5529875, s.l.bril@student.rug.nl)
Description: a file for the k nearest neighbour;
        check neigbours for their values and turn yourself into said value
"""
from collections import Counter

import numpy as np
from pydantic import Field

from autoop.core.ml.model.model import ClassificationModel

AMOUNT_OF_NEIGHBOURS: int = 3


class KNearestNeighbors(ClassificationModel):
    """K-Nearest Neighbors Algorithm"""

    k: int = Field(title="Number of neighbors", default=AMOUNT_OF_NEIGHBOURS)

    def _amount_of_neighbours_greator_than_zero(self):
        if self.k <= 0:
            raise ValueError("amount_of_neighbours must be greater than 0")

    def fit(self, observations: np.ndarray, ground_truth: np.ndarray) -> None:
        super().fit(observations, ground_truth)
        self._parameters.update(
            {
                "amount_of_neigbours": self.k
            }
        )

    def predict(self, observations: np.ndarray) -> np.ndarray:
        """
        predicts the ground truths of individual observations and
        returns the predictions
        """
        super().predict(observations)
        self._amount_of_neighbours_greator_than_zero()
        predictions = [self._predict_single(x) for x in observations]
        return predictions

    def _predict_single(self, observation) -> str:
        """
        calculates distance between observation and every other point and
        returns the most common ground truth
        """
        distances = np.linalg.norm(
            self._parameters["observations"] - observation, axis=1
        )
        # step 2: sort the array of the distances and take first k
        neighbour_indices = np.argsort(distances)[: self.k]

        # make sure to turn the arrays into strings,
        #  so that Counter can parse it
        neighbour_nearest_labels = [
            self._parameters["ground_truth"][i].tolist()[0]
            for i in neighbour_indices
        ]

        # now we have k = 3, 3 labels inside an array
        # step4: take most common label and return it to the caller
        most_common = Counter(neighbour_nearest_labels).most_common()
        return most_common[0][0]
