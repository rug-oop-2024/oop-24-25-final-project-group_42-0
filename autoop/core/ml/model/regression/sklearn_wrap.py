from copy import deepcopy

import numpy as np
import sklearn.linear_model as linear
from pydantic import PrivateAttr

from autoop.core.ml.model.model import Model


class Lasso(Model):
    """
    A class that acts as a wrapper for the
    Lasso function from scikit-learn.linear_model.Lasso
    """

    _instance_of_lasso: linear.Lasso = PrivateAttr(default=linear.Lasso())

    @property
    def lasso(self):
        return deepcopy(self._instance_of_lasso)

    def fit(self, observations: np.ndarray, ground_truth: np.ndarray) -> None:
        """
        Uses the observations and ground truth to create the intercept and
        coefficient for prediction. saves the values in self._parameters
        """
        super().fit(observations, ground_truth)
        self._instance_of_lasso.fit(observations, ground_truth)
        self._parameters.update(
            {
                "_coef": self._instance_of_lasso.coef_,
                "_intercept": self._instance_of_lasso.intercept_,
            }
        )

    def predict(self, observations: np.ndarray) -> np.ndarray:
        """
        predicts the ground truth based on the observations,
        the intercept and the coefficient
        """
        super().predict(observations)
        return self._instance_of_lasso.predict(observations)
