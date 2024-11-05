from copy import deepcopy

import numpy as np
import sklearn.ensemble as ensemble
from pydantic import PrivateAttr

from autoop.core.ml.model.model import Model


class RandomForestRegressor(Model):
    """
    A class that acts as a wrapper for the
    Lasso function from scikit-learn.linear_model.Lasso
    """

    _instance_of_random_forest_regressor: ensemble.RandomForestRegressor = PrivateAttr(default=ensemble.RandomForestRegressor)

    @property
    def random_forest_regressor(self):
        return deepcopy(self._instance_of_random_forest_regressor)

    def fit(self, observations: np.ndarray, ground_truth: np.ndarray) -> None:
        """
        Uses the observations and ground truth to create the intercept and
        coefficient for prediction. saves the values in self._parameters
        """
        super().fit(observations, ground_truth)
        self._instance_of_random_forest_regressor.fit(observations, ground_truth)
        self._parameters.update(
            {
                "_coef": self._instance_of_random_forest_regressor.coef_,
                "_intercept": self._instance_of_random_forest_regressor.intercept_,
            }
        )

    def predict(self, observations: np.ndarray) -> np.ndarray:
        """
        predicts the ground truth based on the observations,
        the intercept and the coefficient
        """
        super().predict(observations)
        return self._instance_of_random_forest_regressor.predict(observations)
