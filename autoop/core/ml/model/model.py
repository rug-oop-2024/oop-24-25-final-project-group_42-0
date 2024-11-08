
from abc import ABC, abstractmethod
from copy import deepcopy
from typing import Literal

import numpy as np
from pydantic import BaseModel, PrivateAttr

from autoop.core.ml.artifact import Artifact


class Model(ABC, BaseModel):
    """
    Base class for all metrics.
    input: ground truth and prediction
    return: a real number
    """

    _parameters: dict = PrivateAttr(default=dict())

    @property
    def parameters(self) -> dict:
        return deepcopy(self._parameters)

    @property
    def observations(self) -> np.ndarray:
        return deepcopy(self._parameters["observations"])

    @property
    def ground_truth(self) -> np.ndarray:
        return deepcopy(self._parameters["ground_truth"])

    @abstractmethod
    def fit(self, observations: np.ndarray, ground_truth: np.ndarray) -> None:
        """
        all children have these checks in common so they are done in the parent
        """
        if type(observations) is not np.ndarray:
            raise TypeError(
                "fit doesn't accept observations of type:"
                + f"{type(observations)}"
            )
        elif type(ground_truth) is not np.ndarray:
            raise TypeError(
                "fit doesn't accept observations of type:"
                + f"{type(ground_truth)}"
            )
        self._parameters = {
            "observations": observations,
            "ground_truth": ground_truth
        }

    @abstractmethod
    def predict(self, observations: np.ndarray) -> np.ndarray:
        """take an observations and return a prediction"""
        if type(observations) is not np.ndarray:
            raise TypeError(
                "predict doesn't accept observations of type:"
                + f"{type(observations)}"
            )
