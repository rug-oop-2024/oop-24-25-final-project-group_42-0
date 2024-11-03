
from abc import ABC, abstractmethod
from autoop.core.ml.artifact import Artifact
import numpy as np
from copy import deepcopy
from typing import Literal
from pydantic import BaseModel, PrivateAttr

    
class Model(ABC, BaseModel):
    """
    Base class for all metrics.
    """
    # your code here
    # remember: metrics take ground truth and prediction as input and return a real number

    # add here concrete implementations of the Metric class
    _parameters: dict = PrivateAttr(default=dict())

    @property
    def parameters(self) -> dict:
        return deepcopy(self._parameters)

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