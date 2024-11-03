from abc import ABC, abstractmethod
from typing import Any
import numpy as np
from typechecker import Type_Checker
from copy import deepcopy
from pydantic import BaseModel, PrivateAttr
from model import Model 

METRICS = [
    "mean_squared_error",
    "accuracy",
] # add the names (in strings) of the metrics you implement

def get_metric(name: str) -> "Metric":
    # Factory function to get a metric by name.
    # Return a metric instance given its str name.
    if name in METRICS:
        return "IDK"
    
    
    raise ValueError(f"{name} not in METRICS.")


class Metric(Model):
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
        
    def __str__(self):
        return f"self._parameters:\n{self._parameters}"

    def __call__(self,name :str):
        match name:
            case "mean_squared_error":
                self._mean_squared_error()
            case "accuracy":
                pass
            case _:
                raise NotImplementedError("To be implemented.")
            
    def _mean_squared_error():
        pass

    
# class Meansqerror(Metric):

#     _name: str = PrivateAttr("mean_squared_error")

#     def __call__():
#         raise NotImplementedError("To be implemented.")

#     def fit(self, observations: np.ndarray, ground_truth: np.ndarray) -> None:
#         return super().fit(observations, ground_truth)

#     def predict(self, observations: np.ndarray) -> np.ndarray:
#         return super().predict(observations)
    
# asd =  Meansqerror
# asd(argument)