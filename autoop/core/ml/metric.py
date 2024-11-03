from abc import ABC, abstractmethod
from typing import Any
import numpy as np
from typechecker import Type_Checker
from copy import deepcopy
from pydantic import BaseModel, PrivateAttr
from model import Model
from autoop.core.ml.model.regression.multiple_linear_regression import MultipleLinearRegression
from autoop.core.ml.dataset import Dataset

METRICS = [
    "mean_squared_error",
    "accuracy",
] # add the names (in strings) of the metrics you implement

def get_metric(name: str) -> "Metric":
    # Factory function to get a metric by name.
    # Return a metric instance given its str name.
    if name in METRICS:
        return Metric()
    raise ValueError(f"{name} not in METRICS.")
    


class Metric(Model):
    """
    Base class for all metrics.
    """

    _metrics: list[Model] = PrivateAttr(default=list())

    def fit(self, observations: np.ndarray, ground_truth: np.ndarray) -> None:
        super().fit(observations, ground_truth)
        

    def predict(self, observations: np.ndarray) -> np.ndarray:
        super().predict(observations)
        
    def __str__(self):
        return f"self._parameters:\n{self._parameters}"

    def __call__(self,name :str, data: Dataset):
        match name:
            case "mean_squared_error":
                self._mean_squared_error()
            case "accuracy":
                pass
            case _:
                raise NotImplementedError("This metric is not implemented.")
            
    def _mean_squared_error(self):
        if asd not in self._metrics:
            self._metrics.append(asd)
        asd = MultipleLinearRegression()
        asd.fit(self._parameters, )
        
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