from abc import ABC, abstractmethod
from typing import Any
import numpy as np
from typechecker import Type_Checker, Raise_Type_Error
from copy import deepcopy
from pydantic import BaseModel, PrivateAttr
from autoop.core.ml.model.model import Model
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

    # _metrics: list[Model] = PrivateAttr(default=list())

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
            
    def evaluate(self, prediction: np.ndarray, ground_truth: np.ndarray):
        if not Type_Checker(prediction, np.ndarray):
            Raise_Type_Error(prediction, np.ndarray, "prediction")

        if not Type_Checker(ground_truth, np.ndarray):
            Raise_Type_Error(ground_truth, np.ndarray, "Y")

class Accuracy(Metric):
    _name: str = PrivateAttr("accuracy")
    #_MLR: MultipleLinearRegression = PrivateAttr(default=MultipleLinearRegression())

    def __call__(self):
        pass

    def evaluate(self, prediction: np.ndarray, ground_truth: np.ndarray) -> float:
        super().evaluate(prediction, ground_truth)
        the_same_amount = sum(ground_truth==prediction)
        return the_same_amount / len(prediction)


class MeanSquaredError(Metric):

    _name: str = PrivateAttr("mean_squared_error")
    #_MLR: MultipleLinearRegression = PrivateAttr(default=MultipleLinearRegression())

    def __call__(self):
        pass

    def evaluate(self, prediction: np.ndarray, ground_truth: np.ndarray) -> float:
        super().evaluate(prediction, ground_truth)
        error = np.subtract(ground_truth, prediction) #ndarray
        squared_error = np.square(error)
        total_error = np.sum(squared_error)

        return total_error / len(prediction)
    
class RootMeanSquaredError(MeanSquaredError):

    def evaluate(self, prediction: np.ndarray, ground_truth: np.ndarray) -> float:
        return np.sqrt(super().evaluate(prediction, ground_truth))
    
class MeanAbsolutePercentageError(Metric):
    
    def evaluate(self, prediction: np.ndarray, ground_truth: np.ndarray) -> float:
        super().evaluate(prediction, ground_truth)
        abs_error = np.abs(np.divide(np.subtract(ground_truth, prediction), ground_truth))
        answer  = (np.sum(abs_error) / len(ground_truth)) * 100
        return answer
    
class clas():
    pass

# Cohen’s Kappa: This metric measures the agreement between two raters
# (or a model and ground truth) accounting for the possibility of agreement occurring by chance.
# It’s useful for evaluating classification tasks where chance agreement is possible.