from abc import ABC, abstractmethod
from copy import deepcopy
from typing import Any

import numpy as np
from pydantic import BaseModel, PrivateAttr

from autoop.core.ml.dataset import Dataset
from autoop.core.ml.model.model import Model
from autoop.core.ml.model.regression.multiple_linear_regression import (
    MultipleLinearRegression,
)
from typechecker import Raise_Type_Error, Type_Checker

METRICS = [
    "mean_squared_error",
    "mean_absolute_percentage_error",
    "root_mean_squared_error"
    "accuracy",
    "precision",
    "recall"
]


def get_metric(name: str) -> "Metric":
    # Factory function to get a metric by name.
    # Return a metric instance given its str name.
    lower_case_name = name.lower()
    if lower_case_name in METRICS:
        match lower_case_name:
            case "mean_squared_error":
                return MeanSquaredError()
            case "mean_absolute_percentage_error":
                return MeanAbsolutePercentageError()
            case "root_mean_squared_error":
                return RootMeanSquaredError()
            case "accuracy":
                return Accuracy()
            case "precision":
                return Precision()
            case "recall":
                return Recall()
            case _:
                raise NotImplementedError(f"We didn't implement {lower_case_name} in get_model yet, sorry")
    else:
        raise ValueError(f"{lower_case_name} not in METRICS.")


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

    def __call__(self, prediction: np.ndarray, ground_truth: np.ndarray):
        return self.evaluate(prediction, ground_truth)

    def evaluate(self, prediction: np.ndarray, ground_truth: np.ndarray):
        if not Type_Checker(prediction, np.ndarray):
            Raise_Type_Error(prediction, np.ndarray, "prediction")

        if not Type_Checker(ground_truth, np.ndarray):
            Raise_Type_Error(ground_truth, np.ndarray, "ground_truth")
        if len(ground_truth) != 1:
            raise ValueError("Ground truth should have only one column,"
                             + f"instead it has {len(prediction)} columns.")
        if len(prediction) != 1:
            raise ValueError("Prediction should have only one column,"
                             + f"instead it has {len(prediction)} columns.")


class Accuracy(Metric):
    _name: str = PrivateAttr("accuracy")

    def __call__(self):
        pass

    def evaluate(self, prediction: np.ndarray, ground_truth: np.ndarray) -> float:
        super().evaluate(prediction, ground_truth)
        the_same_amount = sum(ground_truth == prediction)
        return the_same_amount / len(prediction)


class Precision(Metric):

    _name: str = PrivateAttr("precision")

    def evaluate(self, prediction: np.ndarray, ground_truth: np.ndarray) -> float:
        super().evaluate(prediction, ground_truth)

        true_positive_dict = {}
        false_negative_dict = {}

        for i in range(len(prediction)):
            prediction_string = prediction[i]
            ground_truth_string = ground_truth[i]
            if not Type_Checker(prediction_string, str):
                Raise_Type_Error(ground_truth_string, str, "prediction[i]")

            if not Type_Checker(prediction_string, str):
                Raise_Type_Error(ground_truth_string, str, "ground_truth[i]")

            if prediction_string not in true_positive_dict.keys():
                true_positive_dict.update({prediction_string: 0})
                false_negative_dict.update({prediction_string: 0})

            if ground_truth_string not in false_negative_dict.keys():
                true_positive_dict.update({ground_truth_string: 0})
                false_negative_dict.update({ground_truth_string: 0})

            if prediction_string == ground_truth_string:
                true_positive_dict[prediction_string] += 1
            else:
                false_negative_dict[prediction_string] += 1

        answer = 0
        for key in true_positive_dict.keys():
            answer += (
                        true_positive_dict[key] / (
                            true_positive_dict[key] + false_negative_dict[key]
                        )
                       )

        return answer / len(true_positive_dict)


class Recall(Metric):

    _name: str = PrivateAttr("recall")

    def evaluate(self, prediction: np.ndarray, ground_truth: np.ndarray) -> float:
        super().evaluate(prediction, ground_truth)

        true_positive_dict = {}
        false_positive_dict = {}

        for i in range(len(prediction)):
            prediction_string = prediction[i]
            ground_truth_string = ground_truth[i]
            if not Type_Checker(prediction_string, str):
                Raise_Type_Error(ground_truth_string, str, "prediction[i]")

            if not Type_Checker(prediction_string, str):
                Raise_Type_Error(ground_truth_string, str, "ground_truth[i]")

            if prediction_string not in true_positive_dict.keys():
                true_positive_dict.update({prediction_string: 0})
                false_positive_dict.update({prediction_string: 0})

            if ground_truth_string not in false_positive_dict.keys():
                true_positive_dict.update({ground_truth_string: 0})
                false_positive_dict.update({ground_truth_string: 0})

            if prediction_string == ground_truth_string:
                true_positive_dict[prediction_string] += 1
            else:
                false_positive_dict[ground_truth_string] += 1

        answer = 0
        for key in true_positive_dict.keys():
            answer += (
                        true_positive_dict[key] / (
                            true_positive_dict[key] + false_positive_dict[key]
                        )
                      )

        return answer / len(true_positive_dict)


class MeanSquaredError(Metric):

    _name: str = PrivateAttr("mean_squared_error")

    def __call__(self):
        pass

    def evaluate(self, prediction: np.ndarray, ground_truth: np.ndarray) -> float:
        super().evaluate(prediction, ground_truth)
        error = np.subtract(ground_truth, prediction)
        squared_error = np.square(error)
        total_error = np.sum(squared_error)

        return total_error / len(prediction)


class RootMeanSquaredError(MeanSquaredError):
    _name: str = PrivateAttr("root_mean_squared_error")

    def evaluate(self, prediction: np.ndarray, ground_truth: np.ndarray) -> float:
        return np.sqrt(super().evaluate(prediction, ground_truth))


class MeanAbsolutePercentageError(Metric):
    _name: str = PrivateAttr("mean_absolute_percentage_error")

    def evaluate(self, prediction: np.ndarray, ground_truth: np.ndarray) -> float:
        super().evaluate(prediction, ground_truth)
        abs_error = np.abs(
           np.divide(np.subtract(ground_truth, prediction), ground_truth)
        )
        answer = (np.sum(abs_error) / len(ground_truth)) * 100
        return answer
