from .model import Model
from .regression import MultipleLinearRegression, RandomForestRegressor, Lasso
from .classification import (
    KNearestNeighbors, LogisticRegressionModel, RandomForestClassifier
)

REGRESSION_MODELS = [
    "multiple_linear_regression",
    "random_forest_regressor",
    "lasso"
]

CLASSIFICATION_MODELS = [
    "logistic_regression",
    "k_nearest_neighbours",
    "random_forest_classifier"
]


def get_model(model_name: str) -> Model:
    """Factory function to get a model by name."""
    lowercase_model_name = model_name.lower()

    if lowercase_model_name in REGRESSION_MODELS:
        match lowercase_model_name:
            case "multiple_linear_regression":
                return MultipleLinearRegression()
            case "k_nearest_neighbours":
                return RandomForestRegressor()
            case "random_forest_classifier":
                return Lasso()
            case _:
                raise NotImplementedError(f"We didn't implement {lowercase_model_name} in get_model yet, sorry")
    elif lowercase_model_name in CLASSIFICATION_MODELS:
        match lowercase_model_name:
            case "k_nearest_neighbours":
                return KNearestNeighbors()
            case "logistic_regression":
                return LogisticRegressionModel()
            case "random_forest_classifier":
                return RandomForestClassifier()
            case _:
                raise NotImplementedError(f"We didn't implement {lowercase_model_name} in get_model yet, sorry")
                
    else:
        raise ValueError("That don exis bruh.")
