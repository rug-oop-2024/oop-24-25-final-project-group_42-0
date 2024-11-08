# import unittest
# from autoop.tests.test_database import TestDatabase
# from autoop.tests.test_features import TestFeatures
# from autoop.tests.test_pipeline import TestPipeline
# from autoop.tests.test_storage import TestStorage

# if __name__ == "__main__":
#     unittest.main()

# import streamlit as st
# import app.Welcome

from sklearn.datasets import fetch_openml, load_iris

from autoop.core.ml.metric import Accuracy

accuracy_test = Accuracy()

dataset = load_iris()
dataset2 = fetch_openml(name="adult", version=1, parser="auto")

# So I can splice "iris" but not "adult"? okay?

print("iris (continuous) Accuracy:"
      + f"{accuracy_test.evaluate(dataset.data[:, 0], dataset.data[:, 1])}")
print("adult (categorical) Accuracy:"
      + f"{accuracy_test.evaluate(dataset2.data[:, 0], dataset2.data[:, 1])}")

# Metrics we can make ig
# Root Mean Squared Error (RMSE):
# This is the square root of the Mean Squared Error
# and is often used alongside MSE to interpret error
# in the same units as the original data, making it more interpretable.

# Mean Absolute Percentage Error (MAPE): This metric measures
# the average magnitude of errors as a percentage of actual values,
# making it useful for understanding error in terms of relative scale.

# Logarithmic Loss (Log Loss):
# This metric evaluates the uncertainty of the predictions,
# especially useful for probabilistic classification models.
# Lower values indicate better performance.
# “Log Loss = −1/n * sum i=1 to n
# (yi * log(yi predict) + (1 - yi) * log(1 - yi predict))
