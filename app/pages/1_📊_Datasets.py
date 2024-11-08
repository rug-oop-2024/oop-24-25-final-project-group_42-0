from glob import glob
from pathlib import Path

import pandas as pd
import streamlit as st
import io

from app.core.system import AutoMLSystem
from autoop.core.ml.dataset import Dataset
from autoop.core.ml.artifact import Artifact

class Management():

    def create(self, name: str, asset_path: str):
        # csv = Artifact.static_read("./datasets/adult.csv")
        with open("./datasets/adult.csv", "r") as file:
                csv = file.read()
        df = pd.read_csv(io.StringIO(csv))

        dataset = Dataset.from_dataframe(
            name = name,
            asset_path = "adult.csv",
            data = df,
        )
        st.write(dataset.version)
        # automl.registry.register(dataset)

options = glob("**/*.csv", recursive=True)


path = st.selectbox("Select a dataset", options)
automl = AutoMLSystem.get_instance()

testing_something = Management()

testing_something.create("test", path)

datasets = automl.registry.list(type="dataset")

st.write(f"datasets: {datasets}")

"""
docstring format
Args:
    arguments
Returns:
    dictionary of metrics
"""

# Choose columns to plot
# columns = st.multiselect("Select columns to plot", dataset.columns)
# fig = None
# print(columns)
# if len(columns) == 0:
#     st.write("Please select at least one column to plot.")
# if len(columns) == 1:
#     fig = plotter.hist_1d(columns[0])
# if len(columns) == 2:
#     fig = plotter.scatter_2d(columns[0], columns[1])
# if len(columns) == 3:
#     fig = plotter.scatter_3d(columns[0], columns[1], columns[2])
# if len(columns) > 3:
#     st.error("Please select at most 3 columns to plot.")
# if fig:
#     st.pyplot(fig)


# your code here
