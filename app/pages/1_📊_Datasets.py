import io
import os
from glob import glob
from pathlib import Path

import pandas as pd
import streamlit as st

from app.core.system import AutoMLSystem
from autoop.core.ml.artifact import Artifact
from autoop.core.ml.dataset import Dataset


class Management():

    def create(self, full_asset_path: str) -> Dataset:
        # csv = Artifact.static_read("./datasets/adult.csv")
        data_path = Path(full_asset_path)
        df = pd.read_csv(data_path)

        name_and_asset_path = full_asset_path.split("\\")[-1:][0]
        dataset = Dataset.from_dataframe(
            name=name_and_asset_path,
            asset_path=name_and_asset_path,
            data=df,
        )
        self.save(dataset)
        return dataset

    def delete(self, artifact: Artifact):
        st.write("file deleted")
        automl.registry.delete(artifact.id)
        os.remove("./assets/dbo/artifacts/" + artifact.id)
        st.rerun()

    def save(self, artifact: Artifact):
        instance_of_automl = AutoMLSystem.get_instance()
        instance_of_automl.registry.register(artifact)


automl = AutoMLSystem.get_instance()

management = Management()

options = glob("**/*.csv", recursive=True)

path = st.selectbox("Select a dataset", options)
uploaded_file = st.file_uploader("Or upload a .csv file")

if uploaded_file is not None and uploaded_file.type == "text/csv":
    df = pd.read_csv(io.StringIO(uploaded_file.getvalue().decode()))
    dataset = Dataset.from_dataframe(df, uploaded_file.name, uploaded_file.name)
    management.save(dataset)
    # uploaded_file.get_value() is the data in byte format
    # .name is "adult.csv"
    # .type is "text/csv"
# data_path = Path(path)
# test = Dataset(data_path)

if st.button("delete file") and path is not None:
    management.delete(management.create(path))

if path is not None:
    dataset = management.create(path)

datasets = automl.registry.list(type="dataset")
st.dataframe(dataset.read())

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
