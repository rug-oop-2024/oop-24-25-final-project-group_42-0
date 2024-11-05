import streamlit as st
import pandas as pd
from app.core.system import AutoMLSystem
from autoop.core.ml.dataset import Dataset
from pathlib import Path
from glob import glob


options = glob("**/*.csv", recursive=True)
path = st.selectbox("Select a dataset", options)
automl = AutoMLSystem.get_instance()
datasets = automl.registry.list(type="dataset")
data_path = Path(path)
dataset = Dataset.from_dataframe(data = Dataset.read(), name = "", asset_path = data_path)
df = dataset.data
print(datasets, data_path, dataset)
st.write(df.head())

# Choose columns to plot
columns = st.multiselect("Select columns to plot", dataset.columns)
fig = None
print(columns)
if len(columns) == 0:
    st.write("Please select at least one column to plot.")
# if len(columns) == 1:
#     fig = plotter.hist_1d(columns[0])
# if len(columns) == 2:
#     fig = plotter.scatter_2d(columns[0], columns[1])
# if len(columns) == 3:
#     fig = plotter.scatter_3d(columns[0], columns[1], columns[2])
if len(columns) > 3:
    st.error("Please select at most 3 columns to plot.")
if fig:
    st.pyplot(fig)


# your code here
