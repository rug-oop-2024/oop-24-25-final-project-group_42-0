from glob import glob
from pathlib import Path

import pandas as pd
import streamlit as st

from app.core.system import AutoMLSystem
from autoop.core.ml.dataset import Dataset

options = glob("**/*.csv", recursive=True)


path = st.selectbox("Select a dataset", options)
automl = AutoMLSystem.get_instance()
datasets = automl.registry.list(type="dataset")
data_path = Path(path)
# automl.registry.register()
# print(path)
# print(data_path)
st.write(f"datasets: {datasets}\noptions: {options}\n"
         + f"registry: {automl.registry.__dict__}")
st.write(f"\nautoml._storage: {automl._storage.__dict__}")

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
