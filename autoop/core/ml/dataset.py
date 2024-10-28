import io
from abc import ABC, abstractmethod

import pandas as pd

from autoop.core.ml.artifact import Artifact


class Dataset(Artifact):

    def __init__(self, *args, **kwargs):
        super().__init__(type="dataset", *args, **kwargs)

    @staticmethod
    def from_dataframe(data: pd.DataFrame, name: str, asset_path: str, version: str="1.0.0"):
        dataset_2 =  Dataset(
            name=name,
            asset_path=asset_path,
            data=data.to_csv(index=False).encode(),
            version=version,
        )
        return dataset_2

    def read(self) -> pd.DataFrame:
        bytes = super().read()
        csv = bytes.decode()
        # A str
        return pd.read_csv(io.StringIO(csv))

    def save(self, data: bytes) -> bytes:
        # bytes = data.to_csv(index=False).encode()
        return super().save(data)
