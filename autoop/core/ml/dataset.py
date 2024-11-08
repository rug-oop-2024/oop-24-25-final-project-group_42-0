import io
import pandas as pd
from autoop.core.ml.artifact import Artifact


class Dataset(Artifact):

    def __init__(self, *args, **kwargs):
        super().__init__(type="dataset", *args, **kwargs)

    @staticmethod
    def from_dataframe(data: pd.DataFrame, name: str,
                       asset_path: str, version: str = "1.0.0"):
        dataset_2 = Dataset(
            name=name,
            asset_path=asset_path,
            data=data.to_csv(index=False).encode(),
            version=version,
        )
        return dataset_2

    def read(self) -> pd.DataFrame:
        csv = super().read()
        return pd.read_csv(io.StringIO(csv))

    def save(self, data: bytes) -> bytes:
        # bytes = data.to_csv(index=False).encode()
        return super().save(data)

    @staticmethod
    def static_read(asset_path: str) -> pd.DataFrame:
        csv = Artifact().static_read(asset_path)
        return pd.read_csv(io.StringIO(csv))

    @staticmethod
    def static_save(asset_path: str, data: bytes) -> bytes:
        # bytes = data.to_csv(index=False).encode()
        return super().static_save(asset_path, data)
