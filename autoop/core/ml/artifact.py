import csv
import os
import re


class Artifact():

    """
    "asset_path": "users/mo-assaf/models/yolov8.pth",
    "version": "1.0.2",
    "data": b"binary_state_data",
    "metadata": {
        "experiment_id": "exp-123fbdiashdb",
        "run_id": "run-12378yufdh89afd",
    },
    "type": "model:torch",
    "tags": ["computer_vision", "object_detection"]
    """

    # type: str = Field(default_factory=str)
    def __init__(self,
                 type: str, name: str, data, asset_path: str, tags: list = [],
                 metadata: dict = {}, version: str = "1.0.0"):
        self._name = name
        self._type = type
        self._asset_path = asset_path
        self._data = data
        self._tags = tags
        self._metadata = metadata
        self._version = version

        self.save(self._data)

    def save(self, bytes: bytes) -> bytes:
        """
        saves the dataset on the datasets folder
        If datasets doesn't exist yet will create the folder
        """
        if not os.path.exists("./datasets"):
            os.makedirs("./datasets")

        data = bytes.decode().split("\r")

        with open("./datasets/" + self._asset_path, "w") as file:
            file.writelines(data)
            # for line in data:
            #     csv_file.writerow([line])
            # file.write(encoded_stuff, indent=4)
        return bytes

    @staticmethod
    def static_save(asset_path: str, bytes: bytes) -> bytes:
        """
        saves the dataset on the datasets folder
        If datasets doesn't exist yet will create the folder
        """
        if not os.path.exists("./datasets"):
            os.makedirs("./datasets")

        data = bytes.decode().split("\r")

        with open("./datasets/" + asset_path, "w") as file:
            file.writelines(data)
            # for line in data:
            #     csv_file.writerow([line])
            # file.write(encoded_stuff, indent=4)
        return bytes

    def read(self) -> str:
        """reads the data from the asset_path directory"""
        if not os.path.exists("./datasets/"):
            raise FileNotFoundError("dataset directory not found")
        if not os.path.exists("./datasets/" + self._asset_path):
            raise FileNotFoundError(f"{self._asset_path} file not found")
        try:
            with open("./datasets/" + self._asset_path, "r") as file:
                return file.read()
        except ValueError:
            raise ValueError("couldn't import from save,"
                             + " the file might be corrupted")

    @staticmethod
    def static_read(asset_path: str) -> str:
        """reads the data from the asset_path directory"""
        if not os.path.exists("./datasets/"):
            raise FileNotFoundError("dataset directory not found")
        if not os.path.exists(asset_path):
            raise FileNotFoundError(f"{asset_path} file not found")
        try:
            with open(asset_path, "r") as file:
                return file.read()
        except ValueError:
            raise ValueError("couldn't import from save,"
                             + " the file might be corrupted")
