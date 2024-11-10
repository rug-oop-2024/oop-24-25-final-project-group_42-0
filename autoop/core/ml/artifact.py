import base64
import os
from copy import deepcopy


class Artifact():

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
        encoded = base64.b64encode(asset_path.encode("utf-8"))
        encoded_string = encoded.decode("utf-8")
        self._id = f"{encoded_string}={version}"

        self.save(self._data)

    @property
    def name(self):
        return deepcopy(self._name)

    @property
    def type(self):
        return deepcopy(self._type)

    @property
    def asset_path(self):
        return deepcopy(self._asset_path)

    @property
    def data(self):
        return deepcopy(self._data)

    @property
    def tags(self):
        return deepcopy(self._tags)

    @property
    def metadata(self):
        return deepcopy(self._metadata)

    @property
    def version(self):
        return deepcopy(self._version)

    @property
    def id(self):
        return deepcopy(self._id)

    def save(self, bytes: bytes) -> None:
        """
        Saves the dataset in the datasets folder.
        If datasets doesn't exist yet will create the folder.
        Args:
            bytes[bytes]: Bytes of the dataset that need to be saved.
        Returns:
            None
        """

        if not os.path.exists("./assets/objects/"):
            os.makedirs("./assets/objects/", exist_ok=True)

        data = bytes.decode().split("\r")

        with open("./assets/objects/" + self._asset_path, "w") as file:
            file.writelines(data)

    def remove(self) -> None:
        """
        Removes the artifact.
        Args:
            None
        Returns:
            None
        """

        if os.path.exists("./assets/objects/" + self.asset_path):
            os.remove("./assets/objects/" + self.asset_path)

    def read(self) -> str:
        """
        Reads the data from the asset_path directory
        Args:
            None
        Returns:
            str
        """

        if not os.path.exists("./assets/objects/"):
            raise FileNotFoundError("assets directory not found")
        if not os.path.exists("./assets/objects/" + self._asset_path):
            raise FileNotFoundError(f"{self._asset_path} file not found")
        try:
            with open("./assets/objects/" + self._asset_path, "r") as file:
                return file.read()
        except ValueError:
            raise ValueError("couldn't import from save,"
                             + " the file might be corrupted")
