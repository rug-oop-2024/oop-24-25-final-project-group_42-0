import csv, re
import os


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
    
    name=data["name"],
    version=data["version"],
    asset_path=data["asset_path"],
    tags=data["tags"],
    metadata=data["metadata"],
    data=self._storage.load(data["asset_path"]),
    type=data["type"],
    """

    # type: str = Field(default_factory=str)
    def __init__(self, *args, **kwargs):
        if "type" in kwargs.keys():
            self.type = kwargs["type"]

        for key, value in kwargs.items():
            if key == "name":
                self.name = value
            elif key == "type":
                self.type = value
            elif key == "asset_path":
                self.asset_path = value
            elif key == "data":
                self.data = value
                self.save(self.data)

    def save(self, bytes: bytes) -> bytes:
        """
        saves the dataset on the datasets folder
        If datasets doesn't exist yet will create the folder
        """
        if not os.path.exists("./datasets"):
            os.makedirs("./datasets")

        data = bytes.decode().split("\r")

        with open("./datasets/" + self.asset_path, "w") as file:
            file.writelines(data)
            # for line in data:
            #     csv_file.writerow([line])
            # file.write(encoded_stuff, indent=4)
        return bytes

    def read(self) -> str:
        """reads the data from the asset_path directory"""
        if not os.path.exists("./datasets/"):
            raise FileNotFoundError("dataset directory not found")
        if not os.path.exists("./datasets/" + self.asset_path):
            raise FileNotFoundError(f"{self.asset_path} file not found")
        try:
            with open("./datasets/" + self.asset_path, "r") as file:
                return file.read()
        except ValueError:
            raise ValueError("couldn't import from save,"
                             + " the file might be corrupted")