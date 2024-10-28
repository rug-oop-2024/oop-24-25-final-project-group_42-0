import base64
import csv, re
import os

from pydantic import BaseModel, Field


class Artifact():

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
        saves the game om the savedgames folder
        If savedgames doesn't exist yet will create the folder
        """
        if not os.path.exists("./savedgames"):
            os.makedirs("./savedgames")

        data = bytes.decode().split("\r\n")

        with open("./savedgames/save.csv", "w") as file:
            print(file)
            csv_file = csv.writer(file, delimiter=" ")
            for line in data:
                csv_file.writerow([line])
            # file.write(encoded_stuff, indent=4)
        return bytes

    def read(self) -> bytes:
        """reads the game from the savedgames directory"""
        if not os.path.exists("./savedgames/"):
            raise FileNotFoundError("savedgames directory not found")
        if not os.path.exists("./savedgames/save.csv"):
            raise FileNotFoundError("save file not found")
        try:
            with open("./savedgames/save.csv", "r") as file:
                csv_string = file.read()
            return csv_string.encode()
        except ValueError:
            raise ValueError("couldn't import from save,"
                             + " the file might be corrupted")