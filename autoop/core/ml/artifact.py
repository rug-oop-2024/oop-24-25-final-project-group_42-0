import base64

from pydantic import BaseModel, Field


class Artifact(BaseModel):

    def __init__(self, *args, **kwargs):
        print(type(args))

        if "type" in kwargs.keys():
            self.type = kwargs["type"]

        for key, value in kwargs:
            if key == "type":
                self.type = value
            elif key == "asset_path":
                self.asset_path == value

    def read():
        print("Fuck you too")
        pass

    def save(self, bytes):
        pass
