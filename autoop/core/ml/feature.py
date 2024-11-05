
from typing import Literal

import numpy as np
from pydantic import BaseModel, Field

from autoop.core.ml.dataset import Dataset


class Feature():
    # attributes here
    # type: str = Field(default = "Not identified")

    def __init__(self, name: str, type: str = "Not identified"):
        self.name = name
        self.type = type

    def __str__(self) -> str:
        return f"name: {self.name}\ntype: {self.type}"
        raise NotImplementedError("To be implemented.")
