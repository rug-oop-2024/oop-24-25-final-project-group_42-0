
from typing import Literal

import numpy as np
from pydantic import BaseModel, Field

from autoop.core.ml.dataset import Dataset


class Feature():
    # attributes here
    type: str = Field(default = "Not identified")

    def __init__(self, name: str):
        self.name = name

    def __str__(self) -> str:
        raise NotImplementedError("To be implemented.")
