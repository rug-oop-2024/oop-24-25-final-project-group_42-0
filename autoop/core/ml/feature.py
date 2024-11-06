
from typing import Literal

import numpy as np
from pydantic import BaseModel, Field
from copy import deepcopy

from autoop.core.ml.dataset import Dataset


class Feature():
    # attributes here
    # type: str = Field(default = "Not identified")

    def __init__(self, name: str, type: str = "Not identified"): #data: np.ndarray, 
        self.name = name
        # self._data = data
        self._type = type

    @property
    def type(self):
        return deepcopy(self._type)
    
    # @property
    # def data(self):
    #     return deepcopy(self._data)

    def __str__(self) -> str:
        return f"name: {self.name}\ntype: {self.type}"
        raise NotImplementedError("To be implemented.")
