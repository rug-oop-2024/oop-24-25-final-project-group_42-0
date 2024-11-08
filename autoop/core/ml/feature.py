
from copy import deepcopy
from typechecker import Type_Checker, Raise_Type_Error
from pydantic import PrivateAttr

import numpy as np


class Feature():

    _data: np.ndarray = PrivateAttr(default=np.ndarray)

    def __init__(self, name: str, type: str = "Not identified"):
        self._name = name
        self._type = type

    @property
    def name(self):
        return deepcopy(self._name)
    
    @property
    def type(self):
        return deepcopy(self._type)
    
    @property
    def data(self):
        return deepcopy(self._type)
    
    @type.setter
    def type(self, new_type):
        if new_type in ["categorical", "numerical"]:
            self._type = new_type
        else:
            print("type has to be categorical or numerical,"
                  + f" was given {new_type} instead")
            
    @data.setter
    def data(self, new_data: np.ndarray):
        if not Type_Checker(new_data, np.ndarray):
            Raise_Type_Error(new_data, np.ndarray, "new_data")
        elif len(new_data.shape) != 1:
            raise ValueError("new_data should have one column,"
                             + f"instead got an np.ndarray of {len(new_data)} length")
        self._data = new_data

    def __str__(self) -> str:
        return f"name: {self.name}\ntype: {self.type}\ndata: {self.data}"
