
from copy import deepcopy


class Feature():
    """
    Feature class either categorical and continuous.
    """

    def __init__(self, name: str, type: str = "Not identified"):
        self._name = name
        self._type = type

    @property
    def name(self):
        return deepcopy(self._name)

    @property
    def type(self):
        return deepcopy(self._type)

    @type.setter
    def type(self, new_type):
        if new_type in ["categorical", "continuous"]:
            self._type = new_type
        else:
            print("type has to be categorical or continuous,"
                  + f" was given {new_type} instead")

    def __str__(self) -> str:
        return f"name: {self.name}\ntype: {self.type}"
