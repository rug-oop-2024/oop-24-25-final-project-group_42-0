
from typing import List
from autoop.core.ml.dataset import Dataset
from autoop.core.ml.feature import Feature

def detect_feature_types(dataset: Dataset) -> List[Feature]:
    """
    Assumption: only categorical and numerical features and no NaN values.
    Args:
        dataset: Dataset
    Returns:
        List[Feature]: List of features with their types.
    """
    
    pandaframe = dataset.read()
    #pandatypes = pandaframe.dtypes.iloc[:,0]
    featurelist = []
    i = 0
    
    for label, content in pandaframe.items():
        if i == 0:
            pass
        print(f"({i}): {label}")
        datatype = content.dtype.name
        feature = Feature(label)
        if datatype == "object":
            # it is categorical
            feature.type = "categorical"
            featurelist.append(feature)
        else:
            #datatype.startswith("float"):
           # it is numerical
            feature.type = "numerical"
            featurelist.append(feature)
        i += 1
    return featurelist
