
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
    ground_turth = 1
    observations = 1 
    featurelist = []
    i = 0
    
    #     DataFrame.iterrows()
    # Iterate over DataFrame rows as (index, Series) pairs.

    # pandaframe.columns[0]
    # 'sepal length (cm),sepal width (cm),petal length (cm),petal width (cm)'

    # pandaframe.loc[pandaframe.columns]
    #              sepal length (cm),sepal width (cm),petal length (cm),petal width (cm)
    # NaN NaN NaN                                                NaN                    

    # pandaframe.loc[pandaframe.columns]
    # for _ in pandaframe.columns:
    #     print(f"({i}): {content.array[:, i]}")
    for label, content in pandaframe.items():
        # print(f"({i}): {label}")
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
    
    print(featurelist)
    return featurelist
