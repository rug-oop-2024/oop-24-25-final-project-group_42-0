
from typing import List
from autoop.core.ml.dataset import Dataset
from autoop.core.ml.feature import Feature

def detect_feature_types(dataset: Dataset) -> List[Feature]:
    """Assumption: only categorical and numerical features and no NaN values.
    Args:
        dataset: Dataset
    Returns:
        List[Feature]: List of features with their types.
    """
    
    # categorical
    # numerical
    pandaframe  = dataset.read()
    pandatypes = pandaframe.dtypes.iloc[:,0]
    featurelist = []
    for row in pandatypes:  
        datatype = pandatypes.iloc[row]
        
    raise NotImplementedError("This should be implemented by you.")