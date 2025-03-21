import os
import pickle 

from typing import Tuple

def get_snelson(
        name: str,
        path: str,
        train: bool,
        between: bool=True, 
        standardized: bool=False
    ) -> Tuple:
    filename = os.path.join(path, name)
    
    data = "train" if train else "test"
    if not standardized:
        if between:
            filename += "_between.pkl"
        else:
            filename += ".pkl"

        with open(filename, "rb") as f: snelson_data = pickle.load(f)
        X_test = snelson_data[f"{data}_inputs"]
        y_test = snelson_data[f"{data}_outputs"]
        return X_test, y_test
    
    else:
        if between:
            filename += "_between_standardized.pkl"
        else:
            filename += "_standardized.pkl"

        with open(filename, "rb") as f:
            snelson_data = pickle.load(f)
        X_test = snelson_data[f"{data}_inputs"]
        y_test = snelson_data[f"{data}_outputs"]
        y_mean = snelson_data["y_mean"]
        y_std = snelson_data["y_std"]
        return X_test, y_test, y_mean, y_std