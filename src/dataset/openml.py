import numpy as np
from sklearn.datasets import fetch_openml

def load_openml(name):

    dataset = fetch_openml(name=name, as_frame=True, parser='auto')
    X = dataset.data
    y = dataset.target

    # Identify and remove categorical columns
    categorical_cols = X.select_dtypes(include=['category', 'object']).columns
    X_non_categorical = X.drop(categorical_cols, axis=1)

    X = X_non_categorical
    X = np.array(X)
    y = np.array(y)

    return X.astype(np.float64), y.astype(np.float64)
