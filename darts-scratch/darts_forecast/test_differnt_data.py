import numpy as np
import pandas as pd
from darts import TimeSeries
from darts.models import RNNModel
from darts.dataprocessing.transformers import Scaler


def min_max_scale(X, min=None, max=None):
    if min is None:
        X_min = np.min(X, axis=0)
        X_max = np.max(X, axis=0)
        X_scaled = (X - X_min) / (X_max - X_min)
        print(X_min, X_max)
        return X_scaled
    else:
        X_scaled = (X - min) / (max - min)
        return X_scaled


def inverse_min_max(X_scaled, min, max):
    inverse = (X_scaled * (max - min)) + min
    return inverse


value = np.array([2,4,6,8,10,12,14,16,18])
values = min_max_scale(value, 1, 25)
series = TimeSeries.from_values(values)

my_model = RNNModel()

aa = my_model.load_model("weight.pth.tar")

predicted = aa.predict(series=series, n=10)

res = inverse_min_max(predicted, 1, 25)

print(res)
