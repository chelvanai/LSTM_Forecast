import numpy as np
import pandas as pd
from darts import TimeSeries
from darts.models import RNNModel
from darts.dataprocessing.transformers import Scaler

value = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25]
series = TimeSeries.from_values(np.array(value))

transformer = Scaler()
series_transformed = transformer.fit_transform(series)

my_model = RNNModel()

aa = my_model.load_model("weight.pth.tar")

predicted = aa.predict(series=series_transformed, n=10)

res = transformer.inverse_transform(predicted)

print(res)
