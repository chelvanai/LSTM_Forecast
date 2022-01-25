import numpy as np
from darts import TimeSeries
from darts.dataprocessing.transformers import Scaler
from darts.models import RNNModel

value = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25]
series = TimeSeries.from_values(np.array(value))

transformer = Scaler()
series_transformed = transformer.fit_transform(series)


my_model = RNNModel(
    model='LSTM',
    hidden_dim=20,
    dropout=0,
    batch_size=2,
    n_epochs=200,
    optimizer_kwargs={'lr': 1e-3},
    model_name='univariate',
    log_tensorboard=True,
    random_state=42,
    training_length=5,
    input_chunk_length=4,
    force_reset=True,
    save_checkpoints=True
)

my_model.fit(series_transformed,
             verbose=True)

my_model.save_model("weight.pth.tar")

predicted = my_model.predict(series=series_transformed, n=10)

res = transformer.inverse_transform(predicted)

print(res)