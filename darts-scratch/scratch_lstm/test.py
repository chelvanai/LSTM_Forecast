import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
from torch.autograd import Variable

from model import LSTM
from hparams import input_size, hidden_size, num_epochs, num_layers, learning_rate, num_classes


def test():
    model = LSTM(num_classes, input_size, hidden_size, num_layers)
    model.load_state_dict(torch.load('weight.pth'))
    model.eval()

    # training_set = pd.read_csv('airline-passengers.csv')
    #
    # training_set = training_set.iloc[0:8, 1:2].values

    training_set = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25])
    training_set = np.expand_dims(training_set, axis=1)

    sc = MinMaxScaler()
    data = sc.fit_transform(training_set)

    dataX = torch.Tensor(np.array(data))
    dataX = dataX.unsqueeze(0)

    predict = model(dataX).data.numpy()
    # print(predict)

    inverse_pred = sc.inverse_transform(predict)
    print(inverse_pred)


if __name__ == '__main__':
    test()
