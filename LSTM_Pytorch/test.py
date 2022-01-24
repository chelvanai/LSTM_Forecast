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
    model.load_state_dict(torch.load('weight/weight.pth'))
    model.eval()

    # training_set = pd.read_csv('airline-passengers.csv')
    #
    # training_set = training_set.iloc[0:8, 1:2].values

    training_set = np.array([2,4,6,8,10,12,14,16,18,20,22,24,26,28])
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
