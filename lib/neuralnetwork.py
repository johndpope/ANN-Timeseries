from helpers import binance
import pandas as pd
import numpy as np



def createTrainingData(limit):
    values = binance.getCandleSticks("BTCUSDT","1h",limit=limit)
    # values['ema'] = values["close"].ewm(span=limit, adjust=False).mean()
    values['close'] = values['close'].astype('float64')
    values['openTime'] = pd.to_datetime(values['openTime'])
    return values['close']


def trainNetwork():

    data = createTrainingData(100)
    n = 100

    # split the data 80% training 20% testing
    train_start = 0
    train_end = int(np.floor(0.8*n))
    test_start = train_end
    test_end = n
    data_train = data[np.arange(train_start, train_end)]
    data_test = data[np.arange(test_start, test_end)]

    # Scale Data
    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler()
    scaler.fit(data_train)
    data_train = scaler.transform(data_train)
    data_test = scaler.transform(data_test)

    # build X and y
    X_train = data_train[:, 1:]
    y_train = data_train[:, 0]
    X_test = data_train [:, 1:]
    y_test = data_test[:, 0]


