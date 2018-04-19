from helpers import binance
import pandas as pd
import numpy as np



def createTrainingData(limit):
    values = binance.getCandleSticks("BTCUSDT","4h",limit=limit)
    # values['ema'] = values["close"].ewm(span=limit, adjust=False).mean()
    values['close'] = values['close'].astype('float64')
    values['openTime'] = pd.to_datetime(values['openTime'])
    return values['close']



def create_dataset(dataset, look_back=1):
    dataX, dataY = [], []
    for i in range(len(dataset)-look_back-1):
        a = dataset[i:(i+look_back), 0]
        dataX.append(a)
        dataY.append(dataset[i + look_back, 0])
    return np.array(dataX), np.array(dataY)


def trainNetwork():


    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import math

    from sklearn.preprocessing import MinMaxScaler
    from sklearn.metrics import mean_squared_error

    from keras.models import Sequential
    from keras.layers import Dense
    from keras.layers import LSTM

    data = createTrainingData(200)
    dataset = data.values
    dataset = dataset.astype('float32')
    

    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler()
    dataset = dataset.reshape(-1,1)
    dataset = scaler.fit_transform(dataset)
    
    train_size = int(len(dataset) * 0.60)
    test_size = len(dataset) - train_size

    train, test = dataset[0:train_size, :], dataset[train_size:len(dataset), :]

    look_back = 2
    trainX, trainY = create_dataset(train, look_back=look_back)
    testX, testY = create_dataset(test, look_back=look_back)

    trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
    testX = np.reshape(testX, (testX.shape[0], 1, testX.shape[1]))

    model = Sequential()
    model.add(LSTM(4, input_shape=(1, look_back)))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    model.fit(trainX, trainY, epochs=100, batch_size=256, verbose=2)
 
    trainPredict = model.predict(trainX)
    testPredict = model.predict(testX)
    trainPredict = scaler.inverse_transform(trainPredict)
    trainY = scaler.inverse_transform([trainY])
    testPredict = scaler.inverse_transform(testPredict)
    testY = scaler.inverse_transform([testY])

    trainScore = math.sqrt(mean_squared_error(trainY[0], trainPredict[:, 0]))
    print('Train Score: %.2f RMSE' % (trainScore))
    testScore = math.sqrt(mean_squared_error(testY[0], testPredict[:, 0]))
    print('Test Score: %.2f RMSE' % (testScore))

    # shift train predictions for plotting
    trainPredictPlot = np.empty_like(dataset)
    trainPredictPlot[:, :] = np.nan
    trainPredictPlot[look_back:len(trainPredict) + look_back, :] = trainPredict

     # shift test predictions for plotting
    testPredictPlot = np.empty_like(dataset)
    testPredictPlot[:, :] = np.nan
    testPredictPlot[len(trainPredict) + (look_back * 2) + 1:len(dataset) - 1, :] = testPredict

    plt.plot(data, label='Actual')
    plt.plot(pd.DataFrame(trainPredictPlot, columns=["close"], index=data.index).close, label='Training')
    plt.plot(pd.DataFrame(testPredictPlot, columns=["close"], index=data.index).close, label='Testing')
    plt.legend(loc='best')
    plt.show()