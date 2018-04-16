from helpers import binance




def createTrainingData(limit):
    values = binance.getCandleSticks("BTCUSDT","1h",limit=limit)
    # values['ema'] = values["close"].ewm(span=limit, adjust=False).mean()
    return values


