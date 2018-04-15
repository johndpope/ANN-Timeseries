import requests
from models import candle as model

URL = 'https://api.binance.com/api/v1/klines'

# Get Candle values from Binance
def _getBinanceCandles(symbol, interval):



    resp = requests.get(url=URL, params=params)
    return resp.json()

# Convert Json list to list of objects
def _createCandleObject(candleList):
    finalResult = []
    for values in candleList:
        finalResult.append(model.Candle(*values))
    return finalResult

# Get data from Binance and convert it
def getCandleSticks(symbol, interval):

    params = dict(
        symbol=symbol,
        interval=interval
    )
    return _createCandleObject(_getBinanceCandles(params))
