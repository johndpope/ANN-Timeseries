import requests
import pandas as pd
from helpers import helper

URL = 'https://api.binance.com/api/v1/klines'

# Use proxy if needed
https_proxy = { "https": "http://xq66:1qaz%23EDC@xdcwsa.central.piraeusgroup.gr:8080"}


# Get Candle values from Binance
def _getBinanceCandles(params):

    resp = requests.get(url=URL, params=params, proxies = https_proxy,verify=False)
    return resp.json()

# Create panda dataframe from Binance API response
def _createCandleObject(candleList):
    
    df = pd.DataFrame(candleList)

    # Remove unwanted columns
    df.drop(df.columns[[7,9,10,11]], axis=1, inplace=True)
    df.columns = ['openTime', 'open', 'high', 'low', 'close', 'volume', 'closeTime','numOfTrades']

    #Convert epoch to datetime values
    df.openTime = df.openTime.apply(lambda x: helper.convertFromEpoch(x)) 
    df.closeTime = df.closeTime.apply(lambda x: helper.convertFromEpoch(x)) 
    return df

 
# Get data from Binance and convert it
def getCandleSticks(symbol, interval,limit):

    #Define API parameters
    params = dict(
        symbol=symbol,
        interval=interval,
        limit=limit
    )
    return _createCandleObject(_getBinanceCandles(params))
