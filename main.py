from helpers import binance

values = binance.getCandleSticks()

print(values[0].openTime)
