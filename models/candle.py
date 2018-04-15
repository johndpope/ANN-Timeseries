from helpers import helper

class Candle:
    def __init__(self,openTime, open, high, low, close, volume, closeTime, quoteAssVol, numOfTrades, tbav, tqav, ignore):
        self.openTime = helper.convertFromEpoch(openTime)
        self.open = open
        self.high = high
        self.low = low
        self.close = close
        self.volume = volume
        self.closeTime = helper.convertFromEpoch(closeTime)
        self.quoteAssVol = quoteAssVol
        self.numOfTrades = numOfTrades
        self.tbav = tbav
        self.tqav = tqav
        self.ignore = ignore