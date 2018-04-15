import datetime

def convertFromEpoch(epochTime):
    fmt = "%Y-%m-%d %H:%M:%S"
    time = datetime.datetime.fromtimestamp(float(epochTime) / 1000.)

    return time.strftime(fmt)

