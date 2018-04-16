import pandas as pd
from lib import neuralnetwork
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

value = neuralnetwork.createTrainingData(55)


value.plot(x='openTime',y='close')

# value['x'] = pd.to_datetime(value['openTime'])

# plt.plot(value['x'], float(value['close']))
# plt.show()
# plt.scatter(values['openTime'], values['close'])
# plt.show()
