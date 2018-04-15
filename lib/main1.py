import keras
from keras.models import load_model

# Load predictive algorithm
model = load_model('./models/pattern_rec.h5py')


# import test data
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


dataset = pd.read_csv('./test/market-price.csv')
X_test = dataset.iloc[:, :11].values
#x_test = X_test.reshape(-1,1)

#Normalize the data
import sklearn.preprocessing 
X_test = sklearn.preprocessing.normalize(X_test,norm='max')

Y_pred = model.predict(X_test)
Y_pred