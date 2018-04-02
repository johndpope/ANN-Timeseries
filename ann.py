import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('trainingdata.csv')
X = dataset.iloc[:, 0:11].values

#Split the data to 80% training 20% test
#from sklearn.model_selection import train_test_split
#X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=0)

#Feature Scaling
#from sklearn.preprocessing import StandardScaler
#sc = StandardScaler()
#X_train = sc.fit_transform(X_train)
#X_test = sc.transform(X_test)

import keras
from keras.models import Sequential
from keras.layers import Dense


classifier = Sequential()
classifier.add(Dense(1, input_dim = 10, kernel_initializer = 'uniform', activation = 'sigmoid'))
#classifier.add(Dense(12, kernel_initializer = 'uniform', activation = 'sigmoid'))

#compile the ANN
classifier.compile(optimizer = 'adam', loss='binary_crossentropy',metrics = ['binary_accuracy'])

#fitting the ANN to the training set
classifier.fit(X, '1', 100, 1)

#predicting the test set results
y_pred = classifier.predict(X_test)


y_pred = (y_pred > 0.5)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
