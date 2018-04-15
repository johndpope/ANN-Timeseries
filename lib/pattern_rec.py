import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#import training set from CSV file
dataset = pd.read_csv('./training/patterns.csv')
X_train = dataset.iloc[:, :13].values

#Normalize the data
import sklearn.preprocessing
X_train = sklearn.preprocessing.normalize(X_train,norm='max')

import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import LearningRateScheduler

classifier = Sequential()
classifier.add(Dense(16, input_dim = 10, kernel_initializer = 'uniform', activation = 'sigmoid'))
classifier.add(Dense(12, kernel_initializer = 'uniform', activation = 'sigmoid'))

sgd = keras.optimizers.SGD(lr=0.5,momentum=0)

#compile the ANN
# classifier.compile(optimizer = 'adam', loss='sparse_categorical_crossentropy',metrics = ['binary_accuracy'])
classifier.compile(sgd, loss='sparse_categorical_crossentropy')


#scheduler
def scheduler(epoch):
    if epoch > 5000:
        return 0.1
    else:
        return 0.5

change_lr = LearningRateScheduler(scheduler)


#fitting the ANN to the training set
label= np.array(range(0,1))
classifier.fit(X_train,label,1,7000,2,callbacks=[change_lr])

#predicting the test set results
y_pred = classifier.predict(X_train)

classifier.save('./models/pattern_rec.h5py')

#Y_pred = (y_pred > 0.001)