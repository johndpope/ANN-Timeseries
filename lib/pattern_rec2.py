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
from keras.callbacks import LearningRateScheduler, EarlyStopping

classifier = Sequential()
classifier.add(Dense(16, input_dim = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))
classifier.add(Dense(12, kernel_initializer = 'uniform', activation = 'sigmoid'))

sgd = keras.optimizers.SGD(lr=0.5,momentum=0)

#compile the ANN
# classifier.compile(optimizer = 'adam', loss='sparse_categorical_crossentropy',metrics = ['binary_accuracy'])
classifier.compile(sgd, loss='mean_squared_error')


#scheduler
def scheduler(epoch):
    if epoch > 1500:
        return 0.1
    else:
        return 0.5

change_lr = LearningRateScheduler(scheduler)

cb_list = [change_lr]

#fitting the ANN to the training set
label= keras.utils.np_utils.to_categorical(np.array(range(0,12)))
# label = np.array(range(0,12))
classifier.fit(X_train,label,1,2000,2,callbacks=cb_list)

#predicting the test set results
y_pred = classifier.predict(X_train)

classifier.save('./models/pattern_rec.h5py')

#Y_pred = (y_pred > 0.001)