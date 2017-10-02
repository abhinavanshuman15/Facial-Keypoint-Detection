import keras
import numpy as np
from load_data import load
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.optimizers import SGD

def execute(X_train,y_train):
    loss_method = 'mean_squared_error'
    lr = 0.01
    momentum = 0.9
    nesterov = True
    nb_epoch = 400
    validation_split = 0.33
    model = Sequential()
    model.add(Dense(100, input_dim=9216))
    model.add(Activation('relu'))
    model.add(Dense(30))
    sgd = SGD(lr=lr, momentum=momentum, nesterov=nesterov)
    model.compile(loss=loss_method, optimizer=sgd)
    model.fit(X_train, y_train,validation_split=validation_split, nb_epoch=nb_epoch) 

if __name__ == '__main__':
    X_train,y_train = load()
    execute(X_train,y_train)
