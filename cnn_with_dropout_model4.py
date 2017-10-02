import numpy as np
from load_data import load
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Convolution2D, MaxPooling2D,Activation
from keras.optimizers import SGD

def execute(X_train,y_train):
    nb_epoch = 400
    validation_split = 0.33
    lr = 0.01
    momentum = 0.9
    nesterov = True
    loss_method = 'mean_squared_error'
    input_shape = (96,96,1)

    model = Sequential()

    model.add(Convolution2D(32, 3, 3, input_shape=input_shape))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.5))

    model.add(Convolution2D(64, 2, 2))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.5))

    model.add(Convolution2D(128, 2, 2))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.5))

    model.add(Flatten())
    model.add(Dense(500))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(500))
    model.add(Activation('relu'))
    model.add(Dense(30))

    sgd = SGD(lr=lr, momentum=momentum, nesterov=nesterov)
    model.compile(loss=loss_method, optimizer=sgd)

    model.fit(X_train, y_train, nb_epoch=nb_epoch, validation_split=validation_split)

if __name__ == '__main__':
    X_train,y_train = load()
    X_train = X_train.reshape(X_train.shape[0],96,96,1)
    execute(X_train,y_train)
