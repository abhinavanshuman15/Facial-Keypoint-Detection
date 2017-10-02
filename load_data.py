import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

def load():
    train = pd.read_csv('dataset/training.csv')
    train['Image'] = train['Image'].apply(lambda im: np.fromstring(im, sep=' '))
    train = train.dropna()
    X_train = np.vstack(train['Image'].values)/255
    X_train = X_train.astype(np.float32)
    y_train = train[train.columns[:-1]].values
    y_train = (y_train - 48) / 48
    y_train = y_train.astype(np.float32)
    return X_train,y_train
