import pandas as pd
import numpy as np
from load_data import load
from sklearn.multioutput import MultiOutputRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

def execute(X_train, X_test, y_train, y_test):
    clf = MultiOutputRegressor(GradientBoostingRegressor(random_state=0))
    clf.fit(X_train,y_train)
    y_pred = clf.predict(X_test)
    acc_score = mean_squared_error(y_test, y_pred)
    print(np.sqrt(acc_score) * 48)

if __name__ == '__main__':
    X_train, y_train = load()
    X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.33, random_state=42)
    execute(X_train, X_test, y_train, y_test)
    
