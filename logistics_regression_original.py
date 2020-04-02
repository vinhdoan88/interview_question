import pandas as pd
import numpy as np
from sklearn import metrics 
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import statsmodels.api as sm

def constructColumnName(shape):
    colList = []
    for i in range(shape):
        colList.append('column'+str(i+1))
    return colList



if __name__ == "__main__":
    df = pd.read_csv("data_file.csv",parse_dates=['timestamp'])
    input_list = ["bid_price","bid_qty","ask_price","ask_qty","bid_qty_changed","ask_qty_changed","delta_bid","delta_ask","vol_imbalance","bid_ask_ratio"]
    label_list = ["_1s_side","_3s_side","_5s_side"]
    df = df[input_list+label_list]
    df = df.dropna()
    # X = df.drop("_1s_side", axis=1)
    for col in label_list:
        X = df.drop(col,axis=1)

    for col in label_list:
        Y = df[col].astype('category')
        X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size = 0.10, random_state = 99)
        logistic_regression = LogisticRegression()
        logistic_regression.fit(X_train, Y_train)
        Y_pred = logistic_regression.predict(X_test)
        accuracy = metrics.accuracy_score(Y_test, Y_pred)
        accuracy_percentage = 100 * accuracy
        print(col,accuracy_percentage)