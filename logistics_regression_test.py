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
    df = df[["bid_price","bid_qty","ask_price","ask_qty","_5s_side","bid_qty_changed","ask_qty_changed","delta_bid","delta_ask","vol_imbalance","bid_ask_ratio"]]
    # processedData = df.loc[abs(df['bid_qty_changed'])>]
    print(len(df))
    df = df.fillna(0)
    lower = -4
    upper = 3
    # mask = (df['bid_qty_changed']>=upper) | (df['bid_qty_changed']<=lower) | (df['ask_qty_changed']>=upper) | (df['ask_qty_changed']<=lower)

    mask = (df['bid_qty_changed']!=0) & (df['ask_qty_changed']!=0)
    df = df[mask]
    mask = (df['bid_qty_changed'] <= upper) & (df['bid_qty_changed'] >= lower) & (df['ask_qty_changed'] <= upper) & (
                df['ask_qty_changed'] >= lower)
    df = df[mask]
    pcaMat = df.copy()
    pcaMat  = pcaMat.drop('_5s_side',axis=1)
    print(pcaMat.head())
    pca = PCA(n_components=0.9)
    pca.fit(pcaMat)
    B = pca.transform(pcaMat)
    print(B.shape)

    processedMatrix = pd.DataFrame(data=B, columns=constructColumnName(B.shape[1]))
    processedMatrix = processedMatrix.join(df[["_5s_side"]])


    # df = processedMatrix

    df = df.dropna()
    X = df.drop("_5s_side", axis=1)
    Y = df['_5s_side'].astype('category')
    X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size = 0.10, random_state = 99)
    logistic_regression = LogisticRegression()
    logistic_regression.fit(X_train, Y_train)

    Y_pred = logistic_regression.predict(X_test)
    accuracy = metrics.accuracy_score(Y_test, Y_pred)
    accuracy_percentage = 100 * accuracy
    print(accuracy_percentage)