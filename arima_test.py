import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.tsa.stattools import acf, pacf
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

'''
Volume Imbalance in order book is an important part to predict the trend of mid-price
Investigating the data, we found that we can modeling the volume imbalance by simple ARMA(2,2) to detect the trend of next

'''
if __name__ == '__main__':
    df = pd.read_csv('data_file.csv')
    # # ADF test and stationary if p-value <0.05=> i=0
    result = adfuller(df['vol_imbalance'])
    print('ADF Statistic: %f' % result[0])
    print('p-value: %f' % result[1])
    # vol b


    # we find the p - lag of arma model model
    volAcf = acf(df['vol_imbalance'])
    plt.plot(volAcf[:32])
    plt.grid(True)
    plt.show()

    # choose p =2 here

    # # plot_pacf to choose the q-lag
    plot_pacf(df['vol_imbalance'], lags=100)
    plt.show()
    # so q = 2

    # run the arima(2,2) for order_imbalance
    model = ARIMA(df['vol_imbalance'][:20000], order=(2, 0, 2))
    model_fit = model.fit()
    print(model_fit.summary())
    print(model_fit.params)

    # we found the following parameters to fit
    # '''
    # const                 -0.006992
    # ar.L1.vol_imbalance    0.154005
    # ar.L2.vol_imbalance    0.684406
    # ma.L1.vol_imbalance   -0.112758
    # ma.L2.vol_imbalance   -0.584724
    # dtype: float64

    # # test predict vs real
    test_data = df['vol_imbalance'][20000:21000]
    test_data = test_data.reset_index(drop=True)
    # construct AR(2) part
    ar_1 = test_data.shift(1)
    ar_2 = test_data.shift(2)
    predict = 0.154005 * ar_1 + 0.684406 * ar_2
    # print(predict.head())

    # construct MA(2) part
    res1 = predict[4] - test_data[4]
    res2 = predict[3] - test_data[3]
    for i in range(5,len(predict)):
        predict[i] = predict[i] -0.112758*res1 - 0.584724*res2 -0.006992
        res2 = res1
        res1 = predict[i] - test_data[i]
    #
    #
    # check the model by plot (use cumsum to see the fit of volume imbalance)
    plt.plot(np.cumsum(predict),label='predict')
    plt.plot(np.cumsum(test_data),label='test_data')
    plt.grid(True)
    plt.show()






