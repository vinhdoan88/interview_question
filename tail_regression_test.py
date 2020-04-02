import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def deltaBidLogic(s):
    if s['bid_change'] > 0:
        return s['bid_qty']
    elif s['bid_change'] == 0:
        return s['bid_qty_changed']
    else:
        return 0

def deltaAskLogic(s):
    if s['ask_change'] < 0:
        return s['ask_qty']
    elif s['ask_change'] == 0:
        return s['ask_qty_changed']
    else:
        return 0


if __name__ == '__main__':
    df = pd.read_csv('data/interview_task.csv')
    print(len(df))
    df['bid_change'] = df['bid_price'] - df['bid_price'].shift(1)
    df['bid_qty_changed'] = df['bid_qty'] - df['bid_qty'].shift(1)
    df['ask_change'] = df['ask_price'] - df['ask_price'].shift(1)
    df['ask_qty_changed'] = df['ask_qty'] - df['ask_qty'].shift(1)
    df['delta_bid'] = df.apply(deltaBidLogic, axis=1)
    df['delta_ask'] = df.apply(deltaAskLogic, axis=1)
    # print(df['delta_bid'].head(50))
    # print(df['delta_bid'].head())

    import statsmodels.api as sm


    df['vol_imbalance'] = df['delta_bid'] - df['delta_ask']
    a = np.percentile(df['vol_imbalance'],5)
    print(len(df.loc[df['vol_imbalance']<=-4].dropna()))
    df['roll_bid'] = df['bid_price'].rolling(window=120).min()
    df['bidDecreased'] = df['bid_price'] - df['roll_bid'].shift(-121)
    countTrue = 0
    countFalse = 0
    for i in range(30,len(df)):
        if df.loc[i]['vol_imbalance'] <= a:
            # print(df.loc[i]['vol_imbalance'])
            # if df[i+1:i+30]['bid_price'].mean() < df.loc[i]['bid_price']:
            if df.loc[i]['bidDecreased'] > 0:
                countTrue +=1
                print(countTrue)
            else:
                countFalse += 1
    print('true', countTrue)
    print('false',countFalse)
    print(countTrue/countFalse)
    #
    # df['askIncreased'] = df['vol_imbalance'].loc[df['vol_imbalance']>10]
    # df['askDecreased'] = df['askDecreased'].dropna()
    # print(df['askIncreased'].dropna())
    # print(df[['ask_change','delta_ask','bid_change','delta_bid','vol_imbalance']].loc[21])
    # plt.hist(df['vol_imbalance'],bins=50)
    # plt.show()

    # print(df['vol_imbalance'].head(20))
    # from statsmodels.tsa.stattools import acf, pacf
    # from statsmodels.tsa.stattools import adfuller
    # from statsmodels.tsa.seasonal import seasonal_decompose
    # from statsmodels.tsa.arima_model import ARIMA
    # from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

    # ADF test and stationary if p-value <0.05=> i=0
    # result = adfuller(df['vol_imbalance'])
    # print('ADF Statistic: %f' % result[0])
    # print('p-value: %f' % result[1])
    # acf p = 2
    # volAcf = acf(df['vol_imbalance'])
    # plt.plot(volAcf[:32])
    # plt.grid(True)
    # plt.show()

    # plot_pacf=> q=1,2
    # plot_pacf(df['vol_imbalance'], lags=100)
    # plt.show()
    #
    # model = ARIMA(df['vol_imbalance'][:20000], order=(2, 0, 2))
    # model_fit = model.fit()
    # print(model_fit.summary())
    # print(model_fit.params)

    '''
    const                 -0.006992
    ar.L1.vol_imbalance    0.154005
    ar.L2.vol_imbalance    0.684406
    ma.L1.vol_imbalance   -0.112758
    ma.L2.vol_imbalance   -0.584724
    dtype: float64


    '''

    # test predict vs real
    # test_data = df['vol_imbalance'][20000:21000]
    # test_data = test_data.reset_index(drop=True)
    # print(test_data.head())
    # predict = []
    # ar_1 = test_data.shift(1)
    # ar_2 = test_data.shift(2)
    # predict = 0.154005 * ar_1 + 0.684406 * ar_2
    # print(predict.head())
    # res1 = predict[4] - test_data[4]
    # res2 = predict[3] - test_data[3]
    # for i in range(5,len(predict)):
    #     predict[i] = predict[i] -0.112758*res1 - 0.584724*res2 -0.006992
    #     # print(predict[i])
    #     res2 = res1
    #     res1 = predict[i] - test_data[i]
    #
    # from scipy.stats import pearsonr
    #
    # corr, _ = pearsonr(predict[3:], test_data[3:])
    # print('Pearsons correlation: %.3f' % corr)
    # # plt.plot(np.cumsum(test_data))
    # # print(predict)
    # plt.plot(np.cumsum(predict))
    # plt.plot(np.cumsum(test_data))
    #
    # plt.show()



    #
    # plt.plot(model_fit.forecast(steps=100)[0])
    # plt.show()






