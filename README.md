
#### Python Package
    pip install -r requirement.txt

#### HOW TO RUN
Add additional information to data:
    
    python process_data.py

Run the logistics regressions for 3 label columns with some improvement.
    
    python logistics_regression_original.py
    python logistics_regression_optimized.py
    python pcaTest.py
    
#### ANALYSIS

Since the task is trying to the side in 1s,3s,5s of the limit order book (LOB).
We need to understand the shape of the LOB to determine the trend.

First we define volume imbalance (order imbalance as following) definition:

If the current bid price is lower than the previous bid price, that implies that either the trader cancelled his buy
limit order or an order was filled. As we do not have a more granular order
or message book, we cannot be certain of the trader intent, hence we conservatively
delta_bid = 0. If the current bid price is the same as the previous price, we take
the difference of the bid volume to represent incremental buying pressure from last period.

Downward price momentum and sell pressure can be interpreted analogously
from the current and previous ask prices

delta_bid:
  
    0 if bid price decresed
    volume bid now - last volume bid if price unchanged
    volume bid now if bid price increased

deta_ask: vice versa

the order imbalance:
    
    OI = delta_bid - delta_ask
   

also, we want to understand the relative ratio of bid and ask volume

    bid_ask_ratio = (volume_bid - volume_ask)/(volume_bid + volume_ask)    

With some understand of the shape orderbook and buy/sell pressure at each tick data, we now can process with some simple logistics regression (LR) to modeling the label 1s_side, 3s_side and 5s_side

Using simple LR (running logistics_regression_original.py) does not produce good accurate result as following:
    
    _1s_side 77.92738275340393 %
    _3s_side 72.81391830559758 %
    _5s_side 71.27836611195158 %

Digging deeper in the data, we can see that in most period, the orderbook remains unchanged, so it may cause a lot of sparsed data and create noise for the model.

So we only care about moments that there are changes in buy/sell orders are added or remove from the LOB. Filtering the data by the bid_qty_changed and ask_qty_changed as following:

    mask = (df['bid_qty_changed']!=0) & (df['ask_qty_changed']!=0)
    df = df[mask]
    
Re-run the LR with the first filtering steps, we see some changes in the quality:
    
    _1s_side 73.37662337662337
    _3s_side 73.37662337662337
    _5s_side 74.02597402597402

The accuracy is better for 3s and 5s and worse for 1s. Next we are looking to remove the outliers of the dataset where there are big changed in best_bid and best_ask volume because it may cause the high volatility in the market and model may not work well during those period. can find the 5-95 percentile of the quantity changed as following checkData.py:

    print(np.percentile(df['bid_qty_changed'], 5))
    print(np.percentile(df.bid_qty_changed, 95))
    print(np.percentile(df.ask_qty_changed, 5))
    print(np.percentile(df.ask_qty_changed, 95))
 
 Bid/ask all give us left tail =  -4 and and right tail = 3 which are quite consistent. Now we remove the outlier data:
 
    mask = (df['bid_qty_changed'] <= upper) & (df['bid_qty_changed'] >= lower) & (df['ask_qty_changed'] <= upper) & (
                df['ask_qty_changed'] >= lower)
    df = df[mask]
    
Then re-run the LR model again, this time the model is significantly improved:
    
     _1s_side 89.1891891891892%
     _3s_side 86.48648648648648%
    _5s_side 83.7837837837837%
    
All models pass the threshold 80%.

Since the input have similar nature and maybe high correlated(overlappping) to each other, we can try to PCA the input to reduce the dimensions to fine tune the model. We choose to keep 80% of information. The converted matrix B is now only 2 columns compare with original data is 10 columns. Re-run the LR, we have a little better results:

    python pcaTest.py
    _1s_side 93.24324324324324%
    _3s_side 87.83783783783784%
    _5s_side 83.78378378378379%




### Further analysis


Other than predict the state of LOB, I am interested in detect short term momentum in mid-price changed as well.

Study the volume_imbalance data, we can find that it is stationary, also it is auto-correlated until lag-2 and positively correlated with contemporaneous mid-price
changed.

We can actually predict the volume_imbalane by simpe ARMA(2,2) model as described in:

    python arima_test.py
    # const                 -0.006992
    # ar.L1.vol_imbalance    0.154005
    # ar.L2.vol_imbalance    0.684406
    # ma.L1.vol_imbalance   -0.112758
    # ma.L2.vol_imbalance   -0.584724
    
With the positive correlated with positively correlated with contemporaneous price price_changes. Therefore, we can try  to predict the average price changes in next x seconds by simple Ordinary Least Square (OLS) method

    Y: delta mid-price(mid-price - last mid-price) 
    X: last 2 volume imbalance and last 2 predictions errors according to ARMA(2,2)
