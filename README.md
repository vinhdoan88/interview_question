
#### Python Package
    pip install -r requirement.txt

#### HOW TO RUN
Add additional information to data:
    
    python process_data.py

Run the logistics regressions for 3 label columns
    
    python logistics_regression_test.py
    
#### ANALYSIS

Since the task is trying to the side in 1s,3s,5s of the limit order book (LOB).
We need to understand the shape of the LOB to determine the trend.

First we define volume imbalance (order imbalance as following) definition:

delta_bid:
  
    0 if bid price decresed
    volume bid now - last volume bid if price unchanged
    volume bid now if bid price increased

deta_ask: vice versa


the order imbalance:
    
    delta_bid - delta_ask