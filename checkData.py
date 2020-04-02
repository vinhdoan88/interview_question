import numpy as np
import pandas as pd

if __name__ == '__main__':
    df = pd.read_csv('data_file.csv')
    df = df.fillna(0)
    print(np.percentile(df['bid_qty_changed'], 5))
    print(np.percentile(df.bid_qty_changed, 95))
    print(np.percentile(df.ask_qty_changed, 5))
    print(np.percentile(df.ask_qty_changed, 95))