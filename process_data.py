import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


'''
'''
def deltaBidLogic(s):
    if s['bid_change'] > 0:
        return s['bid_qty']
    elif s['bid_change'] == 0:
        return s['bid_qty_changed']
    else:
        return 0

'''

'''
def deltaAskLogic(s):
    if s['ask_change'] < 0:
        return s['ask_qty']
    elif s['ask_change'] == 0:
        return s['ask_qty_changed']
    else:
        return 0


def bidAskRatio(s):
    return (s['bid_qty'] - s['ask_qty']) / (s['bid_qty']+s['ask_qty'])


def nextEventLogic(df):
    if df['bid_change'] < 0:
        return 1
    elif df['ask_change'] >0:
        return 2
    else:
        return 0

def calculateZscores(df,field,window):
    mean = df[field].rolling(window=window).mean()
    std = df[field].rolling(window=window).mean()
    return ((df[field] - mean)/std).fillna(0)



if __name__ == '__main__':
    df = pd.read_csv('questions/interview_task.csv')
    df['bid_change'] = df['bid_price'] - df['bid_price'].shift(1)
    df['bid_qty_changed'] = df['bid_qty'] - df['bid_qty'].shift(1)
    df['ask_change'] = df['ask_price'] - df['ask_price'].shift(1)
    df['ask_qty_changed'] = df['ask_qty'] - df['ask_qty'].shift(1)
    df['delta_bid'] = df.apply(deltaBidLogic, axis=1)
    df['delta_ask'] = df.apply(deltaAskLogic, axis=1)
    df['first_event'] = df.apply(nextEventLogic, axis = 1)
    df['first_event'] = df['first_event'].shift(1)
    df['vol_imbalance'] = df['delta_bid'] - df['delta_ask']
    df['bid_ask_ratio'] = df.apply(bidAskRatio,axis=1)
    df['z_bid_qty'] = calculateZscores(df,'bid_qty',120) # 120 ticks approximately 20s
    df['z_ask_qty'] = calculateZscores(df, 'ask_qty', 120)  # 120 ticks approximately 20s
    df.to_csv('data_file.csv',index=False)









