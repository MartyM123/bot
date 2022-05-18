import numpy as np
import pandas as pd

def data(period='2wk', interval = '1h'):
    #Data Source
    import yfinance as yf

    # Get data
    data = yf.download(tickers='BTC-USD', period = period, interval = interval)[['Open', 'High', 'Low', 'Close']]
    return data

def make_frame(data, n=10):
    frame = np.arange(n*4).reshape(n, 4, 1)
    for i in range(len(data.iloc[:,0])-n):
        frame = np.append(frame,data[i:i+n].values.reshape(n, 4, 1), axis=2)
    return frame[:,:,1:]

def make_list(data, n=10):
    a = []
    c = data.shape[2]
    for i in range(c):
        a.append(data[:,:,i].reshape(data[:,:,i].size))
    return np.array(a)
#(328, 40)
