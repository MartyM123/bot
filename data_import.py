import numpy as np
import pandas as pd

def data(period='2mo', interval = '1d'):
    #Data Source
    import yfinance as yf

    # Get data
    data = yf.download(tickers='BTC-USD', period = period, interval = interval)[['Close']]#add next Parameters such as open high low etc.
    return data
a = 1

def make_frame(data, n=10):
    frame = np.arange(n*a).reshape(n, a, 1)
    for i in range(len(data.iloc[:,0])-n):
        frame = np.append(frame,data[i:i+n].values.reshape(n, a, 1), axis=2)
    return frame[:,:,1:]

def make_list(data, n=10):
    a = []
    c = data.shape[2]
    for i in range(c):
        a.append(data[:,:,i].reshape(data[:,:,i].size))
    return np.array(a)
#(328, 40)
