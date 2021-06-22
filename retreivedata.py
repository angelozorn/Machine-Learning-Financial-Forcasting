from pandas_datareader.data import DataReader as DR
from datetime import datetime
import math
import warnings
import pandas as pd
import numpy as np
warnings.filterwarnings("ignore")

startDate = '2019-01-01'
endDate = '2021-01-01'
Dates = None

def parse_date(date_str: str, format='%Y-%m-%d'):
    rv = datetime.strptime(date_str, format)
    rv = np.datetime64(rv)
    return rv

def create_labels(data: pd.DataFrame):
    labels = []
    close = data['Close']
    for i in range(1, len(close)):
        last = close[i-1]
        curr = close[i]
        diff = curr - last
        sign = math.copysign(1, diff)
        if sign > 0:
            labels.append(1)
        else:
            labels.append(0)
    data = data.iloc[:-1]
    data['Label'] = labels
    return data

def ohlc_data(ticker):
    data = DR(ticker, start=startDate, end=endDate, data_source='yahoo')
    data = data[['Open', 'High', 'Low', 'Close', 'Volume']]
    data = create_labels(data)
    return data

df = ohlc_data('SPY')
print(df.head(10))