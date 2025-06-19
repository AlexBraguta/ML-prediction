from binance.websocket.um_futures.websocket_client import UMFuturesWebsocketClient
from binance.um_futures import UMFutures
from binance.error import ClientError
from datetime import datetime
import pandas as pd
import os

API_KEY = os.getenv("API_KEY")
API_SECRET = os.getenv("API_SECRET")

client = UMFutures(API_KEY, API_SECRET)


def time(server_time):
    new_time = datetime.fromtimestamp(int(server_time)/1000).strftime('%d.%m.%Y %H:%M:%S')
    return new_time


def get_candles(symbol, chart, nr_candles):
    """Returns specific nr of candles on a specific timeframe chart for a certain symbol"""
    response = client.klines(symbol=symbol, interval=chart, limit=nr_candles)
    for item in response:
        new_time = str(time(item[0]))
        item[0] = new_time
    return response


def display_candles(candles):
    orig_data = pd.DataFrame(candles)
    convert_dict = {0: str, 1: float, 2: float, 3: float, 4: float}
    data = orig_data.astype(convert_dict)
    ohlc_data = data.iloc[:, [0, 1, 2, 3, 4]].copy()
    data = ohlc_data.rename(columns={0: 'time', 1: 'open', 2: 'high', 3: 'low', 4: 'close'})
    return data
