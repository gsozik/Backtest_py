import pandas as pd
import numpy as np
import requests

def fetch_ohlcv_df(exchange, symbol, timeframe, start, end):
    start_ms = int(pd.Timestamp(start, tz="UTC").timestamp() * 1000)
    end_ms = int(pd.Timestamp(end, tz="UTC").timestamp() * 1000)

    data = []
    since = start_ms

    while since < end_ms:
        batch = exchange.fetch_ohlcv(symbol, timeframe, since=since, limit=1000)
        if not batch:
            break
        data.extend(batch)
        since = batch[-1][0] + 1

    df = pd.DataFrame(data, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df = df.set_index("timestamp")
    return df.loc[start:end]

def moex_fetch_ohlcv_df(ticker, timeframe, start, end, board="TQBR"):
    tf = {"1m": 1, "10m": 10, "1h": 60, "1d": 24, "1w": 7, "1M": 31, "1Q": 4}
    interval = tf[timeframe]

    url = f"https://iss.moex.com/iss/engines/stock/markets/shares/boards/{board}/securities/{ticker}/candles.json"
    start_ts = pd.Timestamp(start, tz="UTC")
    end_ts   = pd.Timestamp(end, tz="UTC")

    rows, off = [], 0
    while True:
        r = requests.get(url, params={"from": start, "till": end, "interval": interval, "start": off}).json()["candles"]
        chunk = r["data"]
        if not chunk:
            break
        rows += chunk
        off += len(chunk)

    df = pd.DataFrame(rows, columns=r["columns"])
    df["timestamp"] = pd.to_datetime(df["begin"]).dt.tz_localize("Europe/Moscow").dt.tz_convert("UTC")
    df = df.set_index("timestamp")[["open","high","low","close","volume"]]
    return df.loc[start_ts:end_ts]