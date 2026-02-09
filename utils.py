import pandas as pd
import numpy as np
import requests
from scipy.optimize import minimize

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

def search_for_solutions(
    returns: pd.Series,
    cov: pd.DataFrame,
    risk: float | None = None,
    target_return: float | None = None
):
    tickers = returns.index
    r = returns.values.astype(float)
    C = cov.loc[tickers, tickers].values.astype(float)
    n = len(r)

    w0 = np.ones(n) / n
    bounds = [(0.0, 1.0)] * n
    cons = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]

    # 4) Минимальный риск при заданной доходности (target_return)
    if target_return is not None:
        cons.append({"type": "eq", "fun": lambda w: (w @ r) - target_return})
        obj = lambda w: (w.T @ C @ w)

    # 1) Минимальный риск (GMV)
    elif risk is None:
        obj = lambda w: (w.T @ C @ w)

    # 2) Макс. доходность при заданном риске (vol <= risk) или
    # 3) Макс. доходность без ограничения по риску (risk == 1)
    else:
        obj = lambda w: -(w @ r)
        if risk != 1:
            if risk <= 0:
                raise ValueError("risk должен быть > 0, либо risk=1 для max return без ограничений")
            cons.append({"type": "ineq", "fun": lambda w: risk - np.sqrt(max(w.T @ C @ w, 0.0))})

    res = minimize(obj, w0, method="SLSQP", bounds=bounds, constraints=cons,
                   options={"maxiter": 2000, "ftol": 1e-12})

    if not res.success:
        raise RuntimeError("risk слишком мал или target_return недостижим при long-only.")

    w = res.x
    w[np.abs(w) < 1e-10] = 0.0
    w = w / w.sum()

    var = float(w.T @ C @ w)
    return {
        "weights": pd.Series(w, index=tickers),
        "portfolio_risk": float(np.sqrt(max(var, 0.0))),
        "portfolio_return": float(w @ r),
    }