# indicators.py
import pandas as pd
import ta

def add_indicators(
    df: pd.DataFrame,
    rsi_window: int = 9,
    ema_fast: int = 8,
    ema_slow: int = 21,
    macd_fast: int = 8,
    macd_slow: int = 21,
    macd_signal: int = 7,
) -> pd.DataFrame:
    x = df.copy()

    rsi_ind = ta.momentum.RSIIndicator(close=x["close"], window=rsi_window)
    x["rsi"] = rsi_ind.rsi()

    ema_f = ta.trend.EMAIndicator(close=x["close"], window=ema_fast)
    ema_s = ta.trend.EMAIndicator(close=x["close"], window=ema_slow)
    x["ema_fast"] = ema_f.ema_indicator()
    x["ema_slow"] = ema_s.ema_indicator()

    macd = ta.trend.MACD(close=x["close"],
                         window_fast=macd_fast,
                         window_slow=macd_slow,
                         window_sign=macd_signal)
    x["macd"] = macd.macd()
    x["macd_signal"] = macd.macd_signal()
    x["macd_hist"] = macd.macd_diff()
    return x
