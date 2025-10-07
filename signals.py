# signals.py
import pandas as pd
import numpy as np

def two_of_three_signals(
    df: pd.DataFrame,
    rsi_lo: float = 35.0,
    rsi_hi: float = 65.0,
    ema_bias: float = 5.0,
    macd_bias: float = 2.0,
) -> pd.DataFrame:
    z = df.copy()
    bull = (
        (z["rsi"] < rsi_lo).astype(int) +
        ((z["ema_fast"] - z["ema_slow"]) >  ema_bias).astype(int) +
        ((z["macd"] - z["macd_signal"]) > macd_bias).astype(int)
    )
    bear = (
        (z["rsi"] > rsi_hi).astype(int) +
        ((z["ema_fast"] - z["ema_slow"]) < -ema_bias).astype(int) +
        ((z["macd"] - z["macd_signal"]) < -macd_bias).astype(int)
    )
    z["signal"] = 0
    z.loc[(bull >= 2) & ~(bear >= 2), "signal"] = 1
    z.loc[(bear >= 2) & ~(bull >= 2), "signal"] = -1
    return z
