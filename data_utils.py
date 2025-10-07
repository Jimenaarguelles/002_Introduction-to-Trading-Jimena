# data_utils.py
import pandas as pd
import numpy as np

def load_btcusdt_hourly(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path, skiprows=1)
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
    dt = pd.to_datetime(df["date"], utc=True, dayfirst=True, errors="coerce")

    out = pd.DataFrame({
        "datetime": dt,
        "open":  pd.to_numeric(df["open"],  errors="coerce"),
        "high":  pd.to_numeric(df["high"],  errors="coerce"),
        "low":   pd.to_numeric(df["low"],   errors="coerce"),
        "close": pd.to_numeric(df["close"], errors="coerce"),
    })

    out = (
        out.dropna(subset=["datetime","open","high","low","close"])
           .sort_values("datetime")
           .reset_index(drop=True)
    )
    out["ret"] = out["close"].pct_change()
    out["log_ret"] = np.log1p(out["ret"])
    return out

def time_splits_idx(n: int, ratios=(0.6, 0.2, 0.2)):
    i1 = int(n * ratios[0])
    i2 = int(n * (ratios[0] + ratios[1]))
    return {"train": (0, i1), "test": (i1, i2), "valid": (i2, n)}

def slice_df(df: pd.DataFrame, idx):
    a, b = idx
    return df.iloc[a:b].copy()
