# metrics.py
from typing import Optional, Dict
import pandas as pd
import numpy as np

HOURS_PER_YEAR = 24 * 365

def performance_metrics(
    equity: pd.DataFrame,
    trades: Optional[pd.DataFrame] = None,
    rf_annual: float = 0.0
) -> pd.DataFrame:
    eq = equity.copy()
    eq["ret"] = eq["equity"].pct_change().fillna(0.0)

    rf_hourly = 0.0
    mean_hr = eq["ret"].mean()
    std_hr  = eq["ret"].std(ddof=0)
    down_hr = eq.loc[eq["ret"] < 0, "ret"]

    sharpe  = np.sqrt(HOURS_PER_YEAR) * (mean_hr - rf_hourly) / (std_hr if std_hr > 0 else np.nan)
    sortino = np.sqrt(HOURS_PER_YEAR) * (mean_hr - rf_hourly) / (down_hr.std(ddof=0) if len(down_hr) > 0 else np.nan)

    roll_max = eq["equity"].cummax()
    dd = eq["equity"] / roll_max - 1.0
    max_dd = dd.min()

    years = (eq.index[-1] - eq.index[0]).total_seconds() / (365.25 * 24 * 3600)
    cagr = (eq["equity"].iloc[-1] / eq["equity"].iloc[0]) ** (1 / years) - 1 if years > 0 else np.nan
    calmar = cagr / abs(max_dd) if (max_dd not in [None, 0] and not np.isnan(cagr)) else np.nan

    win_rate_bar = (eq["ret"] > 0).mean()
    win_rate_trade = np.nan
    if trades is not None and len(trades) > 0:
        win_rate_trade = (trades["pnl"] > 0).mean()

    return pd.DataFrame({
        "Sharpe": [sharpe],
        "Sortino": [sortino],
        "Calmar": [calmar],
        "Max Drawdown": [max_dd],
        "Win Rate (bars)": [win_rate_bar],
        "Win Rate (trades)": [win_rate_trade],
        "CAGR": [cagr],
        "Final Equity": [eq["equity"].iloc[-1]]
    })

def returns_tables(equity: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    eq = equity.copy()
    eq["ret"] = eq["equity"].pct_change().fillna(0.0)
    eq["Year"] = eq.index.year
    eq["Month"] = eq.index.month
    eq["Quarter"] = ((eq.index.month - 1) // 3 + 1)

    monthly = eq.groupby(["Year","Month"])["ret"].apply(lambda s: (1+s).prod()-1).reset_index()
    quarterly = eq.groupby(["Year","Quarter"])["ret"].apply(lambda s: (1+s).prod()-1).reset_index()
    annual = eq.groupby(["Year"])["ret"].apply(lambda s: (1+s).prod()-1).reset_index()

    tbl_month = monthly.pivot(index="Year", columns="Month", values="ret").sort_index().round(4)
    tbl_quarter = quarterly.pivot(index="Year", columns="Quarter", values="ret").sort_index().round(4)
    tbl_year = annual.set_index("Year").rename(columns={"ret":"Return"}).round(4)
    return {"monthly": tbl_month, "quarterly": tbl_quarter, "annual": tbl_year}

def calmar_from_equity(eq: pd.DataFrame) -> float:
    roll_max = eq["equity"].cummax()
    dd = eq["equity"] / roll_max - 1.0
    max_dd = dd.min()
    years = (eq.index[-1] - eq.index[0]).total_seconds() / (365.25 * 24 * 3600)
    if years <= 0: 
        return -np.inf
    cagr = (eq["equity"].iloc[-1] / eq["equity"].iloc[0]) ** (1 / years) - 1
    return cagr / abs(max_dd) if max_dd != 0 else -np.inf
