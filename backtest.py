# backtest.py
from dataclasses import dataclass
from typing import List, Optional, Dict
import pandas as pd
import numpy as np

@dataclass
class Position:
    side: int              # +1 long, -1 short
    entry_time: pd.Timestamp
    entry_price: float
    qty: float
    notional: float
    fee_open: float
    sl: float
    tp: float

def backtest(
    df: pd.DataFrame,
    init_capital: float = 1_000_000.0,
    ticket_usdt: float = 10_000.0,
    fee_rate: float = 0.00125,
    sl_pct: float = 0.05,
    tp_pct: float = 0.10,
) -> Dict[str, pd.DataFrame]:
    x = df.sort_values("datetime").reset_index(drop=True).copy()

    cash = float(init_capital)
    open_pos: List[Position] = []
    equity_curve = []
    trades = []

    def mtm(price: float) -> float:
        value = cash
        for p in open_pos:
            if p.side == 1:
                value += p.qty * price
            else:
                value += (p.entry_price - price) * p.qty + p.qty * p.entry_price
        return value

    first_dt = x.loc[0, "datetime"]
    first_close = float(x.loc[0, "close"])
    equity_curve.append((first_dt, mtm(first_close)))

    def can_open(cash_amt: float) -> bool:
        return cash_amt >= ticket_usdt * (1.0 + fee_rate)

    for i in range(len(x) - 1):
        row = x.iloc[i]
        nxt = x.iloc[i + 1]
        exec_price = float(nxt["open"])
        signal = int(row["signal"])

        # ABRIR
        if signal == 1 and can_open(cash):
            qty = ticket_usdt / exec_price
            fee = ticket_usdt * fee_rate
            cash -= (ticket_usdt + fee)
            sl = exec_price * (1 - sl_pct)
            tp = exec_price * (1 + tp_pct)
            open_pos.append(Position(+1, nxt["datetime"], exec_price, qty, ticket_usdt, fee, sl, tp))

        elif signal == -1 and can_open(cash):
            qty = ticket_usdt / exec_price
            proceeds = ticket_usdt
            fee = proceeds * fee_rate
            cash -= (proceeds + fee)
            sl = exec_price * (1 + sl_pct)
            tp = exec_price * (1 - tp_pct)
            open_pos.append(Position(-1, nxt["datetime"], exec_price, qty, ticket_usdt, fee, sl, tp))

        # GESTIÓN
        still_open = []
        for p in open_pos:
            if p.side == 1:
                if exec_price <= p.sl or exec_price >= p.tp:
                    proceeds = p.qty * exec_price
                    fee_close = proceeds * fee_rate
                    cash += proceeds - fee_close
                    pnl_net = (p.qty * (exec_price - p.entry_price)) - (p.fee_open + fee_close)
                    ret_net = pnl_net / p.notional if p.notional > 0 else np.nan
                    trades.append({
                        "side": "long", "entry_time": p.entry_time, "exit_time": nxt["datetime"],
                        "entry": p.entry_price, "exit": exec_price, "pnl": pnl_net, "ret": ret_net
                    })
                else:
                    still_open.append(p)
            else:
                if exec_price >= p.sl or exec_price <= p.tp:
                    fee_close = (p.qty * exec_price) * fee_rate
                    pnl_net = (p.qty * (p.entry_price - exec_price)) - (p.fee_open + fee_close)
                    cash += (p.qty * (p.entry_price - exec_price)) * (1 - fee_rate) + p.qty * p.entry_price
                    ret_net = pnl_net / p.notional if p.notional > 0 else np.nan
                    trades.append({
                        "side": "short", "entry_time": p.entry_time, "exit_time": nxt["datetime"],
                        "entry": p.entry_price, "exit": exec_price, "pnl": pnl_net, "ret": ret_net
                    })
                else:
                    still_open.append(p)
        open_pos = still_open

        equity_curve.append((nxt["datetime"], mtm(float(nxt["close"]))))

    eq = pd.DataFrame(equity_curve, columns=["datetime", "equity"]).set_index("datetime")
    trades_df = pd.DataFrame(trades) if len(trades) else pd.DataFrame(
        columns=["side","entry_time","exit_time","entry","exit","pnl","ret"]
    )
    return {"equity": eq, "trades": trades_df}
