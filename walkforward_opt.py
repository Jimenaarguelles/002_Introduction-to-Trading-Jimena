# walkforward_opt.py
from typing import Dict, Tuple
import pandas as pd
import optuna

from indicators import add_indicators
from signals import two_of_three_signals
from backtest import backtest
from metrics import calmar_from_equity

def _warmup_bars(ind_params: Dict) -> int:
    return int(max(
        ind_params.get("ema_slow", 26),
        ind_params.get("macd_slow", 26),
        ind_params.get("rsi_window", 14),
    ))

def prepare_eval_chunk_with_warmup(
    df_raw: pd.DataFrame,
    start_idx: int,
    end_idx: int,
    ind_params: Dict,
    sig_params: Dict
) -> pd.DataFrame:
    w = _warmup_bars(ind_params)
    a = max(0, start_idx - w)
    b = end_idx
    tmp = add_indicators(df_raw.iloc[a:b].copy(), **ind_params)
    tmp = two_of_three_signals(tmp, **sig_params)
    core = tmp.iloc[(start_idx - a):(end_idx - a)].copy()
    return core

def walk_forward_expanding_returns_on_train(
    df_train: pd.DataFrame,
    k_folds: int,
    ind_params: Dict,
    sig_params: Dict,
    bt_params: Dict,
    init_equity: float = 1_000_000.0
) -> pd.DataFrame:
    n = len(df_train)
    edges = [int(n * i / k_folds) for i in range(k_folds + 1)]
    rets = []

    for f in range(k_folds):
        eval_start = edges[f]
        eval_end   = edges[f+1]
        if eval_start < max(2000, _warmup_bars(ind_params)):
            continue
        chunk = prepare_eval_chunk_with_warmup(df_train, eval_start, eval_end, ind_params, sig_params)
        bt = backtest(chunk, **bt_params)
        r = bt["equity"]["equity"].pct_change().fillna(0.0)
        rets.append(r)

    if not rets:
        raise RuntimeError("WF expanding: no hay folds válidos (historial insuficiente).")

    ret_concat = pd.concat(rets).sort_index()
    eq_cont = init_equity * (1.0 + ret_concat).cumprod()
    return pd.DataFrame({"equity": eq_cont})

def make_param_sampler(trial) -> Tuple[Dict, Dict, Dict]:
    ind = dict(
        rsi_window = trial.suggest_int("rsi_window", 6, 60),
        ema_fast   = trial.suggest_int("ema_fast", 5, 30),
        ema_slow   = trial.suggest_int("ema_slow", 20, 140),
        macd_fast  = trial.suggest_int("macd_fast", 6, 20),
        macd_slow  = trial.suggest_int("macd_slow", 16, 40),
        macd_signal= trial.suggest_int("macd_signal", 5, 15),
    )
    sig = dict(
        rsi_lo   = trial.suggest_int("rsi_lo", 15, 40),
        rsi_hi   = trial.suggest_int("rsi_hi", 60, 85),
        ema_bias = trial.suggest_float("ema_bias", -10.0, 10.0),
        macd_bias= trial.suggest_float("macd_bias", -10.0, 10.0),
    )
    bt = dict(
        init_capital=1_000_000.0,
        ticket_usdt=trial.suggest_int("ticket_usdt", 10_000, 100_000, step=1_000),
        fee_rate=0.00125,
        sl_pct=trial.suggest_float("sl_pct", 0.01, 0.15),
        tp_pct=trial.suggest_float("tp_pct", 0.02, 0.25),
    )
    return ind, sig, bt

def objective_walkforward_expanding(trial, df_train: pd.DataFrame, k_folds: int = 5) -> float:
    ind_params, sig_params, bt_params = make_param_sampler(trial)
    eq_cont = walk_forward_expanding_returns_on_train(df_train, k_folds, ind_params, sig_params, bt_params)
    return calmar_from_equity(eq_cont)
