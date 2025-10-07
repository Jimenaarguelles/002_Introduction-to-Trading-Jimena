# config.py
CSV_PATH = "BTCUSDT.csv"

# Parámetros baseline (puedes cambiarlos o dejar que Optuna busque otros)
BASE_IND_PARAMS = dict(rsi_window=9, ema_fast=8, ema_slow=21, macd_fast=8, macd_slow=21, macd_signal=7)
BASE_SIG_PARAMS = dict(rsi_lo=35.0, rsi_hi=65.0, ema_bias=5.0, macd_bias=2.0)
BASE_BT_PARAMS  = dict(init_capital=1_000_000.0, ticket_usdt=10_000.0, fee_rate=0.00125, sl_pct=0.05, tp_pct=0.10)
