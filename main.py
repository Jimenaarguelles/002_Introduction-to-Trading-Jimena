# main.py
import optuna
from config import CSV_PATH, BASE_IND_PARAMS, BASE_SIG_PARAMS, BASE_BT_PARAMS
from data_utils import load_btcusdt_hourly, time_splits_idx, slice_df
from indicators import add_indicators
from signals import two_of_three_signals
from backtest import backtest
from metrics import performance_metrics, returns_tables
from walkforward_opt import objective_walkforward_expanding, walk_forward_expanding_returns_on_train
from viz import plot_equity, print_table

def run_baseline(data):
    print("\n" + "="*80)
    print("EVALUACIÓN BASELINE CON PARÁMETROS OPTIMIZADOS (FIJOS)")
    print("="*80)
    df_base = two_of_three_signals(add_indicators(data, **BASE_IND_PARAMS), **BASE_SIG_PARAMS)
    bt_all = backtest(df_base, **BASE_BT_PARAMS)

    metrics_all = performance_metrics(bt_all["equity"], trades=bt_all.get("trades")).round(4)
    print_table(metrics_all, "Métricas Baseline")
    plot_equity(bt_all["equity"], "Equity Curve — Baseline con Parámetros Optimizados", initial_line=BASE_BT_PARAMS["init_capital"])

    tables_all = returns_tables(bt_all["equity"])
    print_table(tables_all["annual"], "Rendimientos Anuales")
    print_table(tables_all["monthly"], "Rendimientos Mensuales")
    print_table(tables_all["quarterly"], "Rendimientos Trimestrales")
    return bt_all

def run_optuna_and_eval(data):
    print("\n" + "="*80)
    print("OPTIMIZACIÓN BAYESIANA CON OPTUNA (Objetivo: Calmar en WF expanding)")
    print("="*80)

    sp = time_splits_idx(len(data))
    df_train = slice_df(data, sp["train"])
    df_test  = slice_df(data, sp["test"])
    df_valid = slice_df(data, sp["valid"])

    print(f"Train: {len(df_train)} | {df_train.iloc[0]['datetime']} → {df_train.iloc[-1]['datetime']}")
    print(f"Test : {len(df_test)}  | {df_test.iloc[0]['datetime']}  → {df_test.iloc[-1]['datetime']}")
    print(f"Valid: {len(df_valid)} | {df_valid.iloc[0]['datetime']} → {df_valid.iloc[-1]['datetime']}")

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(lambda tr: objective_walkforward_expanding(tr, df_train, k_folds=5),
                   n_trials=60, show_progress_bar=False)

    print("\n=== RESULTADOS DE OPTIMIZACIÓN (WF TRAIN) ===")
    print(f"Mejor Calmar Ratio (WF TRAIN): {study.best_value:.4f}")
    for k, v in study.best_params.items():
        print(f" {k}: {v}")

    bp = study.best_params
    ind_best = {k: int(bp[k]) for k in ["rsi_window","ema_fast","ema_slow","macd_fast","macd_slow","macd_signal"]}
    sig_best = {k: float(bp[k]) for k in ["rsi_lo","rsi_hi","ema_bias","macd_bias"]}
    bt_best  = dict(
        init_capital=1_000_000.0,
        ticket_usdt=int(bp["ticket_usdt"]),
        fee_rate=0.00125,
        sl_pct=float(bp["sl_pct"]),
        tp_pct=float(bp["tp_pct"])
    )

    eq_wf = walk_forward_expanding_returns_on_train(df_train, k_folds=5,
                                                    ind_params=ind_best, sig_params=sig_best, bt_params=bt_best)
    print_table(performance_metrics(eq_wf).round(4), "Métricas WF TRAIN (equity continuo)")
    # TEST
    df_test_sig = two_of_three_signals(add_indicators(df_test, **ind_best), **sig_best)
    bt_test = backtest(df_test_sig, **bt_best)
    print_table(performance_metrics(bt_test["equity"], trades=bt_test.get("trades")).round(4), "Métricas TEST")

    # VALID
    df_valid_sig = two_of_three_signals(add_indicators(df_valid, **ind_best), **sig_best)
    bt_valid = backtest(df_valid_sig, **bt_best)
    print_table(performance_metrics(bt_valid["equity"], trades=bt_valid.get("trades")).round(4), "Métricas VALID")

    # TODO DATASET
    df_opt = two_of_three_signals(add_indicators(data, **ind_best), **sig_best)
    bt_opt = backtest(df_opt, **bt_best)
    print_table(performance_metrics(bt_opt["equity"], trades=bt_opt.get("trades")).round(4),
                "Métricas — Portafolio Optimizado (Dataset Completo)")
    plot_equity(bt_opt["equity"], "Equity Curve — Portafolio Optimizado (Dataset Completo)",
                initial_line=bt_best["init_capital"])

    # Tablas de retornos
    tables_opt = returns_tables(bt_opt["equity"])
    print_table(tables_opt["annual"], "Rendimientos Anuales")
    print_table(tables_opt["monthly"], "Rendimientos Mensuales")
    print_table(tables_opt["quarterly"], "Rendimientos Trimestrales")

    # Análisis de trades (resumen rápido por consola)
    if len(bt_opt["trades"]) > 0:
        import pandas as pd
        trades_summary = bt_opt["trades"].groupby("side").agg({
            "pnl": ["count", "mean", "sum", "min", "max"],
            "ret": ["mean", "std", "min", "max"]
        }).round(4)
        print_table(trades_summary, "Resumen por tipo de operación")

        total_trades = len(bt_opt["trades"])
        winning = (bt_opt["trades"]["pnl"] > 0).sum()
        losing  = (bt_opt["trades"]["pnl"] < 0).sum()
        print(f"\nTotal de trades: {total_trades}")
        print(f"Ganadores: {winning} ({(winning/total_trades*100):.2f}%)")
        print(f"Perdedores: {losing} ({(losing/total_trades*100):.2f}%)")
        print(f"P&L total: ${bt_opt['trades']['pnl'].sum():,.2f}")
        print(f"P&L promedio: ${bt_opt['trades']['pnl'].mean():,.2f}")

        trades = bt_opt["trades"].copy()
        trades["duration"] = (
            pd.to_datetime(trades["exit_time"]) - pd.to_datetime(trades["entry_time"])
        ).dt.total_seconds() / 3600
        print(f"\nDuración promedio: {trades['duration'].mean():.2f} h")
        print(f"Duración mínima : {trades['duration'].min():.2f} h")
        print(f"Duración máxima : {trades['duration'].max():.2f} h")
    else:
        print("\nNo se ejecutaron trades en el período analizado.")

    # Comparativa final
    import pandas as pd
    comparison = pd.DataFrame({
        "WF Train": performance_metrics(eq_wf).iloc[0],
        "Test": performance_metrics(bt_test["equity"], trades=bt_test.get("trades")).iloc[0],
        "Valid": performance_metrics(bt_valid["equity"], trades=bt_valid.get("trades")).iloc[0],
        "Todo": performance_metrics(bt_opt["equity"], trades=bt_opt.get("trades")).iloc[0],
    }).T
    print_table(comparison.round(4), "Comparación Train/Test/Valid/Todo")

    print("\n" + "="*80)
    print("FIN DEL ANÁLISIS")
    print("="*80)
    print(f"\n Capital inicial: $1,000,000")
    print(f" Capital final: ${bt_opt['equity']['equity'].iloc[-1]:,.2f}")
    print(f" Retorno total: {(bt_opt['equity']['equity'].iloc[-1]/1_000_000 - 1)*100:.2f}%")

def main():
    data = load_btcusdt_hourly(CSV_PATH)
    print(data.head(3))
    print(data.tail(3))
    print("Fechas:", data["datetime"].min(), "→", data["datetime"].max())

    # 1) Baseline fijo
    run_baseline(data)

    # 2) Optimización + evaluación (igual que tu PARTE 9 y 10)
    run_optuna_and_eval(data)

if __name__ == "__main__":
    main()

