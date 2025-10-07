# viz.py
import matplotlib.pyplot as plt
import pandas as pd

def plot_equity(equity: pd.DataFrame, title: str, initial_line: float = None):
    plt.figure(figsize=(12,5))
    equity["equity"].plot(linewidth=2)
    if initial_line is not None:
        plt.axhline(y=initial_line, linestyle='--', alpha=0.5, label='Capital Inicial')
        plt.legend()
    plt.title(title)
    plt.xlabel("Fecha")
    plt.ylabel("Equity (USDT)")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

def print_table(df: pd.DataFrame, title: str):
    print(f"\n=== {title} ===")
    try:
        from IPython.display import display
        display(df)
    except Exception:
        print(df.to_string())
