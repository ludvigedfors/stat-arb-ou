import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

'''
Very simple backtesting fucntion
returns calculated as signal * spread_return - transaction_cost * abs(signal.diff())'''

def backtest(signals, pair_log_prices, tickA, tickB, beta, transaction_cost = 0):

    df = signals.copy()

    log_ret_A = pair_log_prices[tickA].diff().fillna(0)
    log_ret_B = pair_log_prices[tickB].diff().fillna(0)

    spread_return = log_ret_B - beta * log_ret_A

    df["exec_position"] = df["position"].shift(1).fillna(0)
    df["gross_return"] = df["exec_position"] * spread_return

    position_changes = df["exec_position"].diff().abs().fillna(0)
    fee_pct = (transaction_cost / 10000.0) * (1.0 + abs(beta))

    df["costs"] = position_changes * fee_pct
    df["net_return"] = df["gross_return"] - df["costs"]
    df["cum_net_return"] = (1 + df["net_return"]).cumprod() - 1

    daily_rets = df["net_return"].dropna()

    total_return = df["cum_net_return"].iloc[-1]
    trading_days = len(daily_rets)
    ann_return = (1 + total_return) ** (252.0 / trading_days) - 1
    ann_vol = daily_rets.std() * np.sqrt(252)

    sharpe_ratio = ann_return / ann_vol if ann_vol > 0 else 0.0

    equity_curve = 1 + df["cum_net_return"]
    peak = equity_curve.cummax()
    drawdown = (equity_curve - peak) / peak
    max_drawdown = drawdown.min()

    
    active_days = daily_rets[df["exec_position"] != 0]
    win_rate = (active_days > 0).mean() if len(active_days) > 0 else 0.0

    metrics = {
        "Total Return (%)": total_return * 100,
        "Annualized Return (%)": ann_return * 100,
        "Annualized Volatility (%)": ann_vol * 100,
        "Sharpe Ratio": sharpe_ratio,
        "Max Drawdown (%)": max_drawdown * 100,
        "Win Rate (Active Days) (%)": win_rate * 100,
        "Total Trades Executed": int(position_changes.sum() / 2),
    }

    return {"metrics": metrics, "backtest_df": df}


def plot_backtest_results(backtest_df: pd.DataFrame):
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(14, 8), sharex=True, gridspec_kw={"height_ratios": [2, 1]}
    )

    # Top: Z-Score and Positions
    ax1.plot(
        backtest_df.index,
        backtest_df["z_score"],
        label="Spread Z-Score",
        color="steelblue",
        alpha=0.8,
    )
    ax1.axhline(2.0, color="red", linestyle="--", alpha=0.7, label="Upper Entry (+2)")
    ax1.axhline(
        -2.0, color="green", linestyle="--", alpha=0.7, label="Lower Entry (-2)"
    )
    ax1.axhline(0.0, color="black", linestyle=":", alpha=0.5, label="Mean (0)")

    # Highlight active position areas
    ax1.fill_between(
        backtest_df.index,
        -3.5,
        3.5,
        where=backtest_df["position"] == 1,
        color="green",
        alpha=0.15,
        label="Long Spread Active",
    )
    ax1.fill_between(
        backtest_df.index,
        -3.5,
        3.5,
        where=backtest_df["position"] == -1,
        color="red",
        alpha=0.15,
        label="Short Spread Active",
    )

    ax1.set_ylabel("Z-Score")
    ax1.set_title("Pairs Trading Strategy: Signals & Performance")
    ax1.legend(loc="upper left")
    ax1.grid(True, alpha=0.3)

    # Bottom: Cumulative Net Returns
    ax2.plot(
        backtest_df.index,
        backtest_df["cum_net_return"] * 100,
        label="Cumulative Net Return (%)",
        color="purple",
        lw=1.5,
    )
    ax2.set_ylabel("Net PnL (%)")
    ax2.set_xlabel("Date")
    ax2.legend(loc="upper left")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()