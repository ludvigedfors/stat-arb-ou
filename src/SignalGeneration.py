import pandas as pd
import numpy as np
import statsmodels.api as sm




def calculate_z_score(spread, half_life):
    lookback_window = int(np.round(1.5 * half_life))

    rolling_mean = spread.rolling(window=lookback_window).mean()
    rolling_std = spread.rolling(window=lookback_window).std()

    z_score = (spread - rolling_mean) / rolling_std

    return z_score

def generate_signal(spread: pd.Series, half_life, entry_z = 2.0, exit_z = 0.0, stop_z = 3.5):

    positions = pd.Series(0, index=spread.index, dtype=int)
    current_position = 0 # 0: no position, 1: long, -1: short
    bars_held = 0 # number of trading "steps" the position has been held
    max_bars_held = int(np.round(3 * half_life)) # maximum number of trading "steps" to hold a position 
    # (3 halfings for now)

    z_score = calculate_z_score(spread, half_life)

    for i in range(len(spread)):
        if pd.isna(z_score.iloc[i]):
            continue

        current_z = z_score.iloc[i]

        if current_position == 0:  # No position
            if current_z > entry_z:
                current_position = -1  # Enter short position
                bars_held = 0
            elif current_z < -entry_z:
                current_position = 1  # Enter long position
                bars_held = 0

        elif current_position == 1:  # Long position
            if current_z >= exit_z or bars_held >= max_bars_held or current_z <= -stop_z:
                current_position = 0  # Exit long position
            else:
                bars_held += 1

        elif current_position == -1:  # Short position
            if current_z <= -exit_z or bars_held >= max_bars_held or current_z >= stop_z:
                current_position = 0  # Exit short position
            else:
                bars_held += 1

        positions.iloc[i] = current_position

    signals_df = pd.DataFrame(
        {
            "spread": spread,
            "z_score": z_score,
            "position": positions,  # +1, 0, or -1
        },
        index=spread.index,
    )

    return signals_df








