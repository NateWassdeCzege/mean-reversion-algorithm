import numpy as np
import pandas as pd
import yfinance as yf


# ==========================================
# 0. Top-Level Helper Functions
# ==========================================
def calculate_rolling_window(window, price_col="KO_Open"):
    """Separates the rolling window into historical prices and the live tick."""
    # Use price_col if available, otherwise default to the first column
    col = price_col if price_col in window.columns else window.columns[0]

    history = window.iloc[:-1]
    live_tick = window.iloc[-1]

    historical_price = history[col]
    current_tick = float(live_tick[col])

    return historical_price, current_tick


def calculate_adx(df, period=21):
    """Calculates ADX using Wilder's Smoothing."""
    high, low, close = df["High"], df["Low"], df["Close"]

    tr1 = high - low
    tr2 = np.abs(high - close.shift(1))
    tr3 = np.abs(low - close.shift(1))
    tr = pd.DataFrame({"tr1": tr1, "tr2": tr2, "tr3": tr3}).max(axis=1)

    up_move = high.diff()
    down_move = low.shift(1) - low
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

    tr_smooth = tr.ewm(alpha=1 / period, adjust=False).mean()
    plus_dm_smooth = (
        pd.Series(plus_dm, index=df.index)
        .ewm(alpha=1 / period, adjust=False)
        .mean()
    )
    minus_dm_smooth = (
        pd.Series(minus_dm, index=df.index)
        .ewm(alpha=1 / period, adjust=False)
        .mean()
    )

    plus_di = 100 * (plus_dm_smooth / tr_smooth)
    minus_di = 100 * (minus_dm_smooth / tr_smooth)

    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    return dx.ewm(alpha=1 / period, adjust=False).mean()


def calculate_hurst(series, max_lag=45):
    """Calculates the Hurst Exponent using Rescaled Range (R/S) Analysis."""
    ts = np.asarray(series)
    if len(ts) < max_lag:
        return 0.5

    lags = range(10, max_lag)
    rs_results = []

    for lag in lags:
        num_chunks = len(ts) // lag
        if num_chunks < 1:
            continue

        rs_chunks = []
        for i in range(num_chunks):
            chunk = ts[i * lag : (i + 1) * lag]
            mean = np.mean(chunk)
            cum_dev = np.cumsum(chunk - mean)

            R = np.max(cum_dev) - np.min(cum_dev)
            S = np.std(chunk)

            if S > 0:
                rs_chunks.append(R / S)

        if len(rs_chunks) > 0:
            rs_results.append((lag, np.mean(rs_chunks)))

    if len(rs_results) < 2:
        return 0.5

    log_lags = np.log([item[0] for item in rs_results])
    log_rs = np.log([item[1] for item in rs_results])

    return float(np.polyfit(log_lags, log_rs, 1)[0])


# ==========================================
# 1. Standard Real-Time Streamer (3-tuple)
# ==========================================
def real_time_streamer(df, lookback_periods=210):
    """Streams datetimes, historical lookback prices, and the live tick price."""
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)

    if len(df) <= lookback_periods:
        raise ValueError(
            f"Dataset only has {len(df)} rows, but needs {lookback_periods} for"
            " lookback."
        )

    for i in range(lookback_periods, len(df)):
        window = df.iloc[i - lookback_periods : i + 1]
        current_time_label = df.index[i]

        historical_price, current_tick = calculate_rolling_window(window)

        yield current_time_label, historical_price, current_tick


# ==========================================
# 2. HMM Real-Time Streamer 2 (4-tuple)
# ==========================================
def real_time_streamer2(df_KO, lookback_periods=210):
    """Calculates all technical indicators on df_KO first, then streams:

    - current_time_label
    - historical_price
    - current_tick
    - live_features_df (1-row DataFrame containing HMM features)
    """
    df = df_KO.copy()

    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)

    if len(df) <= lookback_periods:
        raise ValueError(
            f"Dataset only has {len(df)} rows, but needs {lookback_periods} for"
            " lookback."
        )

    target_price = "KO_Open" if "KO_Open" in df.columns else df.columns[0]

    # --- Pre-compute HMM Features ---
    print("Calculating HMM features for real-time streaming...")
    df["log_return"] = np.log(df[target_price] / df[target_price].shift(1))
    df["real_volatility"] = df["log_return"].rolling(window=25).std()

    fast_ma = df[target_price].rolling(window=20).mean()
    slow_ma = df[target_price].rolling(window=130).mean()
    df["ma_spread"] = (fast_ma - slow_ma) / slow_ma

    df["roc_20"] = df[target_price].pct_change(periods=20)
    df["smooth_momentum"] = df["roc_20"].ewm(span=20, adjust=False).mean()

    df["adx"] = calculate_adx(df, period=21)

    df["hurst_exponent"] = (
        df[target_price]
        .rolling(window=130)
        .apply(lambda x: calculate_hurst(x, max_lag=45), raw=True)
    )

    # --- Handle Extreme Outliers ---
    df["log_return"] = df["log_return"].clip(
        lower=df["log_return"].quantile(0.01),
        upper=df["log_return"].quantile(0.99),
    )
    df["smooth_momentum"] = df["smooth_momentum"].clip(
        lower=df["smooth_momentum"].quantile(0.01),
        upper=df["smooth_momentum"].quantile(0.99),
    )

    # Features expected by the trained HMM
    feature_columns = ["smooth_momentum", "log_return", "adx", "ma_spread"]

    # --- Live Stream Yield Loop ---
    for i in range(lookback_periods, len(df)):
        window = df.iloc[i - lookback_periods : i + 1]
        current_time_label = df.index[i]

        historical_price, current_tick = calculate_rolling_window(
            window, price_col=target_price
        )

        # 1-row DataFrame containing features for current_tick
        #live_features_df = df.iloc[[i]][feature_columns]
        live_features_df = window[feature_columns]

        yield current_time_label, historical_price, current_tick, live_features_df