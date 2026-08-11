import numpy as np
import pandas as pd


# 0. Top-Level Helper Functions (Unchanged)

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



# 2. Real-Time Feature Engine (Stateful)

class RealTimeFeatureEngine:
    def __init__(self, warmup_df, lookback_periods=210, max_buffer_size=300):
        """
        Initializes the engine with historical warmup data so indicators 
        like ADX and moving averages are ready for the first live tick.
        """
        self.lookback_periods = lookback_periods
        # max_buffer_size keeps memory light. Must be > lookback_periods + max_lag
        self.max_buffer_size = max(max_buffer_size, lookback_periods + 50)
        
        # Ensure timestamp index
        if not isinstance(warmup_df.index, pd.DatetimeIndex):
            warmup_df.index = pd.to_datetime(warmup_df.index)
            
        self.buffer = warmup_df.copy()
        self.feature_columns = ["smooth_momentum", "log_return", "adx", "ma_spread"]

    def process_new_bar(self, new_timestamp, new_bar_dict):
        """
        Ingests a single new live bar, updates indicators, and outputs state.
        new_bar_dict format: {'KO_Open': 60.5, 'High': 61.0, 'Low': 60.0, 'Close': 60.8}
        """
        # 1. Append new bar to the buffer
        new_row = pd.DataFrame([new_bar_dict], index=[new_timestamp])
        self.buffer = pd.concat([self.buffer, new_row])

        # 2. Trim buffer to prevent memory bloat
        if len(self.buffer) > self.max_buffer_size:
            self.buffer = self.buffer.iloc[-self.max_buffer_size:]

        # 3. Compute indicators on the rolling buffer
        df = self.buffer.copy()
        target_price = "KO_Open" if "KO_Open" in df.columns else df.columns[0]

        df["log_return"] = np.log(df[target_price] / df[target_price].shift(1))
        
        fast_ma = df[target_price].rolling(window=20).mean()
        slow_ma = df[target_price].rolling(window=130).mean()
        df["ma_spread"] = (fast_ma - slow_ma) / slow_ma

        df["roc_20"] = df[target_price].pct_change(periods=20)
        df["smooth_momentum"] = df["roc_20"].ewm(span=20, adjust=False).mean()

        df["adx"] = calculate_adx(df, period=21)

        # 4. Handle Outliers (Dynamic clipping)
        df["log_return"] = df["log_return"].clip(
            lower=df["log_return"].quantile(0.01),
            upper=df["log_return"].quantile(0.99),
        )
        df["smooth_momentum"] = df["smooth_momentum"].clip(
            lower=df["smooth_momentum"].quantile(0.01),
            upper=df["smooth_momentum"].quantile(0.99),
        )

        # 5. Extract the 4-tuple to return
        current_time_label = df.index[-1]
        
        # Get the exact window size needed for the HMM/MCMC
        window = df.iloc[-self.lookback_periods - 1:]
        
        history = window.iloc[:-1]
        live_tick = window.iloc[-1]
        
        historical_price = history[target_price]
        current_tick = float(live_tick[target_price])
        
        live_features_df = window[self.feature_columns]

        return current_time_label, historical_price, current_tick, live_features_df