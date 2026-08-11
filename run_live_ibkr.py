import os
import numpy as np
import pandas as pd
import importlib
import joblib
from ib_insync import IB, Stock, util

# 1. Custom Imports
from src.z_distro import parms_distribution, summary_statistics
from signal_engine import buy_signal, sell_singnal
from position_record import PairsPortfolioManager

# 2. Load the data feed safely (Import the new Class)
data_feed_module = importlib.import_module('data_feed_live')
if not hasattr(data_feed_module, 'RealTimeFeatureEngine'):
    raise ImportError('data_feed_live.py does not define RealTimeFeatureEngine; check data_feed_live.py')
RealTimeFeatureEngine = data_feed_module.RealTimeFeatureEngine

# 3. Load HMM BEFORE starting the stream
print("Loading HMM pipeline...")
hmm_pipeline = joblib.load('HMM/ko_hmm_final_pipeline2.pkl')
hmm_model = hmm_pipeline['model']
scaler = hmm_pipeline['scaler']

# 4. Initialize Portfolio & Variables
portfolio = PairsPortfolioManager(initial_cash=100000.0)
master_daily_log = []

log_folder = "trading_logs"
if not os.path.exists(log_folder):
    os.makedirs(log_folder)
if not os.path.exists('mcmc_samples_backups'):
    os.makedirs('mcmc_samples_backups')

# =====================================================================
# 5. CONNECT TO INTERACTIVE BROKERS
# =====================================================================
print("Connecting to Interactive Brokers...")
ib = IB()
ib.connect('127.0.0.1', 7497, clientId=1)

ko_contract = Stock('KO', 'SMART', 'USD')

# 6. Download initial historical data to "warm up" the indicators
print("Downloading 30-day warmup data...")
bars = ib.reqHistoricalData(ko_contract, '', '30 D', '1 hour', 'TRADES', True)
df_warmup = util.df(bars)
df_warmup.set_index('date', inplace=True)

# Ensure index is timezone naive
df_warmup.index = pd.to_datetime(df_warmup.index).tz_localize(None)

df_warmup.rename(columns={
    'open': 'KO_Open', 'high': 'High', 'low': 'Low', 'close': 'Close', 'volume': 'Volume'
}, inplace=True)

# Instantiate the Engine
print("Warming up indicators...")
feature_engine = RealTimeFeatureEngine(df_warmup, lookback_periods=130)

# =====================================================================
# 7. DEFINE THE LIVE TRADING CALLBACK
# =====================================================================
def on_new_bar(bars, hasNewBar):
    if hasNewBar:
        latest_bar = bars[-1]
        
        # IBKR returns timezone-aware datetimes. Strip it for consistency with your old code.
        new_timestamp = latest_bar.date.replace(tzinfo=None)
        
        new_bar_data = {
            'KO_Open': latest_bar.open,
            'High': latest_bar.high,
            'Low': latest_bar.low,
            'Close': latest_bar.close,
            'Volume': latest_bar.volume
        }
        
        print(f"\n--- Processing Live Tick at: {new_timestamp} ---")
        
        # 7a. Get updated features from the engine
        current_date, historical_price, current_tick, live_features_df = feature_engine.process_new_bar(new_timestamp, new_bar_data)
        
        # 7b. HMM Prediction
        valid_features_df = live_features_df.dropna()
        if valid_features_df.empty:
            print("Skipping tick: Not enough data for indicators yet.")
            return
            
        scaled_features = scaler.transform(valid_features_df)
        current_regime = int(hmm_model.predict(scaled_features)[-1])
        print('Current Regime is:', current_regime)
        
        # 7c. MCMC & Statistical Math
        history_vector = historical_price.values 
        
        
        prams_distrbatuion = parms_distribution(history_vector, total_iterations=500000, dt=1)
        metrics = summary_statistics(current_tick, prams_distrbatuion)

        theta, mu, sigma = np.median(prams_distrbatuion, axis=0)
        z_score = np.median(metrics['z_dist'])

        # 7d. Signal Generation
        sell_decision = sell_singnal(portfolio, current_tick, prams_distrbatuion, current_date, metrics)

        current_calendar_date = pd.to_datetime(current_date).date()
        active_bought_today = False

        if portfolio.active_pairs:
            last_trade = list(portfolio.active_pairs.values())[-1]
            active_bought_today = (pd.to_datetime(last_trade.entry_time).date() == current_calendar_date)

        if not active_bought_today:
            buy_decision = buy_signal(portfolio, current_tick, metrics, prams_distrbatuion, current_date, current_regime)
        else:
            buy_decision = {"trade": False}

        # 7e. Log Data
        time_str = current_date.strftime('%Y%m%d_%H%M')
        mcmc_filename = f"mcmc_samples_backups/mcmc_{time_str}.npy"
        np.save(mcmc_filename, prams_distrbatuion)

        daily_row = {
            'date': current_date,
            'current_price': current_tick,
            'market_regime': current_regime,
            'theta': theta,
            'mu': mu,
            'sigma': sigma,
            'z_score': z_score,
            'median_z': metrics.get('median_z'),
            'std_zscore': metrics.get('std_zscore'),
            'lower_ci': metrics.get('lower_ci'),
            'upper_ci': metrics.get('upper_ci'),
            'skew_z': metrics.get('skew_z'),
            'kurt_z': metrics.get('kurt_z'),
            'ev': buy_decision.get('ev', 0.0) if isinstance(buy_decision, dict) else 0.0,
            'action_bought': buy_decision.get("trade", False) if isinstance(buy_decision, dict) else False,
            'action_sold': sell_decision.get("trade", False) if isinstance(sell_decision, dict) else False,
            'cash_balance': portfolio.cash,
            'raw_data_path': mcmc_filename 
        }
            
        master_daily_log.append(daily_row)
        
        # 7f. Save Progressively
        df_final = pd.DataFrame(master_daily_log)
        df_final.to_csv(f"{log_folder}/master_daily_log.csv", index=False)
        portfolio.export_trade_log(f"{log_folder}/trade_log.csv")
        print("Tick processed successfully and logs updated.")

# =====================================================================
# 8. START THE LIVE STREAM
# =====================================================================
print("Starting live simulation stream...")
live_bars = ib.reqHistoricalData(ko_contract, '', '1 D', '1 hour', 'TRADES', True, keepUpToDate=True)
live_bars.updateEvent += on_new_bar

try:
    ib.run() 
except KeyboardInterrupt:
    print("\nManual interruption detected. Stopping stream and disconnecting from IBKR...")
finally:
    ib.disconnect()
    print("Disconnected.")