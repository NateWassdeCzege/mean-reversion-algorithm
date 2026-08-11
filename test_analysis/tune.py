import os
import joblib
import numpy as np
import pandas as pd
import yfinance as yf
import importlib

# 1. Custom Imports
from src.z_distro import parms_distribution, summary_statistics
from signal_engine import buy_signal, sell_singnal
from position_record import PairPositionRecord, PairsPortfolioManager

# 2. Load the data feed safely
data_feed_module = importlib.import_module('data_feed')
if not hasattr(data_feed_module, 'real_time_streamer'):
    raise ImportError('data_feed.py does not define real_time_streamer; check data_feed.py')
real_time_streamer = data_feed_module.real_time_streamer2

# 3. Initialize Portfolio & Variables
portfolio = PairsPortfolioManager(initial_cash=100000.0)
active_pair_state = None
active_trade_ids = {"KO": None}
master_daily_log = []

# 4. Download and prep data
ko_ticker = yf.Ticker("KO")
df_KO = ko_ticker.history(period="2y", interval="60m")

# Clean the timezone
df_KO.index = df_KO.index.tz_localize(None)


df_KO.rename(columns={'Open': 'KO_Open'}, inplace=True)
#load HMM
print("Loading HMM pipeline...")
hmm_pipeline = joblib.load('HMM/ko_hmm_final_pipeline2.pkl')
hmm_model = hmm_pipeline['model']
scaler = hmm_pipeline['scaler']

# 5. Initialize the stream
# Ensure we are passing 'lookback_periods' to match the updated data_feed.py
print("Starting live simulation stream...")
streamer_feed = real_time_streamer(df_KO, lookback_periods=130)
print("Starting live simulation stream...")

# 6. Main Simulation Loop
for current_date, historical_price, current_tick , live_features_df in streamer_feed:
    
    print(f"\n--- Processing Live Trading Day: {current_date.strftime('%Y-%m-%d')} ---")
    portfolio.update_time_step()
# 1. Drop the NaN rows caused by indicator warm-up periods
    valid_features_df = live_features_df.dropna()
    
    # 2. Safety check: If the whole window is NaNs, skip this tick
    if valid_features_df.empty:
        continue
        
    # 3. Scale and predict on the valid sequence
    scaled_features = scaler.transform(valid_features_df)
    
    # Optional print to verify it's working (you can remove this later)
    # print('features are', scaled_features)
    
    current_regime = int(hmm_model.predict(scaled_features)[-1])
    print('reagemi is ' , current_regime)
    # Construct the filename to load the pre-calculated MCMC distribution
    # Note: Make sure this directory matches where you saved them! (e.g., 'mcmc_samples_backups' or 'mcmc_backups')
    mcmc_filename = f"mcmc_samples_backups/mcmc_{current_date.strftime('%Y%m%d')}.npy"
    #mcmc_filename = f"pymc_mcmc_backups/mcmc_{current_date.strftime('%Y%m%d_%H%M')}.npy"
    
    # Check if the backup file actually exists before trying to load it
    if not os.path.exists(mcmc_filename):
        print(f"Warning: {mcmc_filename} not found. Skipping this day.")
        continue
        
    # Load the heavy data directly from disk instead of calculating it
    prams_distrbatuion = np.load(mcmc_filename)
    
    # Calculate your standard metrics using the loaded distribution
    metrics = summary_statistics(current_tick, prams_distrbatuion)

    # Extract mean parameters
    theta, mu, sigma = np.median(prams_distrbatuion, axis=0)
    z_score = np.median(metrics['z_dist'])

    # Generate trading signals (Single asset arguments)
    sell_decision = sell_singnal(portfolio, current_tick, prams_distrbatuion, current_date, metrics)
    

    current_calendar_date = pd.to_datetime(current_date).date()
    active_bought_today = False

    if portfolio.active_pairs:
        # Get the single most recent active trade
        last_trade = list(portfolio.active_pairs.values())[-1]
        active_bought_today = (pd.to_datetime(last_trade.entry_time).date() == current_calendar_date)

    if not active_bought_today:
        buy_decision = buy_signal(portfolio, current_tick, metrics, prams_distrbatuion, current_date, current_regime)
    else:
        buy_decision = {"trade": False}

    # Create the flat row for the CSV
    daily_row = {
        'date': current_date,
        'current_price': current_tick,
        'market_regime': current_regime,
        # OU Parameters
        'theta': theta,
        'mu': mu,
        'sigma': sigma,
        # Z-Score and Risk
        'z_score': z_score,
        # Metrics from summary_statistics()
        'median_z': metrics.get('median_z'),
        'std_zscore': metrics.get('std_zscore'),
        'lower_ci': metrics.get('lower_ci'),
        'upper_ci': metrics.get('upper_ci'),
        'skew_z': metrics.get('skew_z'),
        'kurt_z': metrics.get('kurt_z'),

        # Actions
        'ev': buy_decision.get('ev', 0.0) if isinstance(buy_decision, dict) else 0.0,
        'action_bought': buy_decision.get("trade", False) if isinstance(buy_decision, dict) else False,
        'action_sold': sell_decision.get("trade", False) if isinstance(sell_decision, dict) else False,
        # Financials
        'cash_balance': portfolio.cash,
        'raw_data_path': mcmc_filename 
    }
        
    master_daily_log.append(daily_row)

# 7. Convert to DataFrame and Save
print("Simulation complete. Saving fast logs...")

# Create the folder for your logs if it doesn't exist
log_folder = "trading_logs"
if not os.path.exists(log_folder):
    os.makedirs(log_folder)

# Save the files inside the new folder
df_final = pd.DataFrame(master_daily_log)
df_final.to_csv(f"{log_folder}/master_daily_log_fast.csv", index=False)
portfolio.export_trade_log(f"{log_folder}/trade_log_fast.csv")

print(f"Saved master_daily_log_fast.csv and trade_log_fast.csv inside the '{log_folder}' directory.")