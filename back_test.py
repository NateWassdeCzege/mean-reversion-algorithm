import os
import numpy as np
import pandas as pd
import yfinance as yf
import importlib
import joblib
# 1. Custom Imports
from src.z_distro import parms_distribution, summary_statistics
from signal_engine import buy_signal, sell_singnal
from position_record import PairsPortfolioManager

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

# 4. Download and prep 60-minute data
print("Downloading 60-minute market data...")
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
print("Starting live simulation stream...")
streamer_feed = real_time_streamer(df_KO, lookback_periods=130)

# 6. Main Simulation Loop
for current_date, historical_price, current_tick,live_features_df in streamer_feed:
    print(f"\n--- Processing Live Trading Day: {current_date.strftime('%Y-%m-%d')} ---")

# 1. Drop the NaN rows caused by indicator warm-up periods
    valid_features_df = live_features_df.dropna()
    
    # 2. Safety check: If the whole window is NaNs, skip this tick
    if valid_features_df.empty:
        continue
        
    # 3. Scale and predict on the valid sequence
    scaled_features = scaler.transform(valid_features_df)
    
    current_regime = int(hmm_model.predict(scaled_features)[-1])
    print('reagemi is ' , current_regime)
    
    

    
    # Extract historical values for the MCMC algorithm
    history_vector = historical_price.values 
    
    # Calculate distributions and metrics
    prams_distrbatuion = parms_distribution(history_vector, total_iterations=500000, dt=1)
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

    # Directory management for MCMC backups
    if not os.path.exists('mcmc_samples_backups'):
        os.makedirs('mcmc_samples_backups')
    
    
    time_str = current_date.strftime('%Y%m%d_%H%M')
    mcmc_filename = f"mcmc_samples_backups/mcmc_{time_str}.npy"
    np.save(mcmc_filename, prams_distrbatuion)

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
print("Simulation complete. Saving logs...")

# Create the folder for your logs if it doesn't exist
log_folder = "trading_logs"
if not os.path.exists(log_folder):
    os.makedirs(log_folder)

# Save the files inside the new folder
df_final = pd.DataFrame(master_daily_log)
df_final.to_csv(f"{log_folder}/master_daily_log.csv", index=False)
portfolio.export_trade_log(f"{log_folder}/trade_log.csv")

print(f"Saved master_daily_log.csv and trade_log.csv inside the '{log_folder}' directory.")