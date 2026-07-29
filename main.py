import os
import numpy as np
import pandas as pd
import yfinance as yf
import importlib

# 1. Custom Imports
from z_distro import parms_distribution, summary_statistics
from singnal_engine import buy_signal, sell_singnal, calculate_expected_destination
from postion_record import PairPositionRecord, PairsPortfolioManager

# 2. Load the data feed safely
data_feed_module = importlib.import_module('data_feed')
if not hasattr(data_feed_module, 'real_time_streamer'):
    raise ImportError('data_feed.py does not define real_time_streamer; check data_feed.py')
real_time_streamer = data_feed_module.real_time_streamer

# 3. Initialize Portfolio & Variables
portfolio = PairsPortfolioManager(initial_cash=100000.0)
active_pair_state = None
active_trade_ids = {"KO": None}
master_daily_log = []

# 4. Download and prep 15-minute data
# 4. Download and prep 15-minute data
print("Downloading 60-minute market data...")
ko_ticker = yf.Ticker("KO")
df_KO = ko_ticker.history(period="2y", interval="60m")

# Clean the timezone
df_KO.index = df_KO.index.tz_localize(None)

# THE CRITICAL FIX: Rename the 'Open' column to 'KO_Open'
df_KO.rename(columns={'Open': 'KO_Open'}, inplace=True)

# 5. Initialize the stream
# Ensure we are passing 'lookback_periods' to match the updated data_feed.py
print("Starting live simulation stream...")
streamer_feed = real_time_streamer(df_KO, lookback_periods=130)

# 6. Main Simulation Loop
for current_time, historical_price, current_tick in streamer_feed:
    
    # Print the full date and time for the tick
    print(f"[{current_time}] KO Open: {current_tick:.2f}")
    
    # Extract historical values for the MCMC algorithm
    history_vector = historical_price.values 
    
    # Calculate distributions and metrics
    # Note: 500k iterations per tick will take a long time to simulate! 
    prams_distrbatuion = parms_distribution(history_vector, total_iterations=500000, dt=1)
    metrics = summary_statistics(current_tick, prams_distrbatuion)

    # Extract mean parameters (Assuming MCMC returns theta, mu, sigma)
    theta, mu, sigma = np.mean(prams_distrbatuion, axis=0)
    stop_loss_short = mu + (3.5 * sigma)
    stop_loss_long  = mu - (3.5 * sigma)
    z_score = (current_tick - mu) / sigma

    # Generate trading signals
    sell_decision = sell_singnal(portfolio, current_tick, prams_distrbatuion, current_time, z_score)
    buy_decision = buy_signal(portfolio, current_tick, metrics, prams_distrbatuion, current_time, stop_loss_short, stop_loss_long) 

    # Directory management for MCMC backups
    if not os.path.exists('mcmc_samples_backups'):
        os.makedirs('mcmc_samples_backups')
    
    # Add hours and minutes so the files don't overwrite each other intraday
    time_str = current_time.strftime('%Y%m%d_%H%M')
    mcmc_filename = f"mcmc_samples_backups/mcmc_{time_str}.npy"
    np.save(mcmc_filename, prams_distrbatuion)

    # Create the flat row for the CSV
    daily_row = {
        'datetime': current_time,
        'current_price': current_tick,
        # OU Parameters
        'theta': theta,
        'mu': mu,
        'sigma': sigma,
        # Z-Score and Risk
        'z_score': z_score,
        'stop_loss_short': stop_loss_short,
        'stop_loss_long': stop_loss_long,
        # Metrics from summary_statistics()
        'median_z': metrics.get('median_z'),
        'std_zscore': metrics.get('std_zscore'),
        'lower_ci': metrics.get('lower_ci'),
        'upper_ci': metrics.get('upper_ci'),
        'skew_z': metrics.get('skew_z'),
        'kurt_z': metrics.get('kurt_z'),
        # Actions
        'action_bought': buy_decision.get("trade", False),
        'action_sold': sell_decision.get("trade", False),
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