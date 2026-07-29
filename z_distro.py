
# thick later to add inter day stock day chaign dt and xt to be cdaeilssick insated 
import yfinance as yf
import numpy as np
import pandas as pd
import random
import numpy as np
from scipy.stats import gaussian_kde
import numpy as np
from scipy.stats import skew, kurtosis

# thick later to add inter day stock day chaign dt and xt to be cdaeilssick insated 
def parms_distribution(xt, total_iterations=1000000,dt=1): # xtis stock data dt is the diffrece between the current day and the next day
 
    from sklearn.linear_model import LinearRegression

    def get_initial_guesses(xt_vals, dt=1):
        # Slice data into t and t+1
        x_current = xt_vals[:-1].reshape(-1, 1)
        x_next = xt_vals[1:]
        
        # Run quick linear regression: X_{t+1} = slope * X_t + intercept
        reg = LinearRegression().fit(x_current, x_next)
        slope = reg.coef_[0]
        intercept = reg.intercept_
        
        # Calculate residuals to get empirical volatility
        predictions = reg.predict(x_current)
        residuals = x_next - predictions
        residual_std = np.std(residuals)
        
        # Translate AR(1) regression coefficients into OU parameters
        # 1. Theta (speed of mean reversion)
        if slope > 0 and slope < 1:
            theta_start = -np.log(slope) / dt
        else:
            theta_start = 0.1  # Fallback if data isn't mean-reverting today
            
        # 2. Mu (long-term mean)
        if slope != 1:
            mu_start = intercept / (1 - slope)
        else:
            mu_start = np.mean(xt_vals) # Fallback to simple average
            
        # 3. Sigma (volatility)
        sigma_start = residual_std / np.sqrt(dt)
        
        # Quick safety guardrails to ensure positive values
        theta_start = max(theta_start, 0.001)
        sigma_start = max(sigma_start, 0.001)
        
        return theta_start, mu_start, sigma_start

    def ou_log_likelihood(x_current, x_next, dt, theta, mu, sigma): # Current is xt so the lcsoeing precei on the dat next isn the closing price of the next day
        # Enforce strict positive constraints
        if theta <= 0 or sigma <= 0:
            return -float('inf')
            
        expected_mean = x_current * np.exp(-theta * dt) + mu * (1 - np.exp(-theta * dt))
        expected_var = (sigma**2 / (2 * theta)) * (1 - np.exp(-2 * theta * dt))
        
        # Catch tiny float or negative variance edge cases
        if expected_var <= 0:
            return -float('inf')
            
        term1 = -0.5 * np.log(2 * np.pi * expected_var)
        term2 = -((x_next - expected_mean) ** 2) / (2 * expected_var)

        return np.sum(term1 + term2)


    def run_mcmc(current_vector, total_iterations, x_current_series, x_next_series, time_step):# you will ocntuiers updat the cuccuten adn next saera to run the mcmc on
        samples = []
        current_state = list(current_vector)
        
        # Pre-calculate baseline score
        current_score = ou_log_likelihood(
            x_current_series, x_next_series, time_step, 
            current_state[0], current_state[1], current_state[2] # mu theta and sigma
        )
        
        for i in range(total_iterations):
            # Generate micro-scaled steps matched to each parameter's baseline size
            step_theta = random.uniform(-0.0007, 0.0007)
            step_mu    = random.uniform(-0.1, 0.1)
            step_sigma = random.uniform(-0.005, 0.005)
            
            proposed_state = [
                current_state[0] + step_theta,
                current_state[1] + step_mu,
                current_state[2] + step_sigma
            ]
            
            # Boundary constraints check to prevent code crashes before running likelihood math
            if proposed_state[0] <= 0 or proposed_state[2] <= 0:
                proposed_score = -float('inf')
            else:
                proposed_score = ou_log_likelihood(
                    x_current_series, x_next_series, time_step, 
                    proposed_state[0], proposed_state[1], proposed_state[2]
                )
                
            # Metropolis-Hastings acceptance step for log-likelihood scales
            if proposed_score == -float('inf'):
                acceptance_ratio = 0.0
            else:
                # Subtraction in log space represents division in raw probability space
                acceptance_ratio = np.exp(proposed_score - current_score)
                
            if random.random() < acceptance_ratio:
                current_state = proposed_state
                current_score = proposed_score
                
            samples.append(list(current_state))
            
        # Return everything after the 10% burn-in period to guarantee equilibrium
        burn_in_index = int(total_iterations * 0.10)
        return samples[burn_in_index:]



    
    
    # file will take in integation data form windoe in a vectore list of day to day sotkc preices
    # Normalize input to a pandas Series then to a NumPy array
    xt_series = pd.Series(xt).dropna()
    xt_vals = xt_series.values

    # Paired arrays of equal length
    x_current_series = xt_vals[:-1]
    x_next_series = xt_vals[1:]

    x_t1_data = x_next_series
    xt_data = x_current_series

    # 5. Initialize Parameters and Set Data Window
    # Starting points from your initial regression
    theta_start, mu_start, sigma_start = get_initial_guesses(xt_vals, dt=1)  # calulated from the last 30 day window or the last regeim window as an intal geuses tehn tun ethem with mcmc
    initial_vector = [theta_start, mu_start, sigma_start]

    mcmc_samples = run_mcmc(initial_vector, total_iterations, xt_data, x_t1_data, dt)
    
    return mcmc_samples 





# chekc ot amke sure there not clainfianting data with usign differ out agl fo mcmc adn zcore

def summary_statistics(current_tick, mcmc_samples):
    """
    Calculates the full distribution profile of Z-scores from MCMC samples.
    """
    # 1. Generate the Z-score distribution
    # Inside your trading loop:
    safe_regime = is_safe_to_trade(mcmc_samples)


    z_distribution = []
    for sample in mcmc_samples:
        theta, mu, sigma = sample 
        z = ou_zscore(current_tick, theta, mu, sigma)
        z_distribution.append(z)
        
    z_dist = np.array(z_distribution)
    
    # 2. Vectorized Tail Probabilities (Fixed NumPy syntax)
    negative_prob = np.mean(z_dist <= -2)  # P(Z <= -2)
    positive_prob = np.mean(z_dist >= 2)   # P(Z >= 2)
    
    # 3. Core Metrics
    median_z = np.median(z_dist)
    lower_ci = np.percentile(z_dist, 5)
    upper_ci = np.percentile(z_dist, 95)
    std_zscore = np.std(z_dist)
    
    # 4. Metrics for Trading Edge
    skew_z = skew(z_dist)      
    kurt_z = kurtosis(z_dist)  
    
    # NOTE: Define your sign_conflict logic here so it doesn't throw a NameError
    # Example placeholder:
    sign_conflict = float(median_z > 0) != float(current_tick > 0) 
  
    metrics = {
        'z_probe': (negative_prob, positive_prob), 
        "z_dist": z_dist,
        "median_z": median_z,
        "lower_ci": lower_ci,
        "upper_ci": upper_ci,
        "std_zscore": std_zscore,
        "skew_z": skew_z,
        "kurt_z": kurt_z,
        "sign_conflict": sign_conflict,
        "safe_regime" : safe_regime
    }
    
    return metrics

def ou_zscore(current_tick, theta, mu, sigma):
    """
    Calculates the z-score for an Ornstein-Uhlenbeck process.
    """
    return (current_tick - mu) / np.sqrt((sigma**2) / (2 * theta))

def is_safe_to_trade(mcmc_samples):
    # mcmc_samples is your daily array of [Theta, Mu, Sigma]
    df_mcmc = pd.DataFrame(mcmc_samples, columns=['Theta', 'Mu', 'Sigma'])
    
    # 1. Calculate the critical metrics
    theta_median = df_mcmc['Theta'].median()
    mu_std = df_mcmc['Mu'].std()
    
    # 2. Apply the filters based on our historical profiling
    if theta_median < 0.10:
        return False
        
    if mu_std > 12.0:
        return False
        
    return True


