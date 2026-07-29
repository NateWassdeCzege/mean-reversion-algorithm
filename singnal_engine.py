import numpy as np
from scipy.stats import skew, kurtosis, normaltest

def buy_signal(portfolio, current_tick, metrics, mcmc_samples, entry_time, stop_loss_short, stop_loss_long, current_regime):
    """
    Evaluates entry conditions, calculates Kelly sizing, and executes the trade.
    """
    filter_res = trade_filter(metrics,current_regime,prob_threshold=0.75, max_kurtosis=3.0) 
    prob_mu = calculate_expected_destination(current_tick, mcmc_samples, forward_t=None)
    #time_revetion = revete_prob(mcmc_samples, target_prob=85, curxtrent_tick, deltat=1, max_steps=100, num_paths=1000)
    
    if filter_res['trigger_trade']:
        direction = filter_res['direction']
        
        # Determine the correct stop loss based on direction
        stop_loss = stop_loss_long if direction == 'LONG' else stop_loss_short
        
        # Calculate sizing
        ev_metrics = calculate_expected_value(current_tick, mcmc_samples, prob_mu['prob_hitting_target'], stop_loss)
        kelly_fraction = ev_metrics['kelly_fraction']
        dollar_allocation = portfolio.cash * kelly_fraction
        
        # Calculate actual share sizes
        ko_size = int(dollar_allocation / current_tick) if current_tick > 0 else 0
        if ko_size == 0:
            ko_size = 600 # Fallback minimum size
        
        # Execute the trade
        trade_id = portfolio.open_new_pair(
            entry_time, 
            current_tick, 
            ko_size, 
            direction,
        )
        
        # Unified return structure
        return {
            "trade": True,
            "direction": direction,
            "size": ko_size,
            "ev": ev_metrics['expected_value']
        }

    return {"trade": False}


def sell_singnal(portfolio, current_tick, mcmc_samples, exit_time, z_score):
    """
    Evaluates exit conditions based on target HDI density boxes or extreme Z-Score stop losses.
    """
    mcmc_samples = np.array(mcmc_samples)
    mu_samples = mcmc_samples[:, 1]
    mu_lower, mu_upper = calculate_target_zone_hdi(mu_samples, target_mass=0.50)
    lower_bound, upper_bound = np.percentile(mu_samples, [7.5, 92.5])
    width_85_mass = float(upper_bound - lower_bound)
   
    if len(portfolio.active_pairs) > 0 and width_85_mass < 3:
        
        # TAKE PROFIT: The price has successfully returned into the High-Density mean box
        if mu_lower <= current_tick <= mu_upper:
            portfolio.close_all_active_pairs(exit_time, current_tick) 
            print(f"[{exit_time.date()}] TAKE PROFIT: Price {current_tick:.2f} entered target box [{mu_lower:.2f}, {mu_upper:.2f}]")
            return {"trade": True, "reason": "take_profit"}
            
        # STOP LOSS: Z-score blew out past the 3.75 threshold
        '''  if np.abs(z_score) >= 3.5:
            portfolio.close_all_active_pairs(exit_time, current_tick) 
            print(f"[{exit_time.date()}] STOP LOSS: Z-Score hit {z_score:.2f}")
            return {"trade": True, "reason": "stop_loss"}'''
            
    return {"trade": False}


def trade_filter(metrics,current_regime, prob_threshold=0.65, max_kurtosis=3.0):
    """
    Evaluates MCMC distribution metrics to determine trade viability and conviction.
    """
    neg_prob, pos_prob = metrics['z_probe']
    
    is_long_candidate = neg_prob >= prob_threshold
    is_short_candidate = pos_prob >= prob_threshold
    
    threshold_passed = is_long_candidate or is_short_candidate
    no_conflict = not metrics['sign_conflict']
    risk_acceptable = metrics['kurt_z'] < max_kurtosis
    acct_regeim = current_regime != 2
    #acct_regeim = True
    
    trigger_trade = threshold_passed and no_conflict and risk_acceptable and acct_regeim
    conviction_score = 0.0
    direction = "NONE"
    
    if trigger_trade:
        if is_long_candidate:
            direction = "LONG"
            base_score = neg_prob 
            skew_adjustment = metrics['skew_z'] * 0.05
        else:
            direction = "SHORT"
            base_score = pos_prob
            skew_adjustment = -metrics['skew_z'] * 0.05
            
        kurt_penalty = metrics['kurt_z'] * 0.05
        conviction_score = np.clip(base_score + skew_adjustment - kurt_penalty, 0.0, 1.0)

    return {
        "trigger_trade": trigger_trade,
        "direction": direction,
        "conviction_score": round(float(conviction_score), 4),
        "reject_reason": (
            "Threshold Fail" if not threshold_passed else 
            "Sign Conflict" if not no_conflict else 
            "Fat-Tail Risk Too High" if not risk_acceptable else
             "not accepbel regeim" if not acct_regeim else "None"
        )
    }


def calculate_expected_value(current_tick, mcmc_samples, prob_hit_mu, stop_loss):
    """
    Calculates the Expected Value (EV) and time horizon (half-life) of a trade.
    """
    theta, mu, sigma = np.mean(mcmc_samples, axis=0)
    
    half_life = np.log(2) / theta 
    
    potential_reward = np.abs(current_tick - mu)
    potential_risk = np.abs(current_tick - stop_loss)

    ev = (potential_reward * prob_hit_mu) - (potential_risk * (1.0 - prob_hit_mu))
    
    # Avoid division by zero if risk is 0
    if potential_risk == 0:
        b = 0
    else:
        b = potential_reward / potential_risk

    if b == 0:
        kelly_fraction = 0
    else:
        kelly_fraction = (prob_hit_mu * b - (1.0 - prob_hit_mu)) / b
        
    kelly_fraction = max(0, kelly_fraction)
    
    return {
        "expected_value": round(float(ev), 4),
        "kelly_fraction": kelly_fraction,
        "target_price": round(float(mu), 4),
        "reward_distance": potential_reward,
        "risk_distance": potential_risk
    }


def calculate_expected_destination(current_tick, mcmc_samples, forward_t=None):
    """
    Calculates the expected price distribution at a fixed future time 'forward_t'.
    """
    mcmc_samples = np.array(mcmc_samples)
    
    theta = mcmc_samples[:, 0]
    mu = mcmc_samples[:, 1]
    
    if forward_t is None:
        mean_theta = np.mean(theta)
        forward_t = (np.log(2) / mean_theta) * 2
        
    hdi_lower, hdi_upper = calculate_target_zone_hdi(mu, target_mass=0.50)

    decay = np.exp(-theta * forward_t)
    expected_prices = (current_tick * decay) + (mu * (1.0 - decay))
    
    highest_density_price = np.median(expected_prices)
    
    if current_tick > hdi_upper:
        prob_hitting_target = np.mean(expected_prices <= hdi_upper)
    elif current_tick < hdi_lower:
        prob_hitting_target = np.mean(expected_prices >= hdi_lower)
    else:
        prob_hitting_target = np.mean((expected_prices >= hdi_lower) & (expected_prices <= hdi_upper))

    return {
        "future_time": forward_t,
        "most_likely_price": highest_density_price,
        "prob_hitting_target": round(float(prob_hitting_target), 4),
        "expected_price_dist": expected_prices, 
        "target_theta": theta,
        "hdi_lower": hdi_lower,
        "hdi_upper": hdi_upper
    }


def calculate_target_zone_hdi(mu_samples, target_mass=0.50):
    """
    Finds the exact upper and lower price bounds where the spread wants to return to.
    """
    
    total_samples = len(mu_samples)
    
    sorted_mu = np.sort(mu_samples)
    interval_count = int(np.floor(target_mass * total_samples))
    
    interval_widths = sorted_mu[interval_count:] - sorted_mu[:-interval_count]
    min_idx = np.argmin(interval_widths)
    
    hdi_lower = sorted_mu[min_idx]
    hdi_upper = sorted_mu[min_idx + interval_count]
    
    return hdi_lower, hdi_upper
import numpy as np

def calculate_reversion_metrics(mcmc_samples, xt, deltat=1, max_steps=50, num_paths=2000):
    """
    Simulates O-U paths to find the probability of reverting to the HDI target zone
    within a fixed timeframe (max_steps), accounting for theta, mu, and sigma.
    """
    mcmc_samples = np.array(mcmc_samples)
    
    # 1. Sample parameter sets from the posterior distribution
    indices = np.random.choice(len(mcmc_samples), size=num_paths, replace=True)
    theta_samples = mcmc_samples[indices, 0]
    mu_samples = mcmc_samples[indices, 1]
    sigma_samples = mcmc_samples[indices, 2]
    
    # 2. Get the target zone bounds
    lower_hdi, upper_hdi = calculate_target_zone_hdi(mu_samples, target_mass=0.50)
    
    # 3. Initialize paths
    current_prices = np.full(num_paths, float(xt))
    hit_mask = np.zeros(num_paths, dtype=bool) # Tracks paths that successfully entered the zone
    first_hit_times = np.full(num_paths, max_steps, dtype=float) # Tracks time-to-revert
    
    for step in range(1, max_steps + 1):
        # Vectorized Euler-Maruyama step (includes sigma!)
        z = np.random.normal(0, 1, size=num_paths)
        current_prices = current_prices + theta_samples * (mu_samples - current_prices) * deltat + sigma_samples * np.sqrt(deltat) * z
        
        # Check which paths have hit the zone for the first time
        in_zone = (current_prices >= lower_hdi) & (current_prices <= upper_hdi)
        new_hits = in_zone & (~hit_mask)
        
        first_hit_times[new_hits] = step
        hit_mask = hit_mask | in_zone

    # Calculate final metrics for position sizing
    prob_revert = np.sum(hit_mask) / num_paths
    avg_time_to_revert = np.mean(first_hit_times[hit_mask]) if np.any(hit_mask) else max_steps

    return {
        "prob_reverting": round(float(prob_revert), 4),
        "expected_steps_to_revert": round(float(avg_time_to_revert), 1),
        "lower_hdi": lower_hdi,
        "upper_hdi": upper_hdi
    }

import numpy as np

def calculate_ev_and_shares(current_tick, mcmc_samples, total_capital, max_risk_pct=0.02, max_steps=50):
    """
    Calculates Expected Value (EV) and optimal share sizing based purely 
    on the Monte Carlo reversion probability and risk parameters.
    """
    # 1. Run the Monte Carlo reversion metrics
    metrics = calculate_reversion_metrics(mcmc_samples, current_tick, deltat=1, max_steps=max_steps, num_paths=2000)
    
    prob_revert = metrics["prob_reverting"]
    lower_hdi = metrics["lower_hdi"]
    upper_hdi = metrics["upper_hdi"]
    
    mean_sigma = np.mean(mcmc_samples[:, 2])
    target_price = (lower_hdi + upper_hdi) / 2.0
    
    # 2. Determine potential gain and loss based on price position relative to HDI
    if current_tick < lower_hdi:
        potential_gain = target_price - current_tick
        potential_loss = 3.5 * mean_sigma
    elif current_tick > upper_hdi:
        potential_gain = current_tick - target_price
        potential_loss = 3.5 * mean_sigma
    else:
        # Price is inside the equilibrium zone; no edge
        return {
            "expected_value": 0.0,
            "recommended_shares": 0.0,
            "current_price": round(current_tick, 2)
        }
        
    # 3. Calculate Expected Value (EV) per unit
    ev_per_unit = (prob_revert * potential_gain) - ((1.0 - prob_revert) * potential_loss)
    
    # 4. Calculate shares based on risk budget and confidence scaling
    if ev_per_unit <= 0 or potential_loss <= 0:
        recommended_shares = 0.0
    else:
        risk_budget = total_capital * max_risk_pct
        base_shares = risk_budget / potential_loss
        # Scale position size by the model's confidence probability
        recommended_shares = base_shares * prob_revert
        
    # Ensure share allocation never exceeds available capital
    max_shares_by_cash = total_capital / current_tick
    final_shares = min(recommended_shares, max_shares_by_cash)
    
    return {
        "expected_value": round(ev_per_unit, 4),
        "recommended_shares": round(final_shares, 2),
        "current_price": round(current_tick, 2)
    }