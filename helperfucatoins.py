import numpy as np
import scipy.stats as stats

def revete_prob(mcmc_samples, target_prob, xt, deltat, max_steps=100, num_paths=1000):
    """
    Uses Monte Carlo simulation of the Euler-Maruyama discretization of an 
    Ornstein-Uhlenbeck process to calculate the expected time to hit the target zone.
    """
    mcmc_samples = np.array(mcmc_samples)
    
    # Randomly select parameter sets from MCMC samples to run paths
    indices = np.random.choice(len(mcmc_samples), size=num_paths, replace=True)
    theta_samples = mcmc_samples[indices, 0]
    mu_samples = mcmc_samples[indices, 1]
    sigma_samples = mcmc_samples[indices, 2]
    
   
    lower_hdi, upper_hdi = calculate_target_zone_hdi(mcmc_samples[:, 1], target_mass=0.50)
    
    # Initialize paths
    current_prices = np.full(num_paths, float(xt))
    hit_mask = np.zeros(num_paths, dtype=bool) # Tracks which paths have hit the zone
    
    time_step = 0
    current_prob = 0.0
    
    while current_prob < target_prob and time_step < max_steps:
        # Simulate one step forward for ALL paths simultaneously (Vectorized)
        current_prices = expectxt_vectorized(current_prices, deltat, theta_samples, mu_samples, sigma_samples)
        
        # Check which paths have newly entered the HDI zone
        # (Fixing the chained comparison bug: lower_hdi <= prices <= upper_hdi)
        in_zone = (current_prices >= lower_hdi) & (current_prices <= upper_hdi)
        hit_mask = hit_mask | in_zone 
        
        current_prob = np.sum(hit_mask) / num_paths
        time_step += 1
        
    return time_step, current_prob

def expectxt_vectorized(xt, deltat, theta, mu, sigma):
    """
    Euler-Maruyama discretization of the Ornstein-Uhlenbeck SDE.
    Vectorized to run thousands of paths simultaneously.
    """
    # Fix: SDEs require a standard normal distribution N(0,1), NOT a uniform random (-1, 1)
    z = np.random.normal(0, 1, size=len(xt)) 
    
    # Fix: The square root is ONLY applied to time (deltat), not to 'z'
    return xt + theta * (mu - xt) * deltat + sigma * np.sqrt(deltat) * z

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