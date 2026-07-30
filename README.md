content = """# Quantitative Mean Reversion Trading Algorithm

A production-grade, mathematically rigorous statistical arbitrage trading system grounded in stochastic calculus. This project bridges theoretical stochastic differential equations with applied quantitative finance, featuring strict risk management protocols and dynamic regime filtering.

## Overview

Traditional financial analysis often overlooks short-term market dynamics, which can be effectively modeled using stochastic processes. This project implements a Mean-Reversion Strategy using the Ornstein-Uhlenbeck (OU) process, backed by Markov Chain Monte Carlo (MCMC) parameter estimation and a Hidden Markov Model (HMM) market regime filter. 

Unlike static trading bots, this system continuously evaluates parameter uncertainty, filters out hazardous trending markets, and sizes positions dynamically based on expected value calculations.

---

## Core Architecture & Methodology

The system operates on a dual-track integration architecture that processes hourly OHLCV data to make statistically grounded execution decisions:

### 1. Stochastic Modeling: The Ornstein-Uhlenbeck (OU) Process
To capture mean-reverting equity behavior, asset price dynamics are modeled via the stochastic differential equation (SDE):

$$dx_{t} = \theta(\mu - x_{t})dt + \sigma dW_{t}$$

* **Deterministic drift** ($\theta(\mu - x_{t})dt$) pulls the price back toward the long-term equilibrium mean ($\mu$) at a reversion speed ($\theta$).
* **Stochastic diffusion** ($\sigma dW_{t}$) captures market volatility ($\sigma$) and unpredictable random noise via a Wiener process ($dW_{t}$).

### 2. Parameter Estimation via MCMC
To overcome the limitations of static Maximum Likelihood Estimation (MLE), the model applies a Markov Chain Monte Carlo (MCMC) Metropolis-Hastings algorithm over a rolling 130-tick window (approx. 21 trading days). This generates data-driven posterior probability distributions for the parameter set $\Theta = [\theta, \mu, \sigma]$, effectively accounting for statistical uncertainty.

### 3. Real-Time Regime Classification (HMM)
To protect capital from catastrophic drawdowns during aggressive structural trends, the framework incorporates a Hidden Markov Model (HMM) trained via the Baum-Welch (Forward-Backward) algorithm. 
* **Feature Vector ($O_t$)**: Combines Log Returns, Smooth Momentum, Average Directional Index (ADX), and Moving Average Spread.
* **Viterbi Decoding**: Decodes hidden market states in real time, completely muting trading signals during unfavorable trending regimes to prevent adverse selection.

---

## Risk Management & Execution Logic

* **Probabilistic Signal Filtering**: Evaluates the Cumulative Distribution Function (CDF) of MCMC outputs, requiring $\ge 75\%$ probability mass beyond target Z-score thresholds ($\ge 2$ or $\le -2$) alongside strict kurtosis and sign-consensus checks.
* **Path Simulation & Expected Value (EV)**: Simulates 2,000 discrete price paths using Euler-Maruyama discretization over a 50-period horizon to compute the empirical reversion probability ($P_{\text{revert}}$) and ensure positive trade expectancy.
* **Confidence-Scaled Allocation**: Caps maximum trade exposure at 2% of total portfolio capital, linearly scaling base position sizes using model confidence.
* **Dynamic Exits & Stop-Losses**:
  * Take-profit signals are triggered dynamically when prices enter the 50% Highest Density Interval (HDI) of the expected mean.
  * Dual-layered stop-losses include a 50-tick time-stop and a structural blowout stop ($Z \ge 3.75$).
## Academic Paper
For an in-depth mathematical breakdown, full system architecture schematics, and rigorous evaluation metrics, please refer to the complete research paper included in the repository: mean_revert_paper.ipynb.
---

## Project Structure

```text
├── .gitignore
├── HMM_main.py                  # Hidden Markov Model regime classification
├── LICENSE
├── README.md
├── data_feed.py                 # Market data ingestion stream
├── helper_functions.py          # Utility and helper functions
├── ko_hmm_final_pipeline2.pkl   # Serialized HMM pipeline model
├── main.py                      # Main execution entry point
├── mean_revert_paper.ipynb      # Detailed mathematical documentation (Colab PDF)
├── position_record.py           # Position recording and management
├── requirements.txt             # Python dependencies
├── signal_engine.py             # Signal generation and trade filtering engine
└── z_distro.py                  # Z-score and distribution calculations






