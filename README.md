# Quantitative Mean Reversion Trading Algorithm

A production-grade, mathematically rigorous statistical arbitrage trading system grounded in stochastic calculus. This project bridges theoretical stochastic differential equations with applied quantitative finance, featuring strict risk management protocols and dynamic regime filtering.

## Overview

Traditional financial analysis often overlooks short-term market dynamics, which can be effectively modeled using stochastic processes. This project implements a Mean-Reversion Strategy using the Ornstein-Uhlenbeck (OU) process, backed by Markov Chain Monte Carlo (MCMC) parameter estimation and a Hidden Markov Model (HMM) market regime filter. 

Unlike static trading bots, this system continuously evaluates parameter uncertainty, filters out hazardous trending markets, and sizes positions dynamically based on expected value calculations.

## Backtest Performance & Results

Over the one-year backtest period, the mean-reverting algorithm successfully captured an **11.28% annual profit**. The system demonstrated highly efficient capital allocation, characterized by an exceptional profit factor and strictly managed downside risk. 

### Key Performance Metrics

**Return & Efficiency**
* **Win Rate:** 58.2% (across 18 total trades)
* **Profit Factor:** 4.50
* **Expectancy:** +0.97% per trade
* **Sharpe Ratio:** 1.17

**Risk Management**
* **Average Win:** +2.33%
* **Average Loss:** -0.71%
* **Max Drawdown:** -1.30%
* **Average Time in Market:** 10 Days, 18 Hours
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

---

## Academic Paper

<small>For an in-depth mathematical breakdown, full system architecture schematics, and rigorous evaluation metrics, please refer to the [complete research paper](paper_report.pdf) included in the repository.</small>

---

> **Disclaimer:** This model is for educational and informational purposes only. Using this model does not guarantee financial profit or returns. All trading and investments carry inherent risk. Please use this tool at your own discretion and risk.

## Project Structure
```text
single_mean_rev/
├── mcmc_samples_backups/     # Saved MCMC parameter chain sampling backups
├── models/                   # Serialized HMM pipelines and model artifacts (.pkl)
├── src/                      # Core trading pipeline modules
│   ├── data_feed_live.py     # Real-time data streaming interface
│   ├── data_feed.py          # Historical data ingestion stream (yfinance)
│   ├── hmm_main.py           # Hidden Markov Model regime classification logic
│   ├── position_record.py    # Portfolio position tracking and trade management
│   ├── signal_engine.py      # Mean-reversion signal generation and regime filtering
│   └── z_distro.py           # Z-score calculations and probability distributions
├── test_analysis/            # Strategy research and parameter optimization
│   ├── analysis.ipynb        # Exploratory backtest analysis notebook
│   └── tune.py               # Hyperparameter tuning script
├── trading_logs/             # Live and backtest execution logs
├── .gitignore                # Git exclusion configuration
├── back_test.py              # Strategy backtesting runner
├── paper.ipynb               # MCMC parameter estimation notebook
├── paper_report.pdf          # Formatted quantitative research report
├── README.md                 # Project overview and instructions
├── requirements.txt          # Python dependencies
└── run_live_ibkr.py          # Interactive Brokers live trading execution driver
