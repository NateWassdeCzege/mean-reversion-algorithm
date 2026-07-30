# Quantitative Mean Reversion Trading Algorithm

A productiongrade, mathematically rigorous statistical arbitrage trading system grounded in stochastic calculus. This project bridges theoretical stochastic differential equations with applied quantitative finance, featuring strict risk management protocols and dynamic regime filtering.

## Overview

Traditional financial analysis often overlooks shortterm market dynamics, which can be effectively modeled using stochastic processes[cite: 1]. This project implements a MeanReversion Strategy using the OrnsteinUhlenbeck (OU) process, backed by Markov Chain Monte Carlo (MCMC) parameter estimation and a Hidden Markov Model (HMM) market regime filter[cite: 1]. 

Unlike static trading bots, this system continuously evaluates parameter uncertainty, filters out hazardous trending markets, and sizes positions dynamically based on expected value calculations[cite: 1].

## Core Architecture & Methodology

The system operates on a dualtrack integration architecture that processes hourly OHLCV data to make statistically grounded execution decisions[cite: 1]:

### 1. Stochastic Modeling: The Ornstein-Uhlenbeck (OU) Process
To capture mean-reverting equity behavior, asset price dynamics are modeled via the stochastic differential equation (SDE)[cite: 1]:
$$dx_{t}=\theta(\mu-x_{t})dt+\sigma dW_{t}$$
* Deterministic drift ($\theta(\mu-x_{t})dt$) pulls the price back toward the long-term equilibrium mean ($\mu$) at a reversion speed ($\theta$)[cite: 1].
* Stochastic diffusion ($\sigma dW_{t}$) captures market volatility ($\sigma$) and unpredictable random noise via a Wiener process ($dW_{t}$)[cite: 1].

### 2. Parameter Estimation via MCMC
To overcome the limitations of static Maximum Likelihood Estimation (MLE), the model applies a Markov Chain Monte Carlo (MCMC) MetropolisHastings algorithm over a rolling 130tick window (approx. 21 trading days)[cite: 1]. This generates datadriven posterior probability distributions for the parameter set $\Theta = [\theta, \mu, \sigma]$, effectively accounting for statistical uncertainty[cite: 1].

### 3. RealTime Regime Classification (HMM)
To protect capital from catastrophic drawdowns during aggressive structural trends, the framework incorporates a Hidden Markov Model (HMM) trained via the BaumWelch (ForwardBackward) algorithm[cite: 1]. 
* Feature Vector ($O_t$) combines Log Returns, Smooth Momentum, Average Directional Index (ADX), and Moving Average Spread[cite: 1].
* Viterbi Decoding decodes hidden market states in realtime, completely muting trading signals during unfavorable trending regimes to prevent adverse selection[cite: 1].

## Risk Management & Execution Logic

* Probabilistic Signal Filtering evaluates the Cumulative Distribution Function (CDF) of MCMC outputs, requiring $\ge 75\%$ probability mass beyond target Zscore thresholds ($\ge 2$ or $\le 2$) alongside strict kurtosis and signconsensus checks[cite: 1].
* Path Simulation & Expected Value (EV) simulates 2,000 discrete price paths using EulerMaruyama discretization over a 50period horizon to compute the empirical reversion probability ($P_{revert}$) and ensure positive trade expectancy[cite: 1].
* ConfidenceScaled Allocation caps maximum trade exposure at 2% of total portfolio capital, linearly scaling base position sizes using model confidence[cite: 1].
* Dynamic Exits & StopLosses:
  * Takeprofit signals are triggered dynamically when prices enter the 50% Highest Density Interval (HDI) of the expected mean[cite: 1].
  * Duallayered stoplosses include a 50tick timestop and a structural blowout stop ($Z \ge 3.75$)[cite: 1].

## Project Structure

```text
├── .gitignore
├── HMM_main.py                  # Hidden Markov Model regime classification
├── LICENSE
├── README.md
├── data_feed.py                 # Market data ingestion stream
├── helperfucatoins.py           # Utility and helper functions
├── ko_hmm_final_pipeline2.pkl   # Serialized HMM pipeline model
├── main.py                      # Main execution entry point
├── mean_revert_paper.ipynb - Colab.pdf  # Detailed mathematical documentation
├── postion_record.py            # Position recording and management
├── requirements.txt             # Python dependencies
├── singnal_engine.py            # Signal generation and trade filtering engine
└── z_distro.py                  # Z-score and distribution calculations
└── README.md
Documentation
For an in-depth mathematical breakdown, full system architecture schematics, and rigorous evaluation metrics, please refer to the complete research paper included in the repository: mean_revert_paper.ipynb - Colab.pdf[cite: 1].
