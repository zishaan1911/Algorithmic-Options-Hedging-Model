## Options Hedging Engine

**Status:** Private / Proprietary Codebase  
**Focus:** Options Pricing, Volatility Surface Modeling, Monte Carlo Risk Management, and Low-Latency Execution.

## Overview
This repository serves as the architectural overview for a proprietary algorithmic trading and options hedging engine. The system is designed to ingest multi-ticker market data, compute time-series alpha signals, dynamically price options contracts, and execute hedges while remaining strictly within quantitative risk limits.

## Core Tech Stack
* **Language:** Python 3.8+
* **Data Processing:** NumPy, Pandas
* **Database/State:** SQLite (WAL mode for concurrent access)
* **Architecture:** Event-Driven, Multi-threaded, SOLID Principles
* **Math/Quant:** Stochastic Monte Carlo, PCG64 Deterministic RNG, Options Pricing Mathematics

## System Architecture

### 1. Strategy Dispatch & Signal Generation
* **Feature Engineering Pipeline:** Dynamically calculates SMA, EWMA, Realized Volatility, Skew, Kurtosis, and a proprietary Microstructure Noise Variance Estimator.
* **Dispatcher:** An abstraction layer (`run_strategy_dispatch`) that marries signal generation with portfolio sizing and dynamic options pricing.

### 2. Options Pricing & Volatility Surface
* **Dynamic Volatility Surface:** Algorithmically constructs volatility surfaces using recent price history. 
* **Smile & Skew Modeling:** Applies deterministic formulas to account for log-moneyness (Volatility Smile), directional bias (Skew), and time-to-expiry (Term Structure).

### 3. Quantitative Risk Engine
* **Monte Carlo VaR & Expected Shortfall (ES):** Simulates thousands of stochastic paths using an N×N correlation matrix to calculate VaR (95/99) and ES (95/99).
* **Hard Circuit Breakers:** Automatically halts trading (`halt_trading`) if net exposure, gross leverage, or asset-class concentration thresholds are breached.

### 4. Deterministic Backtester
* **Reproducibility:** Utilizes `numpy.random.PCG64` and deterministic seed injection to guarantee reproducible historical simulations.
* **Order Matching Engine:** Accounts for custom slippage coefficients, per-contract fees, and microstructure noise. 

## Licensing & Code Access
*The source code for this engine is proprietary and closed-source. Access to sanitized code snippets or architectural discussions is available upon request*
