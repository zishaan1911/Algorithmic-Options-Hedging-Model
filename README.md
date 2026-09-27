# Options Hedging Engine

A C++20 engine for pricing options, measuring portfolio risk and backtesting
delta-hedged volatility strategies. It covers the pieces a small derivatives desk
needs: models and Greeks, a volatility surface, Monte Carlo VaR with circuit
breakers, an execution-cost model, and a reproducible backtester that records every
fill to SQLite.

Every model is checked against an independent result in the test suite:
closed form vs Monte Carlo, analytic Greeks vs finite differences, and the random
number generator bit-for-bit against NumPy.

## What's inside

| Module | What it does |
| --- | --- |
| `rng` | PCG64 (XSL-RR 128/64), **bit-identical to `numpy.random.PCG64`**, with O(log n) jump-ahead and numbered streams. Seeds can be integers or text. A portable 128-bit path covers MSVC. |
| `parallel` | Monte Carlo split into fixed blocks, each with its own stream and summed in block order, so **results are identical on 1 thread or 64**. |
| `black_scholes` | Prices and Greeks (delta, gamma, vega, theta, rho) with a dividend yield; implied vol by Newton with a bisection fallback. |
| `models` | **Merton** jump-diffusion (Poisson series), **Heston** semi-closed form (characteristic function, "little Heston trap" formulation), **Bates** by Monte Carlo (full-truncation Euler, antithetic variates), and a router that picks the right method. |
| `vol_surface` | Smile, skew and term structure. Linear in log-moneyness within an expiry, linear in total variance across expiries, flat wings. Calendar-arbitrage check, and rescaling to a quoted ATM vol. |
| `features` | SMA, EWMA, realized and EWMA volatility, skewness, excess kurtosis, momentum, and microstructure noise variance (Roll estimator). |
| `risk` | Correlated Monte Carlo **VaR and Expected Shortfall (95/99)** with full revaluation of options. Net-delta, leverage and concentration measures. A **circuit breaker** latches on any limit breach. |
| `backtest` | Event-driven, deterministic backtester for a delta-hedged straddle strategy, with an execution model (slippage, random fill noise, per-share and per-contract fees, bid/ask in vol points) and daily P&L attribution. |
| `ledger` | Fills, daily P&L and events in **SQLite (WAL mode)**. A writer thread commits in batches while other processes read. |

## Build

Requirements: CMake ≥ 3.20 and a C++20 compiler (GCC 11+, Clang 14+, MSVC 2022). SQLite
comes from the system if installed; otherwise CMake downloads the pinned, hash-checked
amalgamation.

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel
ctest --test-dir build --output-on-failure
```

CI builds and tests on GCC and Clang (Linux), MSVC (Windows) and AppleClang (macOS).

## Command line

```bash
# Black-Scholes with Greeks; Heston closed form; the same by Monte Carlo; Bates
ohe price --type call --spot 42 --strike 40 --tau 0.5 --rate 0.1 --vol 0.2
ohe price --type call --spot 100 --strike 100 --tau 1 --heston 0.04,2,0.04,0.4,-0.7
ohe price --type call --spot 100 --strike 100 --tau 1 --heston 0.04,2,0.04,0.4,-0.7 --mc 400000
ohe price --type put  --spot 100 --strike 90  --tau 0.5 --heston 0.04,2,0.04,0.4,-0.7 --jumps 0.5,-0.1,0.15

ohe surface --spot 5000 --atm 0.18 --term-long 0.22 --skew -0.08 --smile 0.015
ohe risk                                   # VaR/ES and limit checks on a sample 4-asset book

sh scripts/fetch_fred.sh                   # S&P 500 and VIX from FRED into ./data
ohe backtest --prices data/SP500.csv --iv data/VIXCLS.csv --iv-scale 0.01 \
             --ledger data/runs.db --run spx-vix --equity-out data/equity.csv
ohe ledger --db data/runs.db --run spx-vix

ohe simulate --out data/sim.csv --days 1500 --premium 0.12   # Heston market with a known vol premium
ohe backtest --prices data/sim.csv
```

Pricing output from the second and third commands shows the two independent Heston
methods agreeing:

```
method      heston                         method      monte-carlo (heston)
price       8.523021                       price       8.519788
                                           std error   0.011133  (400000 paths)
```

## The strategy

On each flat day after warmup, `dispatch_strategy` compares the ATM implied vol with
a forecast of realized vol (EWMA of squared returns, 20-day half-life):

- If implied exceeds the forecast by more than `entry_edge` (2 vol points), it
  **sells** a 21-day ATM straddle. If implied is below by the same margin, it buys.
- Size is set by a vega budget: 0.2% of equity per vol point.
- Each day the book is re-hedged to delta-neutral once net delta leaves a band.
- Options are marked on that day's surface, and P&L is attributed to delta, gamma,
  theta and vega using the previous day's Greeks.

Before opening, the hedged book goes through the risk engine, and a trade that would
breach a limit is rejected. Every day the live book is checked again. A breach halts
trading, closes the options and flattens the hedge, and the halt stays latched.

### Ten years of S&P 500 with VIX as implied vol

`ohe backtest --prices data/SP500.csv --iv data/VIXCLS.csv --iv-scale 0.01` (FRED data,
2016-09-26 to 2026-09-22, default settings, seed 42):

```
equity         1000000 -> 1548020  (+54.80%)
annualized     return +4.55%  vol 3.90%  sharpe 0.65  max drawdown 9.15%
trades         107 straddles, 936 hedge fills, 0 risk rejections
p&l            options +323183  hedge +107484  interest +243862  costs -126508
attribution    delta -100754  gamma -1057829  theta +2372968  vega -1249672  residual +358470
```

Read these numbers for what they are:

- **About 244k of the gain is interest** on cash at a flat 2%, not strategy P&L.
  Sharpe is measured over that rate.
- **VIX is used as the ATM implied vol of 21-trading-day straddles.** VIX is a 30-day,
  model-free variance measure, and the skew around it comes from the parametric
  surface, not from real option quotes. With real quotes the P&L would differ.
- **The index has no dividends and the rate is flat**, so carry is approximate.
- **The residual is the part the Taylor expansion misses.** That means vol-spot cross
  effects (vanna), vol convexity (volga) and large gaps such as March 2020.
- The expected short-vol profile is visible: theta pays, while gamma and vega losses
  in volatility spikes take much of it back.

Without an IV column the backtester refuses to run unless you pass `--model-iv
PREMIUM`. Pricing options at realized vol × (1 + premium) builds a volatility premium
into the test, so such a run mostly measures that assumption.

## Testing

`ctest` runs 46 tests, including:

- **RNG:** PCG64 output and jump-ahead match NumPy exactly, and the portable 128-bit
  multiply matches native on a million random inputs.
- **Determinism:** Monte Carlo prices and VaR are identical for any thread count, and
  backtests are identical for a given seed.
- **Pricing:**
  - Black-Scholes reproduces Hull's textbook example (4.7594 / 0.8086).
  - Put-call parity holds with dividends.
  - All five Greeks match finite differences, and implied vol round-trips.
  - Merton's series matches Monte Carlo within 4 standard errors.
  - Heston's closed form matches Monte Carlo and reduces to Black-Scholes as
    vol-of-vol → 0.
  - Negative correlation produces a downward skew.
- **Surface:** it reproduces its nodes, interpolates in total variance, flags calendar
  arbitrage and hits a target ATM vol when rescaled.
- **Features:** realized vol recovers a simulated σ, and the noise estimator recovers
  an injected noise variance.
- **Risk:** single-asset VaR matches the lognormal quantile, diversification lowers
  VaR, options are fully revalued, and the breaker latches until reset.
- **Backtest:**
  - P&L adds up exactly: Δequity = options + hedge + interest − costs.
  - Short vol earns theta over gamma when implied exceeds realized.
  - Pre-trade limits reject trades, and a live breach halts and flattens the book.
- **Ledger:** SQLite records are readable from a second connection while the writer is
  open.

## Layout

```
include/ohe/   public headers (one per module)
src/           implementations
apps/ohe.cpp   command-line interface
tests/         test suite (self-contained harness, no dependencies)
scripts/       data download
```

## Limitations

- European options only. There is no early exercise, so no American pricing.
- The strategy trades one underlying, and the risk engine holds implied vol constant
  over the VaR horizon (spot risk only).
- Heston Monte Carlo uses Euler full truncation, which carries a small discretization
  bias compared with the exact QE scheme.

## Licence

Apache 2.0, see `LICENSE`.
