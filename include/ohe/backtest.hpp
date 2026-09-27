// Event-driven, deterministic backtester for a delta-hedged volatility strategy.
//
// Each bar: mark the option book on the day's vol surface, attribute P&L to
// delta / gamma / theta / vega, settle expiries, run the risk engine (a breach
// halts trading and flattens the book), dispatch the strategy when flat, and
// re-hedge delta through an execution model with slippage, fill noise and fees.
#pragma once

#include <cstdint>
#include <span>
#include <string>
#include <vector>

#include "ohe/data.hpp"
#include "ohe/features.hpp"
#include "ohe/ledger.hpp"
#include "ohe/risk.hpp"
#include "ohe/vol_surface.hpp"

namespace ohe {

struct ExecutionModel {
  double slippage_bps = 1.0;          // underlying: fills this far through mid
  double fill_noise_bps = 0.5;        // std dev of random fill noise (microstructure)
  double fee_per_share = 0.005;
  double option_half_spread = 0.005;  // in vol points: half-spread = this x vega
  double fee_per_contract = 0.65;
};

struct BacktestConfig {
  double initial_capital = 1'000'000;
  double rate = 0.02;
  double dividend_yield = 0.0;
  double trading_days = 252;

  // Strategy: straddles of `tenor_days`, short when implied vol exceeds the
  // realized-vol forecast by more than `entry_edge`, long when below.
  int tenor_days = 21;
  double entry_edge = 0.02;
  bool allow_long_vol = true;
  double vega_budget = 0.002;   // equity fraction at risk per 1 vol point
  int max_contracts = 5'000;
  double multiplier = 100;
  double strike_step = 1.0;
  int warmup = 63;              // bars of history before the first trade

  // Implied vol when the data has no IV column: long-window realized vol
  // scaled by (1 + premium). Reported as modeled in the summary.
  double modeled_iv_premium = 0.10;
  int modeled_iv_window = 63;

  double hedge_band = 0.05;     // re-hedge when |delta| exceeds this share of book size
  ExecutionModel execution;
  SurfaceParams surface;        // smile/skew shape; ATM level comes from the data
  FeatureConfig features;
  RiskLimits limits;
  VarConfig var{1, 252, 20'000, 7, 0};
  bool risk_checks = true;       // daily risk run on the live book (circuit breaker)
  bool pre_trade_checks = true;  // reject trades that would breach a limit once hedged

  std::uint64_t seed = 42;
  std::string run_id = "backtest";
};

struct Decision {
  int side = 0;  // -1 sell vol, +1 buy vol, 0 stay flat
  double implied = 0;
  double forecast = 0;
  double edge = 0;
  int contracts = 0;
  double strike = 0;
  Features features;
  std::string reason;
};

// Strategy dispatch: features -> signal -> size. `history` is closes up to today.
Decision dispatch_strategy(std::span<const double> history, double implied_atm, double equity,
                           const BacktestConfig& cfg);

struct BacktestSummary {
  int days = 0;
  int option_trades = 0;
  int hedge_trades = 0;
  double start_equity = 0, end_equity = 0;
  double total_return = 0, annual_return = 0, annual_vol = 0, sharpe = 0, max_drawdown = 0;
  double option_pnl = 0, hedge_pnl = 0, interest = 0, costs = 0;
  double delta_pnl = 0, gamma_pnl = 0, theta_pnl = 0, vega_pnl = 0, residual_pnl = 0;
  bool implied_modeled = false;
  bool halted = false;
  std::vector<std::string> halt_reasons;
  int risk_rejections = 0;
};

struct BacktestResult {
  BacktestSummary summary;
  std::vector<DayRecord> days;
  std::vector<FillRecord> fills;
  std::vector<EventRecord> events;
};

// Deterministic for a given (bars, config): the seed drives fill noise and the
// risk engine's simulations.
BacktestResult run_backtest(const std::vector<Bar>& bars, const BacktestConfig& cfg,
                            Ledger* ledger = nullptr);

}  // namespace ohe
