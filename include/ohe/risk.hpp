// Portfolio risk: Monte Carlo VaR / Expected Shortfall with full revaluation,
// exposure measures and hard circuit breakers.
#pragma once

#include <cstddef>
#include <cstdint>
#include <map>
#include <string>
#include <vector>

#include "ohe/black_scholes.hpp"

namespace ohe {

struct Asset {
  std::string name;
  std::string asset_class;  // for concentration limits, e.g. "equity-index"
  double spot = 100;
  double vol = 0.2;  // annualized, drives the risk simulation
  double dividend_yield = 0;
};

struct Position {
  enum class Kind { Underlying, Option };
  std::size_t asset = 0;
  Kind kind = Kind::Underlying;
  double quantity = 0;  // shares, or contracts for options (negative = short)
  OptionType type = OptionType::Call;
  double strike = 0;
  double expiry = 0;        // years
  double implied_vol = 0;   // held constant over the risk horizon
  double multiplier = 100;  // shares per contract
};

struct Portfolio {
  std::vector<Asset> assets;
  std::vector<Position> positions;
  std::vector<std::vector<double>> correlation;  // assets x assets; empty = identity
  double cash = 0;
  double rate = 0.0;
};

double position_value(const Portfolio& pf, const Position& p, double spot, double time_elapsed = 0);
double position_delta_shares(const Portfolio& pf, const Position& p, double spot);
double portfolio_value(const Portfolio& pf);

struct VarConfig {
  double horizon_days = 1;
  double trading_days = 252;
  std::size_t paths = 100'000;
  std::uint64_t seed = 7;
  unsigned threads = 0;
};

struct RiskReport {
  double value = 0;               // mark-to-market incl. cash
  double net_delta_notional = 0;  // sum of delta x spot, signed
  double gross_notional = 0;      // sum of |delta x spot| per position
  double gross_leverage = 0;      // gross_notional / value
  double var95 = 0, var99 = 0;    // losses (positive numbers) over the horizon
  double es95 = 0, es99 = 0;
  std::map<std::string, double> class_share;  // share of gross notional per asset class
};

// Simulates correlated lognormal moves of every asset over the horizon and
// revalues each position (options by Black-Scholes at their implied vol).
RiskReport assess(const Portfolio& pf, const VarConfig& cfg = {});

// Lower Cholesky factor; throws if the matrix is not positive definite.
std::vector<std::vector<double>> cholesky(const std::vector<std::vector<double>>& a);

struct RiskLimits {
  double max_net_delta_fraction = 1.5;  // |net delta notional| / equity
  double max_gross_leverage = 4.0;
  double max_class_share = 0.8;         // any single asset class share of gross
  double max_var99_fraction = 0.10;     // 1-day VaR99 / equity
};

struct Breach {
  std::string limit;
  double value;
  double threshold;
};

std::vector<Breach> check_limits(const RiskReport& report, const RiskLimits& limits);

// Latches once any limit is breached: after halt, no new risk may be taken
// until reset() is called explicitly.
class CircuitBreaker {
 public:
  explicit CircuitBreaker(RiskLimits limits) : limits_(limits) {}
  // Returns true if trading may continue.
  bool evaluate(const RiskReport& report);
  void halt_trading(std::string reason);
  void reset();
  bool halted() const { return halted_; }
  const std::vector<std::string>& reasons() const { return reasons_; }
  const RiskLimits& limits() const { return limits_; }

 private:
  RiskLimits limits_;
  bool halted_ = false;
  std::vector<std::string> reasons_;
};

}  // namespace ohe
