// Time-series features computed from a price history (oldest first).
#pragma once

#include <span>
#include <utility>
#include <vector>

namespace ohe {

struct FeatureConfig {
  int sma_fast = 10;
  int sma_slow = 50;
  double ewma_halflife = 20;  // observations
  int vol_window = 21;        // realized-vol, skew and kurtosis window (returns)
  double periods_per_year = 252;
};

struct Features {
  double last = 0;
  double sma_fast = 0;
  double sma_slow = 0;
  double ewma = 0;                  // EWMA of price
  double realized_vol = 0;          // annualized, close-to-close log returns
  double ewma_vol = 0;              // annualized RiskMetrics-style EWMA volatility
  double skew = 0;                  // sample skewness of returns in the window
  double excess_kurtosis = 0;       // sample excess kurtosis
  double noise_variance = 0;        // microstructure noise variance (Roll estimator)
  double momentum = 0;              // log(last / price vol_window returns ago)
  int observations = 0;
};

std::vector<double> log_returns(std::span<const double> prices);

double sma(std::span<const double> x, int window);
// Exponentially weighted mean with the given half-life in observations.
double ewma(std::span<const double> x, double halflife);
// Annualized standard deviation of the last `window` returns.
double realized_vol(std::span<const double> returns, int window, double periods_per_year);
// Annualized EWMA volatility (variance smoothed with the same half-life rule).
double ewma_vol(std::span<const double> returns, double halflife, double periods_per_year);
// Skewness and excess kurtosis of the last `window` returns.
std::pair<double, double> skew_kurtosis(std::span<const double> returns, int window);

// Observation noise variance implied by negative first-order autocovariance of
// returns: if observed log price = efficient price + iid noise, then
// cov(r_t, r_{t-1}) = -omega^2 (Roll 1984). Zero when the autocovariance is >= 0.
double noise_variance(std::span<const double> returns, int window);

Features compute_features(std::span<const double> prices, const FeatureConfig& cfg = {});

}  // namespace ohe
