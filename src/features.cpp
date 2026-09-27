#include "ohe/features.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <tuple>

namespace ohe {

namespace {

std::span<const double> tail(std::span<const double> x, int window) {
  const std::size_t n = std::min<std::size_t>(x.size(), static_cast<std::size_t>(std::max(window, 1)));
  return x.subspan(x.size() - n);
}

double mean(std::span<const double> x) {
  double s = 0;
  for (double v : x) s += v;
  return x.empty() ? 0.0 : s / static_cast<double>(x.size());
}

}  // namespace

std::vector<double> log_returns(std::span<const double> prices) {
  std::vector<double> r;
  if (prices.size() < 2) return r;
  r.reserve(prices.size() - 1);
  for (std::size_t i = 1; i < prices.size(); ++i) {
    if (prices[i] <= 0 || prices[i - 1] <= 0) throw std::invalid_argument("prices must be positive");
    r.push_back(std::log(prices[i] / prices[i - 1]));
  }
  return r;
}

double sma(std::span<const double> x, int window) { return mean(tail(x, window)); }

double ewma(std::span<const double> x, double halflife) {
  if (x.empty()) return 0.0;
  const double alpha = 1.0 - std::exp2(-1.0 / halflife);
  double m = x.front();
  for (std::size_t i = 1; i < x.size(); ++i) m += alpha * (x[i] - m);
  return m;
}

double realized_vol(std::span<const double> returns, int window, double periods_per_year) {
  const auto w = tail(returns, window);
  if (w.size() < 2) return 0.0;
  const double m = mean(w);
  double ss = 0;
  for (double v : w) ss += (v - m) * (v - m);
  return std::sqrt(ss / static_cast<double>(w.size() - 1) * periods_per_year);
}

double ewma_vol(std::span<const double> returns, double halflife, double periods_per_year) {
  if (returns.empty()) return 0.0;
  const double alpha = 1.0 - std::exp2(-1.0 / halflife);
  double var = returns.front() * returns.front();
  for (std::size_t i = 1; i < returns.size(); ++i) var += alpha * (returns[i] * returns[i] - var);
  return std::sqrt(var * periods_per_year);
}

std::pair<double, double> skew_kurtosis(std::span<const double> returns, int window) {
  const auto w = tail(returns, window);
  if (w.size() < 4) return {0.0, 0.0};
  const double m = mean(w);
  double m2 = 0, m3 = 0, m4 = 0;
  for (double v : w) {
    const double d = v - m;
    m2 += d * d;
    m3 += d * d * d;
    m4 += d * d * d * d;
  }
  const double n = static_cast<double>(w.size());
  m2 /= n;
  m3 /= n;
  m4 /= n;
  if (m2 <= 0) return {0.0, 0.0};
  return {m3 / std::pow(m2, 1.5), m4 / (m2 * m2) - 3.0};
}

double noise_variance(std::span<const double> returns, int window) {
  const auto w = tail(returns, window);
  if (w.size() < 3) return 0.0;
  const double m = mean(w);
  double cov = 0;
  for (std::size_t i = 1; i < w.size(); ++i) cov += (w[i] - m) * (w[i - 1] - m);
  cov /= static_cast<double>(w.size() - 1);
  return std::max(-cov, 0.0);
}

Features compute_features(std::span<const double> prices, const FeatureConfig& cfg) {
  if (prices.empty()) throw std::invalid_argument("empty price history");
  Features f;
  const auto r = log_returns(prices);
  f.observations = static_cast<int>(prices.size());
  f.last = prices.back();
  f.sma_fast = sma(prices, cfg.sma_fast);
  f.sma_slow = sma(prices, cfg.sma_slow);
  f.ewma = ewma(prices, cfg.ewma_halflife);
  f.realized_vol = realized_vol(r, cfg.vol_window, cfg.periods_per_year);
  f.ewma_vol = ewma_vol(r, cfg.ewma_halflife, cfg.periods_per_year);
  std::tie(f.skew, f.excess_kurtosis) = skew_kurtosis(r, cfg.vol_window);
  f.noise_variance = noise_variance(r, cfg.vol_window);
  const std::size_t back = std::min<std::size_t>(prices.size() - 1, static_cast<std::size_t>(cfg.vol_window));
  f.momentum = std::log(prices.back() / prices[prices.size() - 1 - back]);
  return f;
}

}  // namespace ohe
