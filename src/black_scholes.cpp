#include "ohe/black_scholes.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numbers>

namespace ohe {

double norm_cdf(double x) { return 0.5 * std::erfc(-x / std::numbers::sqrt2); }

double norm_pdf(double x) { return std::exp(-0.5 * x * x) / std::sqrt(2.0 * std::numbers::pi); }

Greeks black_scholes(OptionType type, double S, double K, double r, double q, double sigma,
                     double tau) {
  Greeks g;
  const bool call = type == OptionType::Call;
  if (tau <= 0.0 || sigma <= 0.0) {
    const double df_q = std::exp(-q * std::max(tau, 0.0));
    const double df_r = std::exp(-r * std::max(tau, 0.0));
    const double fwd_intrinsic = call ? S * df_q - K * df_r : K * df_r - S * df_q;
    g.price = std::max(fwd_intrinsic, 0.0);
    g.delta = fwd_intrinsic > 0.0 ? (call ? df_q : -df_q) : 0.0;
    return g;
  }
  const double sqrt_t = std::sqrt(tau);
  const double d1 = (std::log(S / K) + (r - q + 0.5 * sigma * sigma) * tau) / (sigma * sqrt_t);
  const double d2 = d1 - sigma * sqrt_t;
  const double df_q = std::exp(-q * tau);
  const double df_r = std::exp(-r * tau);
  const double pdf = norm_pdf(d1);

  if (call) {
    g.price = S * df_q * norm_cdf(d1) - K * df_r * norm_cdf(d2);
    g.delta = df_q * norm_cdf(d1);
    g.theta = -S * df_q * pdf * sigma / (2 * sqrt_t) - r * K * df_r * norm_cdf(d2) +
              q * S * df_q * norm_cdf(d1);
    g.rho = K * tau * df_r * norm_cdf(d2);
  } else {
    g.price = K * df_r * norm_cdf(-d2) - S * df_q * norm_cdf(-d1);
    g.delta = -df_q * norm_cdf(-d1);
    g.theta = -S * df_q * pdf * sigma / (2 * sqrt_t) + r * K * df_r * norm_cdf(-d2) -
              q * S * df_q * norm_cdf(-d1);
    g.rho = -K * tau * df_r * norm_cdf(-d2);
  }
  g.gamma = df_q * pdf / (S * sigma * sqrt_t);
  g.vega = S * df_q * pdf * sqrt_t;
  return g;
}

double implied_vol(OptionType type, double price, double S, double K, double r, double q,
                   double tau) {
  const double nan = std::numeric_limits<double>::quiet_NaN();
  if (tau <= 0.0 || price <= 0.0) return nan;
  const double lower = bs_price(type, S, K, r, q, 1e-9, tau);
  const double upper = type == OptionType::Call ? S * std::exp(-q * tau) : K * std::exp(-r * tau);
  if (price < lower - 1e-12 || price >= upper) return nan;

  // Brenner-Subrahmanyam start, then Newton; keep a bracket for bisection.
  double lo = 1e-6, hi = 5.0;
  double sigma = std::clamp(std::sqrt(2.0 * std::numbers::pi / tau) * price / S, 0.05, 2.0);
  for (int i = 0; i < 100; ++i) {
    const Greeks g = black_scholes(type, S, K, r, q, sigma, tau);
    const double diff = g.price - price;
    if (std::abs(diff) < 1e-10 * std::max(1.0, price)) return sigma;
    if (diff > 0) hi = sigma;
    else lo = sigma;
    double next = g.vega > 1e-12 ? sigma - diff / g.vega : 0.5 * (lo + hi);
    if (!(next > lo && next < hi)) next = 0.5 * (lo + hi);
    if (hi - lo < 1e-12) return next;
    sigma = next;
  }
  return sigma;
}

}  // namespace ohe
