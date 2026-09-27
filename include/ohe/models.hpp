// Option pricing beyond Black-Scholes: Merton jump-diffusion, Heston
// stochastic volatility, and Bates (both) by Monte Carlo.
#pragma once

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>

#include "ohe/black_scholes.hpp"

namespace ohe {

// dv = kappa (theta - v) dt + xi sqrt(v) dW2,  corr(dW1, dW2) = rho
struct HestonParams {
  double v0 = 0.04;
  double kappa = 1.5;
  double theta = 0.04;
  double xi = 0.5;
  double rho = -0.7;
};

// Poisson(lambda) jumps per year; log jump size ~ N(mu, delta^2).
struct JumpParams {
  double lambda = 0.5;
  double mu = -0.1;
  double delta = 0.15;
};

struct Model {
  double sigma = 0.2;                  // diffusion vol when there is no Heston block
  std::optional<HestonParams> heston;  // stochastic variance
  std::optional<JumpParams> jumps;     // Merton jumps
};

struct McConfig {
  std::size_t paths = 200'000;
  std::size_t steps_per_year = 252;  // time steps for path-dependent (Heston) dynamics
  std::uint64_t seed = 42;
  unsigned threads = 0;  // 0 = hardware concurrency; results do not depend on it
  bool antithetic = true;
};

struct PriceResult {
  double price = 0;
  double std_error = 0;  // 0 for closed-form methods
  std::string method;
};

// Merton (1976) closed form: Poisson-weighted sum of Black-Scholes prices.
double merton_price(OptionType type, double S, double K, double r, double q, double sigma,
                    double tau, const JumpParams& jumps);

// Heston (1993) semi-closed form via the characteristic function, using the
// "little Heston trap" formulation (Albrecher et al. 2007) for a continuous
// complex logarithm.
double heston_price(OptionType type, double S, double K, double r, double q, double tau,
                    const HestonParams& p);

// Monte Carlo under any combination of the models above. Heston variance uses
// full-truncation Euler; with constant volatility the terminal price is
// sampled exactly in a single step.
PriceResult mc_price(OptionType type, double S, double K, double r, double q, double tau,
                     const Model& model, const McConfig& mc);

enum class PricingPolicy { Auto, Analytic, MonteCarlo };

// Auto picks the closed form when one exists (BS, Merton, Heston) and Monte
// Carlo for Bates. Analytic throws std::invalid_argument when there is none.
PriceResult price_option(OptionType type, double S, double K, double r, double q, double tau,
                         const Model& model, PricingPolicy policy = PricingPolicy::Auto,
                         const McConfig& mc = {});

}  // namespace ohe
