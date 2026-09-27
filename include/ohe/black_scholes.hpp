// Black-Scholes-Merton pricing with continuous dividend yield.
#pragma once

namespace ohe {

enum class OptionType { Call, Put };

struct Greeks {
  double price = 0;
  double delta = 0;  // dV/dS
  double gamma = 0;  // d2V/dS2
  double vega = 0;   // dV/dsigma, per 1.00 of vol
  double theta = 0;  // dV/dt, per year (calendar time passing)
  double rho = 0;    // dV/dr
};

double norm_cdf(double x);
double norm_pdf(double x);

// Spot S, strike K, rate r, dividend yield q, volatility sigma, years to expiry tau.
// tau <= 0 or sigma <= 0 returns the discounted intrinsic value.
Greeks black_scholes(OptionType type, double S, double K, double r, double q, double sigma,
                     double tau);

inline double bs_price(OptionType type, double S, double K, double r, double q, double sigma,
                       double tau) {
  return black_scholes(type, S, K, r, q, sigma, tau).price;
}

// Volatility that reproduces `price`. Newton-Raphson on vega, falling back to
// bisection; NaN if the price violates the no-arbitrage bounds.
double implied_vol(OptionType type, double price, double S, double K, double r, double q,
                   double tau);

}  // namespace ohe
