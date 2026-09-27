#include "ohe/models.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <complex>
#include <numbers>
#include <stdexcept>
#include <vector>

#include "ohe/parallel.hpp"
#include "ohe/rng.hpp"

namespace ohe {

// ------------------------------------------------------------------ Merton

double merton_price(OptionType type, double S, double K, double r, double q, double sigma,
                    double tau, const JumpParams& j) {
  if (tau <= 0.0 || j.lambda <= 0.0) return bs_price(type, S, K, r, q, sigma, tau);
  const double k = std::exp(j.mu + 0.5 * j.delta * j.delta) - 1.0;  // E[jump] - 1
  const double lam = j.lambda * (1.0 + k);
  const double lt = lam * tau;
  double total = 0.0;
  for (int n = 0; n < 200; ++n) {
    const double log_w = -lt + n * std::log(lt) - std::lgamma(n + 1.0);
    const double w = std::exp(log_w);
    const double sigma_n = std::sqrt(sigma * sigma + n * j.delta * j.delta / tau);
    const double r_n = r - j.lambda * k + n * std::log1p(k) / tau;
    total += w * bs_price(type, S, K, r_n, q, sigma_n, tau);
    if (n > lt && w < 1e-16) break;
  }
  return total;
}

// ------------------------------------------------------------------ Heston

namespace {

using cplx = std::complex<double>;

// Probability P_j of the Heston formula (j = 1 or 2).
double heston_p(int j, double x, double lnK, double r, double q, double tau,
                const HestonParams& p) {
  const double u = j == 1 ? 0.5 : -0.5;
  const double b = j == 1 ? p.kappa - p.rho * p.xi : p.kappa;
  const double s2 = p.xi * p.xi;
  const cplx i(0.0, 1.0);

  auto integrand = [&](double phi) {
    const cplx a = b - p.rho * p.xi * i * phi;
    const cplx d = std::sqrt(a * a - s2 * (2.0 * u * i * phi - phi * phi));
    const cplx g = (a - d) / (a + d);
    const cplx e = std::exp(-d * tau);
    const cplx C = (r - q) * i * phi * tau +
                   p.kappa * p.theta / s2 * ((a - d) * tau - 2.0 * std::log((1.0 - g * e) / (1.0 - g)));
    const cplx D = (a - d) / s2 * ((1.0 - e) / (1.0 - g * e));
    const cplx f = std::exp(C + D * p.v0 + i * phi * x);
    return std::real(std::exp(-i * phi * lnK) * f / (i * phi));
  };

  // Composite 8-point Gauss-Legendre on [0, 250]; the integrand decays fast
  // enough for maturities down to a few days.
  static constexpr std::array<double, 8> nodes{-0.9602898564975363, -0.7966664774136267,
                                               -0.5255324099163290, -0.1834346424956498,
                                               0.1834346424956498,  0.5255324099163290,
                                               0.7966664774136267,  0.9602898564975363};
  static constexpr std::array<double, 8> weights{0.1012285362903763, 0.2223810344533745,
                                                 0.3137066458778873, 0.3626837833783620,
                                                 0.3626837833783620, 0.3137066458778873,
                                                 0.2223810344533745, 0.1012285362903763};
  const double upper = 250.0;
  const int panels = 500;
  const double h = upper / panels;
  double sum = 0.0;
  for (int k = 0; k < panels; ++k) {
    const double mid = (k + 0.5) * h;
    for (std::size_t n = 0; n < nodes.size(); ++n) {
      sum += weights[n] * integrand(mid + 0.5 * h * nodes[n]);
    }
  }
  return 0.5 + sum * 0.5 * h / std::numbers::pi;
}

}  // namespace

double heston_price(OptionType type, double S, double K, double r, double q, double tau,
                    const HestonParams& p) {
  if (tau <= 0.0) return bs_price(type, S, K, r, q, 0.0, 0.0);
  const double x = std::log(S), lnK = std::log(K);
  const double call = S * std::exp(-q * tau) * heston_p(1, x, lnK, r, q, tau, p) -
                      K * std::exp(-r * tau) * heston_p(2, x, lnK, r, q, tau, p);
  if (type == OptionType::Call) return std::max(call, 0.0);
  return std::max(call - S * std::exp(-q * tau) + K * std::exp(-r * tau), 0.0);
}

// ------------------------------------------------------------------ Monte Carlo

PriceResult mc_price(OptionType type, double S, double K, double r, double q, double tau,
                     const Model& model, const McConfig& mc) {
  constexpr std::size_t kBlock = 4096;  // paths per block (even, for antithetic pairs)
  const std::size_t blocks = std::max<std::size_t>(1, (mc.paths + kBlock - 1) / kBlock);
  const bool call = type == OptionType::Call;
  const double df = std::exp(-r * tau);

  const JumpParams jp = model.jumps.value_or(JumpParams{0.0, 0.0, 0.0});
  const double jump_comp = jp.lambda * (std::exp(jp.mu + 0.5 * jp.delta * jp.delta) - 1.0);
  const bool stoch_vol = model.heston.has_value();
  const std::size_t steps =
      stoch_vol ? std::max<std::size_t>(
                      1, static_cast<std::size_t>(std::ceil(tau * static_cast<double>(mc.steps_per_year))))
                : 1;
  const double dt = tau / static_cast<double>(steps);

  struct Partial {
    double sum = 0, sum_sq = 0;
    std::size_t n = 0;
  };
  std::vector<Partial> partials(blocks);

  parallel_blocks(blocks, mc.threads ? mc.threads : default_threads(), [&](std::size_t blk) {
    Pcg64 rng = Pcg64::from_seed(mc.seed, blk);
    std::vector<double> z1(steps), z2(steps), jumps(steps);
    Partial part;
    const std::size_t samples = mc.antithetic ? kBlock / 2 : kBlock;
    for (std::size_t s = 0; s < samples; ++s) {
      for (std::size_t t = 0; t < steps; ++t) {
        z1[t] = rng.normal();
        z2[t] = stoch_vol ? rng.normal() : 0.0;
        double jsum = 0.0;
        const unsigned nj = rng.poisson(jp.lambda * dt);
        for (unsigned k = 0; k < nj; ++k) jsum += jp.mu + jp.delta * rng.normal();
        jumps[t] = jsum;
      }
      double payoff_sum = 0.0;
      const int legs = mc.antithetic ? 2 : 1;
      for (int leg = 0; leg < legs; ++leg) {
        const double sign = leg == 0 ? 1.0 : -1.0;
        double log_s = std::log(S);
        if (stoch_vol) {
          const HestonParams& h = *model.heston;
          double v = h.v0;
          const double rho_c = std::sqrt(1.0 - h.rho * h.rho);
          for (std::size_t t = 0; t < steps; ++t) {
            const double vp = std::max(v, 0.0);
            const double w1 = sign * z1[t];
            const double w2 = h.rho * w1 + rho_c * sign * z2[t];
            log_s += (r - q - jump_comp - 0.5 * vp) * dt + std::sqrt(vp * dt) * w1 + jumps[t];
            v += h.kappa * (h.theta - vp) * dt + h.xi * std::sqrt(vp * dt) * w2;
          }
        } else {
          const double sig = model.sigma;
          log_s += (r - q - jump_comp - 0.5 * sig * sig) * tau + sig * std::sqrt(tau) * sign * z1[0] +
                   jumps[0];
        }
        const double ST = std::exp(log_s);
        payoff_sum += call ? std::max(ST - K, 0.0) : std::max(K - ST, 0.0);
      }
      const double sample = payoff_sum / legs;  // antithetic pair average
      part.sum += sample;
      part.sum_sq += sample * sample;
      ++part.n;
    }
    partials[blk] = part;
  });

  Partial all;
  for (const Partial& p : partials) {  // fixed order: deterministic across thread counts
    all.sum += p.sum;
    all.sum_sq += p.sum_sq;
    all.n += p.n;
  }
  const double mean = all.sum / static_cast<double>(all.n);
  const double var = std::max(all.sum_sq / static_cast<double>(all.n) - mean * mean, 0.0);
  PriceResult res;
  res.price = df * mean;
  res.std_error = df * std::sqrt(var / static_cast<double>(all.n));
  res.method = stoch_vol ? (model.jumps ? "monte-carlo (bates)" : "monte-carlo (heston)")
                         : (model.jumps ? "monte-carlo (merton)" : "monte-carlo (gbm)");
  return res;
}

// ------------------------------------------------------------------ Router

PriceResult price_option(OptionType type, double S, double K, double r, double q, double tau,
                         const Model& model, PricingPolicy policy, const McConfig& mc) {
  if (policy == PricingPolicy::MonteCarlo) return mc_price(type, S, K, r, q, tau, model, mc);
  if (!model.heston && !model.jumps) {
    return {bs_price(type, S, K, r, q, model.sigma, tau), 0.0, "black-scholes"};
  }
  if (!model.heston) {
    return {merton_price(type, S, K, r, q, model.sigma, tau, *model.jumps), 0.0, "merton"};
  }
  if (!model.jumps) return {heston_price(type, S, K, r, q, tau, *model.heston), 0.0, "heston"};
  if (policy == PricingPolicy::Analytic) {
    throw std::invalid_argument("no closed form for Heston with jumps (Bates); use Monte Carlo");
  }
  return mc_price(type, S, K, r, q, tau, model, mc);
}

}  // namespace ohe
