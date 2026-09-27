#include <cmath>
#include <vector>

#include "check.hpp"
#include "ohe/features.hpp"
#include "ohe/rng.hpp"
#include "ohe/vol_surface.hpp"

using namespace ohe;

namespace {

std::vector<double> gbm(double sigma, int n, std::uint64_t seed, double noise = 0.0) {
  Pcg64 rng = Pcg64::from_seed(seed);
  std::vector<double> prices;
  double log_s = std::log(100.0);
  const double dt = 1.0 / 252;
  for (int i = 0; i < n; ++i) {
    if (i > 0) log_s += -0.5 * sigma * sigma * dt + sigma * std::sqrt(dt) * rng.normal();
    prices.push_back(std::exp(log_s + noise * rng.normal()));
  }
  return prices;
}

}  // namespace

TEST(surface_reproduces_its_nodes) {
  const SurfaceParams p{0.25, 0.20, 0.5, -0.1, 0.02, 0.01};
  const VolSurface s = VolSurface::parametric(100, p, {0.1, 0.5, 1.0}, 11);
  for (const auto& slice : s.slices()) {
    for (std::size_t i = 0; i < slice.log_strikes.size(); ++i) {
      const double K = 100 * std::exp(slice.log_strikes[i]);
      CHECK_NEAR(s.iv(K, slice.expiry), slice.vols[i], 1e-12);
    }
  }
}

TEST(surface_skew_smile_and_flat_wings) {
  const SurfaceParams p{0.2, 0.2, 0.5, -0.1, 0.03, 0.01};
  const VolSurface s = VolSurface::parametric(100, p, {0.25, 1.0});
  CHECK(s.iv(90, 0.25) > s.iv(100, 0.25));   // negative skew
  CHECK_NEAR(s.iv(100, 0.25), 0.2, 1e-12);   // ATM level
  CHECK_NEAR(s.iv(1, 0.25), s.iv(2, 0.25), 1e-12);         // flat below the lowest node
  CHECK_NEAR(s.iv(10000, 1.0), s.iv(20000, 1.0), 1e-12);   // and above the highest
}

TEST(surface_interpolates_total_variance_between_expiries) {
  const VolSurface s(100, {{0.25, {0.0}, {0.30}}, {1.0, {0.0}, {0.20}}});
  const double t = 0.5;
  const double w = 0.30 * 0.30 * 0.25 + (0.20 * 0.20 * 1.0 - 0.30 * 0.30 * 0.25) * (t - 0.25) / 0.75;
  CHECK_NEAR(s.iv(100, t), std::sqrt(w / t), 1e-12);
}

TEST(calendar_arbitrage_detection) {
  const SurfaceParams sane{0.3, 0.2, 0.5, -0.1, 0.02, 0.01};
  CHECK(VolSurface::parametric(100, sane, {0.1, 0.25, 0.5, 1.0, 2.0}).calendar_violations() == 0);
  // Total variance falls from the first to the second expiry: arbitrage.
  const VolSurface bad(100, {{0.5, {-0.2, 0.2}, {0.60, 0.60}}, {0.6, {-0.2, 0.2}, {0.20, 0.20}}});
  CHECK(bad.calendar_violations() > 0);
}

TEST(surface_rescales_to_a_quoted_atm_vol) {
  const SurfaceParams p{0.25, 0.18, 0.4, -0.12, 0.02, 0.01};
  const VolSurface s = VolSurface::parametric(100, p, {0.05, 0.25, 1.0});
  const VolSurface r = s.rescaled_to_atm(0.1, 0.31);
  CHECK_NEAR(r.iv(100, 0.1), 0.31, 1e-9);
  CHECK(r.iv(90, 0.1) > r.iv(110, 0.1));  // shape preserved
}

TEST(moving_averages) {
  const std::vector<double> x{1, 2, 3, 4, 5, 6};
  CHECK_NEAR(sma(x, 3), 5.0, 1e-12);
  CHECK_NEAR(sma(x, 100), 3.5, 1e-12);
  const std::vector<double> flat(50, 7.0);
  CHECK_NEAR(ewma(flat, 10), 7.0, 1e-12);
  // Half-life h: a step is half absorbed after h observations.
  std::vector<double> step(21, 0.0);
  for (std::size_t i = 1; i < step.size(); ++i) step[i] = 1.0;
  CHECK_NEAR(ewma(std::span<const double>(step).first(11), 10), 0.5, 1e-12);
}

TEST(realized_vol_recovers_simulated_sigma) {
  const auto prices = gbm(0.25, 20'000, 3);
  const auto r = log_returns(prices);
  CHECK_NEAR(realized_vol(r, 20'000, 252), 0.25, 0.005);
  CHECK_NEAR(ewma_vol(r, 250, 252), 0.25, 0.03);
  const auto [skew, kurt] = skew_kurtosis(r, 20'000);
  CHECK_NEAR(skew, 0.0, 0.05);
  CHECK_NEAR(kurt, 0.0, 0.1);
}

TEST(noise_variance_estimator) {
  const double omega = 0.004;  // 40 bp observation noise
  const auto noisy = log_returns(gbm(0.2, 50'000, 11, omega));
  CHECK_NEAR(noise_variance(noisy, 50'000), omega * omega, 0.2 * omega * omega);
  const auto clean = log_returns(gbm(0.2, 50'000, 11));
  CHECK(noise_variance(clean, 50'000) < 0.1 * omega * omega);
}

TEST(feature_bundle) {
  const auto prices = gbm(0.2, 300, 21);
  const Features f = compute_features(prices);
  CHECK(f.observations == 300);
  CHECK_NEAR(f.last, prices.back(), 1e-12);
  CHECK(f.realized_vol > 0.05 && f.realized_vol < 0.5);
  CHECK_NEAR(f.momentum, std::log(prices.back() / prices[prices.size() - 22]), 1e-12);
  CHECK_THROWS(compute_features(std::vector<double>{}));
}
