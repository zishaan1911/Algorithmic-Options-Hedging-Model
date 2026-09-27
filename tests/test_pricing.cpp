#include <cmath>
#include <stdexcept>

#include "check.hpp"
#include "ohe/black_scholes.hpp"
#include "ohe/models.hpp"

using namespace ohe;

// Hull, Options Futures and Other Derivatives, Example 15.6:
// S=42, K=40, r=10%, sigma=20%, T=0.5 -> c = 4.76, p = 0.81.
TEST(black_scholes_textbook_values) {
  CHECK_NEAR(bs_price(OptionType::Call, 42, 40, 0.10, 0.0, 0.20, 0.5), 4.7594, 1e-4);
  CHECK_NEAR(bs_price(OptionType::Put, 42, 40, 0.10, 0.0, 0.20, 0.5), 0.8086, 1e-4);
}

TEST(put_call_parity_with_dividends) {
  for (double K : {70.0, 100.0, 140.0}) {
    for (double tau : {0.05, 0.5, 2.0}) {
      const double c = bs_price(OptionType::Call, 100, K, 0.03, 0.015, 0.3, tau);
      const double p = bs_price(OptionType::Put, 100, K, 0.03, 0.015, 0.3, tau);
      CHECK_NEAR(c - p, 100 * std::exp(-0.015 * tau) - K * std::exp(-0.03 * tau), 1e-10);
    }
  }
}

TEST(greeks_match_finite_differences) {
  const double S = 105, K = 100, r = 0.02, q = 0.01, v = 0.25, T = 0.75;
  for (OptionType ty : {OptionType::Call, OptionType::Put}) {
    const Greeks g = black_scholes(ty, S, K, r, q, v, T);
    auto P = [&](double s, double rr, double vv, double tt) { return bs_price(ty, s, K, rr, q, vv, tt); };
    const double h = 1e-3, k = 1e-4;  // spot step; smaller step for vol, rate and time
    CHECK_NEAR(g.delta, (P(S + h, r, v, T) - P(S - h, r, v, T)) / (2 * h), 1e-6);
    CHECK_NEAR(g.gamma, (P(S + h, r, v, T) - 2 * P(S, r, v, T) + P(S - h, r, v, T)) / (h * h), 1e-4);
    CHECK_NEAR(g.vega, (P(S, r, v + k, T) - P(S, r, v - k, T)) / (2 * k), 1e-6);
    CHECK_NEAR(g.rho, (P(S, r + k, v, T) - P(S, r - k, v, T)) / (2 * k), 1e-6);
    CHECK_NEAR(g.theta, -(P(S, r, v, T + k) - P(S, r, v, T - k)) / (2 * k), 1e-6);
  }
}

TEST(implied_vol_round_trip) {
  for (OptionType ty : {OptionType::Call, OptionType::Put}) {
    for (double K : {60.0, 90.0, 100.0, 115.0, 160.0}) {
      for (double v : {0.05, 0.2, 0.6, 1.5}) {
        const Greeks g = black_scholes(ty, 100, K, 0.02, 0.0, v, 0.4);
        // Skip quotes whose price barely depends on vol (deep in or out of the
        // money at low vol): no finite-precision price can pin those down.
        if (g.vega < 1e-3) continue;
        const double price = g.price;
        CHECK_NEAR(implied_vol(ty, price, 100, K, 0.02, 0.0, 0.4), v, 1e-6);
      }
    }
  }
  CHECK(std::isnan(implied_vol(OptionType::Call, 150.0, 100, 100, 0.02, 0.0, 1.0)));  // above spot
  CHECK(std::isnan(implied_vol(OptionType::Call, 0.5, 100, 90, 0.0, 0.0, 1.0)));     // below intrinsic
}

TEST(merton_reduces_to_black_scholes_without_jumps) {
  const JumpParams none{0.0, -0.1, 0.2};
  CHECK_NEAR(merton_price(OptionType::Call, 100, 95, 0.03, 0.0, 0.2, 1.0, none),
             bs_price(OptionType::Call, 100, 95, 0.03, 0.0, 0.2, 1.0), 1e-12);
}

TEST(merton_series_agrees_with_monte_carlo) {
  Model m;
  m.sigma = 0.2;
  m.jumps = JumpParams{1.0, -0.1, 0.15};
  McConfig mc;
  mc.paths = 400'000;
  for (OptionType ty : {OptionType::Call, OptionType::Put}) {
    const double closed = merton_price(ty, 100, 100, 0.03, 0.01, 0.2, 1.0, *m.jumps);
    const auto sim = mc_price(ty, 100, 100, 0.03, 0.01, 1.0, m, mc);
    CHECK_NEAR(sim.price, closed, 4 * sim.std_error);
  }
}

TEST(heston_without_vol_of_vol_is_black_scholes) {
  // Small xi with v0 = theta and rho = 0: variance stays at theta, so the price
  // is Black-Scholes at sqrt(theta) up to an O(xi^2) correction (~1e-7 here).
  const HestonParams p{0.09, 2.0, 0.09, 1e-3, 0.0};
  for (double K : {80.0, 100.0, 125.0}) {
    CHECK_NEAR(heston_price(OptionType::Call, 100, K, 0.02, 0.0, 1.0, p),
               bs_price(OptionType::Call, 100, K, 0.02, 0.0, 0.3, 1.0), 1e-5);
  }
}

TEST(heston_put_call_parity) {
  const HestonParams p{0.04, 1.5, 0.05, 0.6, -0.7};
  for (double K : {80.0, 100.0, 120.0}) {
    const double c = heston_price(OptionType::Call, 100, K, 0.03, 0.01, 0.8, p);
    const double put = heston_price(OptionType::Put, 100, K, 0.03, 0.01, 0.8, p);
    CHECK_NEAR(c - put, 100 * std::exp(-0.01 * 0.8) - K * std::exp(-0.03 * 0.8), 1e-8);
  }
}

TEST(heston_closed_form_agrees_with_monte_carlo) {
  const HestonParams p{0.04, 2.0, 0.04, 0.4, -0.7};
  Model m;
  m.heston = p;
  McConfig mc;
  mc.paths = 200'000;
  mc.steps_per_year = 400;
  for (double K : {90.0, 100.0, 110.0}) {
    const double closed = heston_price(OptionType::Call, 100, K, 0.02, 0.0, 1.0, p);
    const auto sim = mc_price(OptionType::Call, 100, K, 0.02, 0.0, 1.0, m, mc);
    // Full-truncation Euler has a small discretization bias on top of noise.
    CHECK_NEAR(sim.price, closed, 4 * sim.std_error + 0.02);
  }
}

TEST(heston_skew_from_negative_correlation) {
  // rho < 0 makes downside strikes richer: implied vol falls with strike.
  const HestonParams p{0.04, 1.5, 0.04, 0.6, -0.8};
  auto iv = [&](double K) {
    return implied_vol(OptionType::Call, heston_price(OptionType::Call, 100, K, 0.0, 0.0, 0.5, p), 100, K,
                       0.0, 0.0, 0.5);
  };
  CHECK(iv(85) > iv(100));
  CHECK(iv(100) > iv(115));
}

TEST(router_picks_the_right_method) {
  Model m;
  CHECK(price_option(OptionType::Call, 100, 100, 0.01, 0, 1, m).method == "black-scholes");
  m.jumps = JumpParams{};
  CHECK(price_option(OptionType::Call, 100, 100, 0.01, 0, 1, m).method == "merton");
  m.jumps.reset();
  m.heston = HestonParams{};
  CHECK(price_option(OptionType::Call, 100, 100, 0.01, 0, 1, m).method == "heston");
  m.jumps = JumpParams{};
  McConfig mc;
  mc.paths = 8192;
  CHECK(price_option(OptionType::Call, 100, 100, 0.01, 0, 1, m, PricingPolicy::Auto, mc).method ==
        "monte-carlo (bates)");
  CHECK_THROWS(price_option(OptionType::Call, 100, 100, 0.01, 0, 1, m, PricingPolicy::Analytic, mc));
}

TEST(expired_option_is_intrinsic) {
  CHECK_NEAR(bs_price(OptionType::Call, 110, 100, 0.05, 0, 0.3, 0.0), 10.0, 1e-12);
  CHECK_NEAR(bs_price(OptionType::Put, 110, 100, 0.05, 0, 0.3, 0.0), 0.0, 1e-12);
}
