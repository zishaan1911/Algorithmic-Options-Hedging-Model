#include <cmath>

#include "check.hpp"
#include "ohe/risk.hpp"

using namespace ohe;

namespace {

Portfolio single_stock(double shares, double vol = 0.2) {
  Portfolio pf;
  pf.assets = {{"A", "equity", 100, vol}};
  pf.positions = {{0, Position::Kind::Underlying, shares}};
  pf.cash = 0;
  pf.rate = 0;
  return pf;
}

}  // namespace

TEST(cholesky_factorizes) {
  const std::vector<std::vector<double>> a{{4, 2, 0.4}, {2, 5, 1}, {0.4, 1, 3}};
  const auto l = cholesky(a);
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) {
      double s = 0;
      for (int k = 0; k < 3; ++k) s += l[i][k] * l[j][k];
      CHECK_NEAR(s, a[i][j], 1e-12);
    }
  }
  CHECK_THROWS(cholesky({{1, 2}, {2, 1}}));
}

TEST(var_matches_lognormal_quantile) {
  // Long 1000 shares at 100: loss = 100000 (1 - e^X), X ~ N(-s^2 h / 2, s^2 h).
  const double sigma = 0.2, h = 1.0 / 252;
  VarConfig cfg;
  cfg.paths = 400'000;
  const RiskReport r = assess(single_stock(1000, sigma), cfg);
  auto loss_at = [&](double z) { return 100'000 * (1 - std::exp(-0.5 * sigma * sigma * h - z * sigma * std::sqrt(h))); };
  CHECK_NEAR(r.var99, loss_at(2.326347874), 0.015 * loss_at(2.326347874));
  CHECK_NEAR(r.var95, loss_at(1.644853627), 0.015 * loss_at(1.644853627));
  CHECK(r.es95 > r.var95);
  CHECK(r.es99 > r.var99);
  CHECK(r.es99 > r.es95);
}

TEST(diversification_lowers_var) {
  Portfolio pf;
  pf.assets = {{"A", "equity", 100, 0.2}, {"B", "equity", 100, 0.2}};
  pf.positions = {{0, Position::Kind::Underlying, 500}, {1, Position::Kind::Underlying, 500}};
  VarConfig cfg;
  cfg.paths = 100'000;
  pf.correlation = {{1, 0.95}, {0.95, 1}};
  const double correlated = assess(pf, cfg).var99;
  pf.correlation = {{1, 0}, {0, 1}};
  const double independent = assess(pf, cfg).var99;
  CHECK(independent < 0.8 * correlated);
}

TEST(options_are_revalued_not_linearized) {
  // A long straddle loses at most its time value over a day, far less than an
  // equal-delta-notional stock position would.
  Portfolio pf;
  pf.assets = {{"A", "equity", 100, 0.3}};
  pf.positions = {{0, Position::Kind::Option, 10, OptionType::Call, 100, 0.1, 0.3, 100},
                  {0, Position::Kind::Option, 10, OptionType::Put, 100, 0.1, 0.3, 100}};
  const RiskReport r = assess(pf);
  const double premium = portfolio_value(pf);
  CHECK(r.var99 > 0);
  CHECK(r.var99 < 0.05 * premium);  // one day of theta, roughly
}

TEST(exposure_measures) {
  Portfolio pf;
  pf.assets = {{"A", "equity", 50, 0.2}, {"B", "rates", 200, 0.1}};
  pf.positions = {{0, Position::Kind::Underlying, 1000}, {1, Position::Kind::Underlying, -100}};
  pf.cash = 100'000;
  const RiskReport r = assess(pf);
  CHECK_NEAR(r.value, 100'000 + 50'000 - 20'000, 1e-9);
  CHECK_NEAR(r.net_delta_notional, 30'000, 1e-9);
  CHECK_NEAR(r.gross_notional, 70'000, 1e-9);
  CHECK_NEAR(r.class_share.at("equity"), 50.0 / 70.0, 1e-12);
}

TEST(limits_and_circuit_breaker) {
  RiskReport r;
  r.value = 1'000'000;
  r.net_delta_notional = 2'000'000;
  r.gross_notional = 5'000'000;
  r.gross_leverage = 5.0;
  r.var99 = 50'000;
  r.class_share = {{"equity", 0.9}, {"rates", 0.1}};
  const RiskLimits lim{1.5, 4.0, 0.8, 0.10};
  const auto breaches = check_limits(r, lim);
  CHECK(breaches.size() == 3);  // net delta, leverage, concentration; VaR 5% is within 10%

  CircuitBreaker cb(lim);
  CHECK(!cb.evaluate(r));
  CHECK(cb.halted());
  RiskReport calm;
  calm.value = 1'000'000;
  CHECK(!cb.evaluate(calm));  // latched until reset
  cb.reset();
  CHECK(cb.evaluate(calm));
}

TEST(var_is_deterministic_across_threads) {
  VarConfig a;
  a.paths = 50'000;
  a.threads = 1;
  VarConfig b = a;
  b.threads = 6;
  const auto pf = single_stock(1000);
  CHECK(assess(pf, a).var99 == assess(pf, b).var99);
}
