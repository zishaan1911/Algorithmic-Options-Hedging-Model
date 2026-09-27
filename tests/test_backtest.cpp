#include <cmath>
#include <cstdio>
#include <filesystem>
#include <string>

#include "check.hpp"
#include "ohe/backtest.hpp"
#include "ohe/data.hpp"
#include "ohe/ledger.hpp"

using namespace ohe;

namespace {

std::vector<Bar> market(double premium, std::uint64_t seed, int days = 750) {
  SimConfig sc;
  sc.days = days;
  sc.iv_premium = premium;
  sc.seed = seed;
  return simulate_market(sc);
}

BacktestConfig fast_config() {
  BacktestConfig c;
  c.var.paths = 4'000;  // keep the per-day risk run cheap in tests
  return c;
}

}  // namespace

TEST(backtest_pnl_adds_up) {
  const auto r = run_backtest(market(0.1, 1), fast_config());
  const auto& s = r.summary;
  CHECK(s.option_trades > 5);
  const double explained = s.option_pnl + s.hedge_pnl + s.interest - s.costs;
  CHECK_NEAR(s.end_equity - s.start_equity, explained, 1e-6 * s.start_equity);
  // Every position is closed on the last bar.
  CHECK(r.days.back().contracts == 0);
  CHECK_NEAR(r.days.back().net_delta, 0.0, 1e-9);
}

TEST(backtest_is_deterministic_per_seed) {
  const auto bars = market(0.1, 2);
  auto cfg = fast_config();
  const auto a = run_backtest(bars, cfg);
  const auto b = run_backtest(bars, cfg);
  CHECK(a.days.size() == b.days.size());
  for (std::size_t i = 0; i < a.days.size(); ++i) CHECK(a.days[i].equity == b.days[i].equity);
  cfg.seed = 99;  // different fill noise
  const auto c = run_backtest(bars, cfg);
  CHECK(c.summary.end_equity != a.summary.end_equity);
}

TEST(short_vol_earns_the_premium_when_implied_exceeds_realized) {
  // In the simulated market implied vol is the true expected vol + 15%, so
  // selling delta-hedged straddles should earn theta in excess of gamma.
  auto cfg = fast_config();
  cfg.allow_long_vol = false;
  double pnl = 0;
  int runs = 0;
  for (std::uint64_t seed : {3, 4, 5, 6}) {
    const auto r = run_backtest(market(0.15, seed, 1000), cfg);
    CHECK(r.summary.theta_pnl > 0);
    CHECK(r.summary.gamma_pnl < 0);
    pnl += r.summary.option_pnl + r.summary.hedge_pnl - r.summary.costs;
    ++runs;
  }
  CHECK(pnl > 0);
}

TEST(attribution_explains_option_pnl) {
  auto cfg = fast_config();
  cfg.execution = ExecutionModel{0, 0, 0, 0, 0};
  const auto r = run_backtest(market(0.1, 7), cfg);
  const auto& s = r.summary;
  CHECK_NEAR(s.costs, 0.0, 1e-9);
  const double gross = std::abs(s.gamma_pnl) + std::abs(s.theta_pnl) + std::abs(s.vega_pnl);
  CHECK(std::abs(s.residual_pnl) < 0.25 * gross);
}

TEST(pre_trade_limits_reject_every_trade) {
  auto cfg = fast_config();
  cfg.limits.max_var99_fraction = 1e-7;
  const auto r = run_backtest(market(0.1, 8), cfg);
  CHECK(r.summary.option_trades == 0);
  CHECK(r.summary.risk_rejections > 0);
  CHECK_NEAR(r.summary.end_equity - r.summary.start_equity, r.summary.interest, 1e-6);
}

TEST(circuit_breaker_halts_and_flattens) {
  // Let the first trade through, then breach a limit on the live book: the
  // breaker must halt, close the options, flatten the hedge and stay halted.
  auto cfg = fast_config();
  cfg.pre_trade_checks = false;
  cfg.limits.max_gross_leverage = 1e-9;
  const auto r = run_backtest(market(0.1, 9), cfg);
  CHECK(r.summary.halted);
  CHECK(r.summary.option_trades == 1);
  int halt_day = -1;
  for (const auto& e : r.events) {
    if (e.kind == "halt" && halt_day < 0) halt_day = e.day;
  }
  CHECK(halt_day > cfg.warmup);
  for (const auto& d : r.days) {
    if (d.day >= halt_day) {
      CHECK(d.halted);
      CHECK(d.contracts == 0);
      CHECK_NEAR(d.net_delta, 0.0, 1e-9);
    }
  }
}

TEST(modeled_implied_vol_is_flagged) {
  auto bars = market(0.1, 10);
  for (Bar& b : bars) b.iv = std::nan("");
  const auto r = run_backtest(bars, fast_config());
  CHECK(r.summary.implied_modeled);
  CHECK(!run_backtest(market(0.1, 10), fast_config()).summary.implied_modeled);
}

TEST(ledger_records_the_run_in_wal_mode) {
  const auto path = (std::filesystem::temp_directory_path() / "ohe_ledger_test.db").string();
  for (const char* suffix : {"", "-wal", "-shm"}) std::filesystem::remove(path + suffix);
  const auto bars = market(0.1, 12, 300);
  auto cfg = fast_config();
  cfg.run_id = "t1";
  {
    Ledger ledger(path);
    const auto r = run_backtest(bars, cfg, &ledger);
    // Readers use their own connection while the writer is still open.
    CHECK(Ledger::count_days(path, "t1") == static_cast<int>(bars.size()));
    CHECK(Ledger::read_fills(path, "t1").size() == r.fills.size());
    CHECK(Ledger::read_events(path, "t1").size() == r.events.size());
    CHECK(std::filesystem::exists(path + "-wal"));
  }
  CHECK(Ledger::count_days(path, "t1") == static_cast<int>(bars.size()));
  for (const char* suffix : {"", "-wal", "-shm"}) std::filesystem::remove(path + suffix);
}

TEST(csv_round_trip) {
  const auto path = (std::filesystem::temp_directory_path() / "ohe_bars_test.csv").string();
  const auto bars = market(0.1, 13, 50);
  save_bars(path, bars);
  const auto back = load_bars(path);
  CHECK(back.size() == bars.size());
  CHECK_NEAR(back[10].close, bars[10].close, 1e-6 * bars[10].close);
  CHECK_NEAR(back[10].iv, bars[10].iv, 1e-9);
  std::filesystem::remove(path);
}
