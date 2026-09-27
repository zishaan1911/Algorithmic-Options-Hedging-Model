// ohe: command-line front end for the options hedging engine.

#include <chrono>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <map>
#include <memory>
#include <optional>
#include <sstream>
#include <string>
#include <vector>

#include "ohe/backtest.hpp"
#include "ohe/data.hpp"
#include "ohe/ledger.hpp"
#include "ohe/models.hpp"
#include "ohe/rng.hpp"
#include "ohe/risk.hpp"
#include "ohe/vol_surface.hpp"

namespace {

using namespace ohe;

constexpr const char* kUsage = R"(ohe - options pricing, risk and delta-hedging backtests

usage:
  ohe price     --type call|put --spot S --strike K --tau T [--rate r] [--div q] [--vol v]
                [--heston v0,kappa,theta,xi,rho] [--jumps lambda,mu,delta]
                [--mc PATHS] [--seed N] [--threads N]
  ohe surface   --spot S --atm V [--skew b] [--smile c] [--term-long V] [--decay Y]
  ohe simulate  --out FILE [--days N] [--spot S] [--heston ...] [--premium p] [--seed N]
  ohe backtest  --prices FILE [--iv FILE --iv-scale 0.01] [--seed N] [--ledger DB]
                [--equity-out FILE] [--tenor DAYS] [--edge V] [--no-long] [--no-risk]
                [--vega-budget F] [--slippage-bps B] [--run ID] [--model-iv PREMIUM]
  ohe risk      [--paths N] [--seed N]          VaR/ES of a sample multi-asset book
  ohe ledger    --db DB [--run ID]              read back a run's fills and events

Seeds may be integers or any text ("2024-q1"); results are reproducible per seed.
)";

struct Args {
  std::string command;
  std::map<std::string, std::string> opts;
  bool has(const std::string& k) const { return opts.count(k) > 0; }
  std::string str(const std::string& k, const std::string& def = "") const {
    const auto it = opts.find(k);
    return it == opts.end() ? def : it->second;
  }
  double num(const std::string& k, double def) const { return has(k) ? std::stod(str(k)) : def; }
  std::vector<double> list(const std::string& k) const {
    std::vector<double> out;
    std::stringstream ss(str(k));
    for (std::string item; std::getline(ss, item, ',');) out.push_back(std::stod(item));
    return out;
  }
};

Args parse(int argc, char** argv) {
  Args a;
  if (argc > 1) a.command = argv[1];
  for (int i = 2; i < argc; ++i) {
    std::string key = argv[i];
    if (key.rfind("--", 0) != 0) throw std::invalid_argument("unexpected argument: " + key);
    key = key.substr(2);
    if (i + 1 < argc && std::string(argv[i + 1]).rfind("--", 0) != 0) a.opts[key] = argv[++i];
    else a.opts[key] = "true";
  }
  return a;
}

std::uint64_t seed_of(const Args& a, std::uint64_t def) {
  return a.has("seed") ? normalize_seed(a.str("seed")) : def;
}

std::optional<HestonParams> heston_of(const Args& a) {
  if (!a.has("heston")) return std::nullopt;
  const auto v = a.list("heston");
  if (v.size() != 5) throw std::invalid_argument("--heston takes v0,kappa,theta,xi,rho");
  return HestonParams{v[0], v[1], v[2], v[3], v[4]};
}

int cmd_price(const Args& a) {
  const OptionType type = a.str("type", "call") == "put" ? OptionType::Put : OptionType::Call;
  const double S = a.num("spot", 100), K = a.num("strike", 100), tau = a.num("tau", 0.5);
  const double r = a.num("rate", 0.02), q = a.num("div", 0.0);
  Model model;
  model.sigma = a.num("vol", 0.2);
  model.heston = heston_of(a);
  if (a.has("jumps")) {
    const auto v = a.list("jumps");
    if (v.size() != 3) throw std::invalid_argument("--jumps takes lambda,mu,delta");
    model.jumps = JumpParams{v[0], v[1], v[2]};
  }
  McConfig mc;
  mc.paths = static_cast<std::size_t>(a.num("mc", 200'000));
  mc.seed = seed_of(a, 42);
  mc.threads = static_cast<unsigned>(a.num("threads", 0));
  const auto policy = a.has("mc") ? PricingPolicy::MonteCarlo : PricingPolicy::Auto;

  const auto t0 = std::chrono::steady_clock::now();
  const PriceResult pr = price_option(type, S, K, r, q, tau, model, policy, mc);
  const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
  std::printf("method      %s\nprice       %.6f\n", pr.method.c_str(), pr.price);
  if (pr.std_error > 0) std::printf("std error   %.6f  (%zu paths)\n", pr.std_error, mc.paths);
  std::printf("time        %.1f ms\n", ms);
  if (!model.heston && !model.jumps) {
    const Greeks g = black_scholes(type, S, K, r, q, model.sigma, tau);
    std::printf("delta       %.6f\ngamma       %.6f\nvega        %.6f  per 1.00 vol\n"
                "theta       %.6f  per year\nrho         %.6f\n",
                g.delta, g.gamma, g.vega, g.theta, g.rho);
  }
  return 0;
}

int cmd_surface(const Args& a) {
  SurfaceParams p;
  p.atm_short = a.num("atm", 0.2);
  p.atm_long = a.num("term-long", p.atm_short);
  p.term_decay = a.num("decay", 0.5);
  p.skew = a.num("skew", -0.10);
  p.smile = a.num("smile", 0.02);
  const double S = a.num("spot", 100);
  const std::vector<double> expiries{1.0 / 12, 0.25, 0.5, 1.0, 2.0};
  const VolSurface surf = VolSurface::parametric(S, p, expiries);
  std::printf("implied vol (%%) by strike / expiry, spot %.2f\n\n  strike", S);
  for (double t : expiries) std::printf("   %5.2fy", t);
  std::printf("\n");
  for (double m = 0.7; m <= 1.301; m += 0.05) {
    std::printf("  %6.1f", S * m);
    for (double t : expiries) std::printf("   %6.2f", 100 * surf.iv(S * m, t));
    std::printf("\n");
  }
  std::printf("\ncalendar-arbitrage violations: %d\n", surf.calendar_violations());
  return 0;
}

int cmd_simulate(const Args& a) {
  SimConfig c;
  c.days = static_cast<int>(a.num("days", 1000));
  c.spot = a.num("spot", 100);
  c.iv_premium = a.num("premium", 0.10);
  c.seed = seed_of(a, 1);
  if (auto h = heston_of(a)) c.heston = *h;
  const std::string out = a.str("out", "simulated.csv");
  save_bars(out, simulate_market(c));
  std::printf("wrote %d days of Heston prices with a %.0f%% implied-vol premium to %s\n", c.days,
              100 * c.iv_premium, out.c_str());
  return 0;
}

int cmd_backtest(const Args& a) {
  if (!a.has("prices")) throw std::invalid_argument("--prices FILE is required");
  std::vector<Bar> bars = load_bars(a.str("prices"));
  if (a.has("iv")) bars = join_implied_vol(bars, load_bars(a.str("iv")), a.num("iv-scale", 0.01));

  bool have_iv = false;
  for (const Bar& b : bars) have_iv |= !std::isnan(b.iv);

  BacktestConfig cfg;
  if (!have_iv) {
    // Without implied vol data the "market" price of options has to be invented,
    // and any premium assumed here is simply what the strategy will harvest.
    if (!a.has("model-iv")) {
      throw std::invalid_argument(
          "no implied vol in the data; pass --iv FILE, or --model-iv PREMIUM to price options at "
          "realized vol x (1 + PREMIUM) (results then only reflect that assumption)");
    }
    cfg.modeled_iv_premium = a.num("model-iv", cfg.modeled_iv_premium);
  }
  cfg.seed = seed_of(a, 42);
  cfg.run_id = a.str("run", "backtest");
  cfg.tenor_days = static_cast<int>(a.num("tenor", cfg.tenor_days));
  cfg.entry_edge = a.num("edge", cfg.entry_edge);
  cfg.allow_long_vol = !a.has("no-long");
  cfg.risk_checks = !a.has("no-risk");
  cfg.vega_budget = a.num("vega-budget", cfg.vega_budget);
  cfg.execution.slippage_bps = a.num("slippage-bps", cfg.execution.slippage_bps);

  std::unique_ptr<Ledger> ledger;
  if (a.has("ledger")) ledger = std::make_unique<Ledger>(a.str("ledger"));
  const auto t0 = std::chrono::steady_clock::now();
  const BacktestResult r = run_backtest(bars, cfg, ledger.get());
  const double secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
  const BacktestSummary& s = r.summary;

  std::printf("run            %s (%d bars, %s to %s, %.2fs)\n", cfg.run_id.c_str(), s.days,
              bars.front().date.c_str(), bars.back().date.c_str(), secs);
  std::printf("implied vol    %s\n", s.implied_modeled ? "MODELED from realized vol (no IV data)"
                                                         : "from data");
  std::printf("equity         %.0f -> %.0f  (%+.2f%%)\n", s.start_equity, s.end_equity, 100 * s.total_return);
  std::printf("annualized     return %+.2f%%  vol %.2f%%  sharpe %.2f  max drawdown %.2f%%\n",
              100 * s.annual_return, 100 * s.annual_vol, s.sharpe, 100 * s.max_drawdown);
  std::printf("trades         %d straddles, %d hedge fills, %d risk rejections\n", s.option_trades,
              s.hedge_trades, s.risk_rejections);
  std::printf("p&l            options %+.0f  hedge %+.0f  interest %+.0f  costs %-.0f\n", s.option_pnl,
              s.hedge_pnl, s.interest, -s.costs);
  std::printf("attribution    delta %+.0f  gamma %+.0f  theta %+.0f  vega %+.0f  residual %+.0f\n",
              s.delta_pnl, s.gamma_pnl, s.theta_pnl, s.vega_pnl, s.residual_pnl);
  if (s.halted) {
    std::printf("HALTED         ");
    for (const auto& why : s.halt_reasons) std::printf("%s; ", why.c_str());
    std::printf("\n");
  }
  if (a.has("equity-out")) {
    std::ofstream out(a.str("equity-out"));
    out << "date,spot,implied_vol,equity,option_pnl,hedge_pnl,costs,delta_pnl,gamma_pnl,theta_pnl,vega_pnl,contracts,halted\n";
    for (const DayRecord& d : r.days) {
      out << d.date << ',' << d.spot << ',' << d.implied_vol << ',' << d.equity << ',' << d.option_pnl << ','
          << d.hedge_pnl << ',' << d.costs << ',' << d.delta_pnl << ',' << d.gamma_pnl << ',' << d.theta_pnl
          << ',' << d.vega_pnl << ',' << d.contracts << ',' << d.halted << '\n';
    }
  }
  if (ledger) std::printf("ledger         %s (run '%s')\n", a.str("ledger").c_str(), cfg.run_id.c_str());
  return 0;
}

int cmd_risk(const Args& a) {
  Portfolio pf;
  pf.assets = {{"SPX", "equity-index", 5000, 0.18}, {"NDX", "equity-index", 18000, 0.24},
               {"GLD", "commodity", 190, 0.15}, {"TLT", "rates", 95, 0.16}};
  pf.correlation = {{1.0, 0.9, 0.05, -0.2}, {0.9, 1.0, 0.0, -0.2}, {0.05, 0.0, 1.0, 0.25}, {-0.2, -0.2, 0.25, 1.0}};
  pf.rate = 0.03;
  pf.cash = 2'000'000;
  using K = Position::Kind;
  pf.positions = {
      {0, K::Underlying, 200},
      {0, K::Option, -40, OptionType::Put, 4750, 30 / 365.0, 0.21, 100},
      {1, K::Option, 25, OptionType::Call, 18500, 60 / 365.0, 0.25, 20},
      {2, K::Underlying, 5000},
      {3, K::Underlying, -4000},
  };
  VarConfig vc;
  vc.paths = static_cast<std::size_t>(a.num("paths", 200'000));
  vc.seed = seed_of(a, 7);
  const RiskReport r = assess(pf, vc);
  std::printf("portfolio value   %.0f\nnet delta         %.0f\ngross notional    %.0f  (leverage %.2fx)\n",
              r.value, r.net_delta_notional, r.gross_notional, r.gross_leverage);
  std::printf("1-day VaR 95/99   %.0f / %.0f\n1-day ES  95/99   %.0f / %.0f\n", r.var95, r.var99, r.es95, r.es99);
  for (const auto& [cls, share] : r.class_share) std::printf("  %-14s %5.1f%% of gross\n", cls.c_str(), 100 * share);
  CircuitBreaker cb{RiskLimits{}};
  if (cb.evaluate(r)) std::printf("circuit breaker   ok\n");
  else
    for (const auto& why : cb.reasons()) std::printf("circuit breaker   HALT: %s\n", why.c_str());
  return 0;
}

int cmd_ledger(const Args& a) {
  if (!a.has("db")) throw std::invalid_argument("--db DB is required");
  const std::string run = a.str("run", "backtest");
  const auto fills = Ledger::read_fills(a.str("db"), run);
  const auto events = Ledger::read_events(a.str("db"), run);
  std::printf("run '%s': %d days, %zu fills, %zu events\n", run.c_str(), Ledger::count_days(a.str("db"), run),
              fills.size(), events.size());
  for (const auto& e : events) std::printf("  %s  %-11s %s\n", e.date.c_str(), e.kind.c_str(), e.message.c_str());
  return 0;
}

}  // namespace

int main(int argc, char** argv) {
  try {
    const Args a = parse(argc, argv);
    if (a.command == "price") return cmd_price(a);
    if (a.command == "surface") return cmd_surface(a);
    if (a.command == "simulate") return cmd_simulate(a);
    if (a.command == "backtest") return cmd_backtest(a);
    if (a.command == "risk") return cmd_risk(a);
    if (a.command == "ledger") return cmd_ledger(a);
    std::fputs(kUsage, a.command.empty() || a.command == "help" || a.command == "--help" ? stdout : stderr);
    return a.command.empty() || a.command == "help" || a.command == "--help" ? 0 : 2;
  } catch (const std::exception& e) {
    std::fprintf(stderr, "error: %s\n", e.what());
    return 1;
  }
}
