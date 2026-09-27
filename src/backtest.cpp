#include "ohe/backtest.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <sstream>
#include <stdexcept>

#include "ohe/rng.hpp"

namespace ohe {

namespace {

struct Leg {
  OptionType type;
  double strike;
  int expiry;        // bar index at which the option settles
  double contracts;  // signed
  double iv = 0;
  Greeks g{};        // per option, at the last mark
  double value = 0;  // contracts x multiplier x price
};

std::string fmt(const char* f, double a, double b) {
  char buf[128];
  std::snprintf(buf, sizeof buf, f, a, b);
  return buf;
}

}  // namespace

Decision dispatch_strategy(std::span<const double> history, double implied_atm, double equity,
                           const BacktestConfig& cfg) {
  Decision d;
  d.features = compute_features(history, cfg.features);
  d.implied = implied_atm;
  d.forecast = d.features.ewma_vol;
  d.edge = d.implied - d.forecast;
  const double S = history.back();
  d.strike = std::max(cfg.strike_step, std::round(S / cfg.strike_step) * cfg.strike_step);

  if (d.edge > cfg.entry_edge) {
    d.side = -1;
    d.reason = fmt("sell vol: implied %.4f > forecast %.4f", d.implied, d.forecast);
  } else if (cfg.allow_long_vol && d.edge < -cfg.entry_edge) {
    d.side = +1;
    d.reason = fmt("buy vol: implied %.4f < forecast %.4f", d.implied, d.forecast);
  } else {
    d.reason = fmt("no trade: implied %.4f vs forecast %.4f", d.implied, d.forecast);
    return d;
  }

  const double tau = cfg.tenor_days / cfg.trading_days;
  const double vega = black_scholes(OptionType::Call, S, d.strike, cfg.rate, cfg.dividend_yield,
                                    d.implied, tau).vega;
  const double straddle_vega_per_point = 2.0 * cfg.multiplier * vega * 0.01;
  const double n = straddle_vega_per_point > 0 ? std::floor(cfg.vega_budget * equity / straddle_vega_per_point) : 0;
  d.contracts = static_cast<int>(std::clamp(n, 0.0, static_cast<double>(cfg.max_contracts)));
  if (d.contracts == 0) {
    d.side = 0;
    d.reason += " (size rounds to zero)";
  }
  return d;
}

BacktestResult run_backtest(const std::vector<Bar>& bars, const BacktestConfig& cfg, Ledger* ledger) {
  const int n = static_cast<int>(bars.size());
  if (n < cfg.warmup + cfg.tenor_days + 2) {
    throw std::invalid_argument("not enough bars for warmup plus one option tenor");
  }
  BacktestResult res;
  BacktestSummary& sum = res.summary;
  const ExecutionModel& ex = cfg.execution;
  const double dt = 1.0 / cfg.trading_days;
  const double mult = cfg.multiplier;

  Pcg64 rng = Pcg64::from_seed(cfg.seed, 0x5eedULL);
  CircuitBreaker breaker(cfg.limits);
  std::vector<double> closes;
  closes.reserve(bars.size());
  std::vector<Leg> legs;
  double cash = cfg.initial_capital, shares = 0;
  double prev_spot = 0, book_prev = 0;
  std::vector<double> curve;
  sum.start_equity = cfg.initial_capital;

  auto leg_name = [&](const Leg& l) {
    char buf[96];
    std::snprintf(buf, sizeof buf, "%c %.2f %s", l.type == OptionType::Call ? 'C' : 'P', l.strike,
                  bars[static_cast<std::size_t>(l.expiry)].date.c_str());
    return std::string(buf);
  };
  auto emit_fill = [&](FillRecord f) {
    if (ledger) ledger->record(f);
    res.fills.push_back(std::move(f));
  };
  auto emit_event = [&](int t, const std::string& kind, const std::string& msg) {
    EventRecord e{cfg.run_id, t, bars[static_cast<std::size_t>(t)].date, kind, msg};
    if (ledger) ledger->record(e);
    res.events.push_back(std::move(e));
  };
  auto mark = [&](Leg& leg, int t, double S, const VolSurface& surf) {
    const double tau = std::max(leg.expiry - t, 0) / cfg.trading_days;
    leg.iv = surf.iv(leg.strike, std::max(tau, dt), S);
    leg.g = black_scholes(leg.type, S, leg.strike, cfg.rate, cfg.dividend_yield, leg.iv, tau);
    leg.value = leg.contracts * mult * leg.g.price;
  };
  auto trade_underlying = [&](int t, double qty, double S, double& costs) {
    if (qty == 0) return;
    const double sgn = qty > 0 ? 1.0 : -1.0;
    const double noise = S * ex.fill_noise_bps * 1e-4 * rng.normal();
    const double fill = S * (1.0 + sgn * ex.slippage_bps * 1e-4) + noise;
    const double fee = std::abs(qty) * ex.fee_per_share;
    const double slip = qty * (fill - S);
    cash -= qty * fill + fee;
    shares += qty;
    costs += fee + slip;
    ++sum.hedge_trades;
    emit_fill({cfg.run_id, t, bars[static_cast<std::size_t>(t)].date, "UNDERLYING", qty, fill, fee, slip});
  };
  auto trade_option = [&](int t, const Leg& leg, double qty, double& costs) {
    const double half = ex.option_half_spread * leg.g.vega;
    const double sgn = qty > 0 ? 1.0 : -1.0;
    const double fill = std::max(leg.g.price + sgn * half, 0.0);
    const double fee = std::abs(qty) * ex.fee_per_contract;
    const double slip = qty * mult * (fill - leg.g.price);
    cash -= qty * mult * fill + fee;
    costs += fee + slip;
    emit_fill({cfg.run_id, t, bars[static_cast<std::size_t>(t)].date, leg_name(leg), qty, fill, fee, slip});
  };
  auto close_book = [&](int t, double& costs) {
    for (const Leg& leg : legs) {
      if (leg.contracts != 0) trade_option(t, leg, -leg.contracts, costs);
    }
    legs.clear();
  };
  auto make_portfolio = [&](double S, double atm_iv, const std::vector<Leg>& book, double sh, double c, int t) {
    Portfolio pf;
    pf.assets.push_back({"UNDERLYING", "equity", S, atm_iv, cfg.dividend_yield});
    pf.rate = cfg.rate;
    pf.cash = c;
    if (sh != 0) pf.positions.push_back({0, Position::Kind::Underlying, sh});
    for (const Leg& l : book) {
      pf.positions.push_back({0, Position::Kind::Option, l.contracts, l.type, l.strike,
                              std::max(l.expiry - t, 0) / cfg.trading_days, l.iv, mult});
    }
    return pf;
  };

  for (int t = 0; t < n; ++t) {
    const Bar& bar = bars[static_cast<std::size_t>(t)];
    const double S = bar.close;
    closes.push_back(S);
    DayRecord day;
    day.run = cfg.run_id;
    day.day = t;
    day.date = bar.date;
    day.spot = S;
    double costs = 0;

    // Implied vol: from the data when present, else modeled from realized vol.
    double atm_iv = bar.iv;
    if (std::isnan(atm_iv)) {
      const auto r = log_returns(closes);
      atm_iv = r.size() >= 10 ? realized_vol(r, cfg.modeled_iv_window, cfg.trading_days) *
                                    (1.0 + cfg.modeled_iv_premium)
                              : std::nan("");
      if (t >= cfg.warmup) sum.implied_modeled = true;
    }
    const bool have_iv = std::isfinite(atm_iv) && atm_iv > 0;
    day.implied_vol = have_iv ? atm_iv : 0.0;

    SurfaceParams sp = cfg.surface;
    sp.atm_short = sp.atm_long = have_iv ? atm_iv : 0.2;
    const double tenor = cfg.tenor_days / cfg.trading_days;
    const VolSurface surf = VolSurface::parametric(S, sp, {dt, tenor, 3 * tenor});

    // 1. Mark to market and attribute yesterday's book.
    if (t > 0) {
      const double interest = cash * cfg.rate * dt;
      cash += interest;
      sum.interest += interest;
      day.hedge_pnl = shares * (S - prev_spot);
      const double dS = S - prev_spot;
      double book = 0;
      for (Leg& leg : legs) {
        const double c = leg.contracts * mult;
        const Greeks before = leg.g;
        const double iv_before = leg.iv;
        mark(leg, t, S, surf);
        day.delta_pnl += c * before.delta * dS;
        day.gamma_pnl += 0.5 * c * before.gamma * dS * dS;
        day.theta_pnl += c * before.theta * dt;
        day.vega_pnl += c * before.vega * (leg.iv - iv_before);
        book += leg.value;
      }
      day.option_pnl = book - book_prev;
    }

    // 2. Settle expiries (marked at intrinsic above).
    for (auto it = legs.begin(); it != legs.end();) {
      if (it->expiry <= t) {
        cash += it->value;
        emit_event(t, "expiry", leg_name(*it) + " settled");
        it = legs.erase(it);
      } else {
        ++it;
      }
    }

    // 3. Risk engine and circuit breaker on the live book.
    if (cfg.risk_checks && have_iv && !breaker.halted() && (!legs.empty() || shares != 0)) {
      VarConfig vc = cfg.var;
      vc.seed = cfg.var.seed + static_cast<std::uint64_t>(t);
      const RiskReport rep = assess(make_portfolio(S, atm_iv, legs, shares, cash, t), vc);
      if (!breaker.evaluate(rep)) {
        for (const std::string& why : breaker.reasons()) emit_event(t, "halt", why);
        close_book(t, costs);
        sum.halted = true;
        sum.halt_reasons = breaker.reasons();
      }
    }

    // 4. Strategy dispatch when flat.
    if (!breaker.halted() && legs.empty() && have_iv && t >= cfg.warmup && t + cfg.tenor_days < n) {
      const double equity_now = cash + shares * S;
      const Decision d = dispatch_strategy(closes, atm_iv, equity_now, cfg);
      if (d.side != 0) {
        std::vector<Leg> cand;
        for (OptionType ty : {OptionType::Call, OptionType::Put}) {
          Leg leg{ty, d.strike, t + cfg.tenor_days, static_cast<double>(d.side * d.contracts), 0.0, {}, 0.0};
          mark(leg, t, S, surf);
          cand.push_back(leg);
        }
        bool ok = true;
        if (cfg.risk_checks && cfg.pre_trade_checks) {
          // Pre-trade check on the book as it would stand once delta-hedged.
          double delta = 0, value = 0;
          for (const Leg& l : cand) {
            delta += l.contracts * mult * l.g.delta;
            value += l.value;
          }
          const double hedge = -std::round(delta);
          VarConfig vc = cfg.var;
          vc.seed = cfg.var.seed + static_cast<std::uint64_t>(t) + 0x9e37ULL;
          const auto breaches = check_limits(
              assess(make_portfolio(S, atm_iv, cand, hedge, equity_now - value - hedge * S, t), vc),
              cfg.limits);
          if (!breaches.empty()) {
            ok = false;
            ++sum.risk_rejections;
            emit_event(t, "risk_reject", breaches.front().limit + " would be breached");
          }
        }
        if (ok) {
          for (const Leg& leg : cand) trade_option(t, leg, leg.contracts, costs);
          legs = std::move(cand);
          ++sum.option_trades;
          emit_event(t, "open", d.reason + ", " + std::to_string(d.contracts) + " straddles");
        }
      }
    }

    // 5. Delta hedge through the execution model (flatten when the book is empty).
    double book_delta = 0, gross = 0;
    for (const Leg& l : legs) {
      book_delta += l.contracts * mult * l.g.delta;
      gross += std::abs(l.contracts) * mult;
    }
    const double target = legs.empty() ? 0.0 : std::round(-book_delta);
    if (std::abs(target - shares) > (legs.empty() ? 0.0 : cfg.hedge_band * gross)) {
      trade_underlying(t, target - shares, S, costs);
    }

    // Last bar: close everything so the result is fully realized.
    if (t == n - 1) {
      close_book(t, costs);
      trade_underlying(t, -shares, S, costs);
    }

    book_prev = 0;
    book_delta = 0;
    for (const Leg& l : legs) {
      book_prev += l.value;
      book_delta += l.contracts * mult * l.g.delta;
    }
    prev_spot = S;
    day.costs = costs;
    day.equity = cash + shares * S + book_prev;
    day.net_delta = book_delta + shares;
    day.contracts = legs.empty() ? 0 : static_cast<int>(std::abs(legs.front().contracts));
    day.halted = breaker.halted();

    sum.costs += costs;
    sum.option_pnl += day.option_pnl;
    sum.hedge_pnl += day.hedge_pnl;
    sum.delta_pnl += day.delta_pnl;
    sum.gamma_pnl += day.gamma_pnl;
    sum.theta_pnl += day.theta_pnl;
    sum.vega_pnl += day.vega_pnl;
    if (t >= cfg.warmup - 1) curve.push_back(day.equity);
    if (ledger) ledger->record(day);
    res.days.push_back(std::move(day));
  }
  if (ledger) ledger->flush();

  sum.days = n;
  sum.end_equity = res.days.back().equity;
  sum.total_return = sum.end_equity / sum.start_equity - 1.0;
  sum.residual_pnl = sum.option_pnl - (sum.delta_pnl + sum.gamma_pnl + sum.theta_pnl + sum.vega_pnl);

  // Performance statistics over the trading period (after warmup).
  std::vector<double> rets;
  for (std::size_t i = 1; i < curve.size(); ++i) rets.push_back(curve[i] / curve[i - 1] - 1.0);
  if (!rets.empty()) {
    double mean = 0;
    for (double r : rets) mean += r;
    mean /= static_cast<double>(rets.size());
    double var = 0;
    for (double r : rets) var += (r - mean) * (r - mean);
    var /= static_cast<double>(std::max<std::size_t>(1, rets.size() - 1));
    const double sd = std::sqrt(var);
    sum.annual_vol = sd * std::sqrt(cfg.trading_days);
    sum.annual_return = std::pow(curve.back() / curve.front(), cfg.trading_days / static_cast<double>(rets.size())) - 1.0;
    sum.sharpe = sd > 0 ? (mean - cfg.rate / cfg.trading_days) / sd * std::sqrt(cfg.trading_days) : 0.0;
    double peak = curve.front();
    for (double e : curve) {
      peak = std::max(peak, e);
      sum.max_drawdown = std::max(sum.max_drawdown, 1.0 - e / peak);
    }
  }
  return res;
}

}  // namespace ohe
