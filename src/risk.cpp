#include "ohe/risk.hpp"

#include <algorithm>
#include <cmath>
#include <sstream>
#include <stdexcept>

#include "ohe/parallel.hpp"
#include "ohe/rng.hpp"

namespace ohe {

double position_value(const Portfolio& pf, const Position& p, double spot, double time_elapsed) {
  if (p.kind == Position::Kind::Underlying) return p.quantity * spot;
  const Asset& a = pf.assets.at(p.asset);
  const double tau = std::max(p.expiry - time_elapsed, 0.0);
  return p.quantity * p.multiplier *
         bs_price(p.type, spot, p.strike, pf.rate, a.dividend_yield, p.implied_vol, tau);
}

double position_delta_shares(const Portfolio& pf, const Position& p, double spot) {
  if (p.kind == Position::Kind::Underlying) return p.quantity;
  const Asset& a = pf.assets.at(p.asset);
  return p.quantity * p.multiplier *
         black_scholes(p.type, spot, p.strike, pf.rate, a.dividend_yield, p.implied_vol, p.expiry).delta;
}

double portfolio_value(const Portfolio& pf) {
  double v = pf.cash;
  for (const Position& p : pf.positions) v += position_value(pf, p, pf.assets.at(p.asset).spot);
  return v;
}

std::vector<std::vector<double>> cholesky(const std::vector<std::vector<double>>& a) {
  const std::size_t n = a.size();
  std::vector<std::vector<double>> l(n, std::vector<double>(n, 0.0));
  for (std::size_t i = 0; i < n; ++i) {
    for (std::size_t j = 0; j <= i; ++j) {
      double s = a[i][j];
      for (std::size_t k = 0; k < j; ++k) s -= l[i][k] * l[j][k];
      if (i == j) {
        if (s <= 0) throw std::invalid_argument("correlation matrix is not positive definite");
        l[i][i] = std::sqrt(s);
      } else {
        l[i][j] = s / l[j][j];
      }
    }
  }
  return l;
}

RiskReport assess(const Portfolio& pf, const VarConfig& cfg) {
  const std::size_t n = pf.assets.size();
  RiskReport rep;
  rep.value = portfolio_value(pf);

  std::map<std::string, double> class_gross;
  for (const Position& p : pf.positions) {
    const Asset& a = pf.assets.at(p.asset);
    const double notional = position_delta_shares(pf, p, a.spot) * a.spot;
    rep.net_delta_notional += notional;
    rep.gross_notional += std::abs(notional);
    class_gross[a.asset_class] += std::abs(notional);
  }
  rep.gross_leverage = rep.value > 0 ? rep.gross_notional / rep.value : INFINITY;
  for (const auto& [cls, g] : class_gross) {
    rep.class_share[cls] = rep.gross_notional > 0 ? g / rep.gross_notional : 0.0;
  }
  if (n == 0 || pf.positions.empty()) return rep;

  std::vector<std::vector<double>> corr = pf.correlation;
  if (corr.empty()) {
    corr.assign(n, std::vector<double>(n, 0.0));
    for (std::size_t i = 0; i < n; ++i) corr[i][i] = 1.0;
  }
  const auto L = cholesky(corr);
  const double h = cfg.horizon_days / cfg.trading_days;

  constexpr std::size_t kBlock = 2048;
  const std::size_t blocks = (cfg.paths + kBlock - 1) / kBlock;
  std::vector<double> losses(blocks * kBlock);

  parallel_blocks(blocks, cfg.threads ? cfg.threads : default_threads(), [&](std::size_t b) {
    Pcg64 rng = Pcg64::from_seed(cfg.seed, b);
    std::vector<double> z(n), spots(n);
    for (std::size_t s = 0; s < kBlock; ++s) {
      for (double& v : z) v = rng.normal();
      for (std::size_t i = 0; i < n; ++i) {
        double w = 0;
        for (std::size_t k = 0; k <= i; ++k) w += L[i][k] * z[k];
        const Asset& a = pf.assets[i];
        spots[i] = a.spot * std::exp((pf.rate - a.dividend_yield - 0.5 * a.vol * a.vol) * h +
                                     a.vol * std::sqrt(h) * w);
      }
      double v = pf.cash * std::exp(pf.rate * h);
      for (const Position& p : pf.positions) v += position_value(pf, p, spots[p.asset], h);
      losses[b * kBlock + s] = rep.value - v;
    }
  });
  losses.resize(cfg.paths);

  std::sort(losses.begin(), losses.end());
  auto tail_stats = [&](double level, double& var, double& es) {
    const std::size_t idx =
        std::min(losses.size() - 1, static_cast<std::size_t>(level * static_cast<double>(losses.size())));
    var = losses[idx];
    double s = 0;
    for (std::size_t i = idx; i < losses.size(); ++i) s += losses[i];
    es = s / static_cast<double>(losses.size() - idx);
  };
  tail_stats(0.95, rep.var95, rep.es95);
  tail_stats(0.99, rep.var99, rep.es99);
  return rep;
}

std::vector<Breach> check_limits(const RiskReport& r, const RiskLimits& lim) {
  std::vector<Breach> out;
  const double equity = r.value;
  if (equity <= 0) {
    out.push_back({"equity", equity, 0.0});
    return out;
  }
  const double net = std::abs(r.net_delta_notional) / equity;
  if (net > lim.max_net_delta_fraction) out.push_back({"net_delta", net, lim.max_net_delta_fraction});
  if (r.gross_leverage > lim.max_gross_leverage) {
    out.push_back({"gross_leverage", r.gross_leverage, lim.max_gross_leverage});
  }
  for (const auto& [cls, share] : r.class_share) {
    // Concentration only means something when there is more than one class.
    if (r.class_share.size() > 1 && share > lim.max_class_share) {
      out.push_back({"concentration:" + cls, share, lim.max_class_share});
    }
  }
  const double var_frac = r.var99 / equity;
  if (var_frac > lim.max_var99_fraction) out.push_back({"var99", var_frac, lim.max_var99_fraction});
  return out;
}

bool CircuitBreaker::evaluate(const RiskReport& report) {
  for (const Breach& b : check_limits(report, limits_)) {
    std::ostringstream msg;
    msg << b.limit << " " << b.value << " > " << b.threshold;
    halt_trading(msg.str());
  }
  return !halted_;
}

void CircuitBreaker::halt_trading(std::string reason) {
  halted_ = true;
  reasons_.push_back(std::move(reason));
}

void CircuitBreaker::reset() {
  halted_ = false;
  reasons_.clear();
}

}  // namespace ohe
