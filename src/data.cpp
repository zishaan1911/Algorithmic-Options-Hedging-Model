#include "ohe/data.hpp"

#include <algorithm>
#include <cctype>
#include <charconv>
#include <cstdio>
#include <fstream>
#include <map>
#include <sstream>
#include <stdexcept>

#include "ohe/rng.hpp"

namespace ohe {

namespace {

std::vector<std::string> split_csv(const std::string& line) {
  std::vector<std::string> out;
  std::string cur;
  bool quoted = false;
  for (char c : line) {
    if (c == '"') quoted = !quoted;
    else if (c == ',' && !quoted) {
      out.push_back(cur);
      cur.clear();
    } else if (c != '\r') cur.push_back(c);
  }
  out.push_back(cur);
  return out;
}

std::string lower(std::string s) {
  std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
  return s;
}

bool parse_double(const std::string& s, double& out) {
  if (s.empty()) return false;
  const auto [ptr, ec] = std::from_chars(s.data(), s.data() + s.size(), out);
  return ec == std::errc{} && ptr == s.data() + s.size();
}

// Accepts YYYY-MM-DD and YYYYMMDD; returns YYYY-MM-DD.
std::string normalize_date(const std::string& s) {
  if (s.size() == 8 && std::all_of(s.begin(), s.end(), [](unsigned char c) { return std::isdigit(c) != 0; })) {
    return s.substr(0, 4) + "-" + s.substr(4, 2) + "-" + s.substr(6, 2);
  }
  return s.substr(0, 10);
}

}  // namespace

std::vector<Bar> load_bars(const std::string& path) {
  std::ifstream in(path);
  if (!in) throw std::runtime_error("cannot open " + path);
  std::string line;
  if (!std::getline(in, line)) throw std::runtime_error(path + " is empty");
  const auto header = split_csv(line);
  int date_col = -1, close_col = -1, adj_col = -1, iv_col = -1;
  for (int i = 0; i < static_cast<int>(header.size()); ++i) {
    const std::string h = lower(header[i]);
    if (h == "date" || h == "observation_date") date_col = i;
    else if (h == "close") close_col = i;
    else if (h == "adj close" || h == "adj_close") adj_col = i;
    else if (h == "iv") iv_col = i;
  }
  if (adj_col >= 0) close_col = adj_col;
  // FRED series (observation_date,SERIES_ID): the single value column is the close.
  if (close_col < 0 && date_col >= 0 && header.size() == 2) close_col = 1 - date_col;
  if (date_col < 0 || close_col < 0) throw std::runtime_error(path + ": need Date and Close columns");

  std::vector<Bar> bars;
  while (std::getline(in, line)) {
    if (line.empty()) continue;
    const auto f = split_csv(line);
    if (static_cast<int>(f.size()) <= std::max(date_col, close_col)) continue;
    Bar b;
    b.date = normalize_date(f[date_col]);
    if (!parse_double(f[close_col], b.close) || b.close <= 0) continue;  // skips "null" rows
    if (iv_col >= 0 && iv_col < static_cast<int>(f.size())) parse_double(f[iv_col], b.iv);
    bars.push_back(std::move(b));
  }
  std::sort(bars.begin(), bars.end(), [](const Bar& a, const Bar& b) { return a.date < b.date; });
  if (bars.empty()) throw std::runtime_error(path + ": no price rows");
  return bars;
}

std::vector<Bar> join_implied_vol(const std::vector<Bar>& bars, const std::vector<Bar>& iv_bars,
                                  double scale) {
  std::map<std::string, double> iv;
  for (const Bar& b : iv_bars) iv[b.date] = b.close * scale;
  std::vector<Bar> out;
  for (Bar b : bars) {
    const auto it = iv.find(b.date);
    if (it == iv.end()) continue;
    b.iv = it->second;
    out.push_back(std::move(b));
  }
  return out;
}

void save_bars(const std::string& path, const std::vector<Bar>& bars) {
  std::ofstream out(path);
  if (!out) throw std::runtime_error("cannot write " + path);
  out << "Date,Close,IV\n";
  out.precision(10);
  for (const Bar& b : bars) {
    out << b.date << ',' << b.close << ',';
    if (!std::isnan(b.iv)) out << b.iv;
    out << '\n';
  }
}

std::vector<Bar> simulate_market(const SimConfig& cfg) {
  Pcg64 rng = Pcg64::from_seed(cfg.seed, 0);
  const HestonParams& h = cfg.heston;
  const double dt = 1.0 / (252.0 * cfg.steps_per_day);
  const double tenor = cfg.iv_tenor_days / 252.0;
  const double rho_c = std::sqrt(1.0 - h.rho * h.rho);

  // Expected average variance over the next `tenor` years given v now.
  auto expected_vol = [&](double v) {
    const double kt = h.kappa * tenor;
    const double avg = h.theta + (v - h.theta) * (1.0 - std::exp(-kt)) / kt;
    return std::sqrt(std::max(avg, 1e-8));
  };

  std::vector<Bar> bars;
  bars.reserve(cfg.days);
  double s = cfg.spot, v = h.v0;
  for (int d = 0; d < cfg.days; ++d) {
    if (d > 0) {
      for (int k = 0; k < cfg.steps_per_day; ++k) {
        const double vp = std::max(v, 0.0);
        const double z1 = rng.normal();
        const double z2 = h.rho * z1 + rho_c * rng.normal();
        s *= std::exp((cfg.rate - 0.5 * vp) * dt + std::sqrt(vp * dt) * z1);
        v += h.kappa * (h.theta - vp) * dt + h.xi * std::sqrt(vp * dt) * z2;
      }
    }
    // Business-day style label; only ordering matters to the backtester.
    char date[16];
    std::snprintf(date, sizeof date, "D%06d", d);
    bars.push_back({date, s, expected_vol(std::max(v, 0.0)) * (1.0 + cfg.iv_premium)});
  }
  return bars;
}

}  // namespace ohe
