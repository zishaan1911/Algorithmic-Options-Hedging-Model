// Market data: CSV loading and a Heston path simulator for synthetic markets.
#pragma once

#include <cmath>
#include <cstdint>
#include <limits>
#include <string>
#include <vector>

#include "ohe/models.hpp"

namespace ohe {

struct Bar {
  std::string date;
  double close = 0;
  double iv = std::numeric_limits<double>::quiet_NaN();  // ATM implied vol if known
};

// Reads a CSV with a header row. Uses the "Date" column and "Adj Close" if
// present, else "Close"; an "IV" column, if present, is read as implied vol.
// Also reads FRED downloads (observation_date + one value column). Rows with
// missing values (FRED writes ".") are skipped; rows are sorted by date.
std::vector<Bar> load_bars(const std::string& path);

// Attaches implied vol from another CSV (e.g. VIX) to matching dates:
// iv = close * scale. Bars without a match are dropped.
std::vector<Bar> join_implied_vol(const std::vector<Bar>& bars, const std::vector<Bar>& iv_bars,
                                  double scale);

void save_bars(const std::string& path, const std::vector<Bar>& bars);

struct SimConfig {
  int days = 1000;
  double spot = 100;
  double rate = 0.02;
  HestonParams heston{0.04, 3.0, 0.04, 0.4, -0.7};
  double iv_premium = 0.10;   // implied = expected vol over the tenor x (1 + premium)
  int iv_tenor_days = 21;
  std::uint64_t seed = 1;
  int steps_per_day = 8;
};

// Daily Heston path with a matching implied-vol series, so strategies can be
// tested on a market whose true volatility premium is known.
std::vector<Bar> simulate_market(const SimConfig& cfg);

}  // namespace ohe
