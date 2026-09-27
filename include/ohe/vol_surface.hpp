// Implied volatility surface over (expiry, strike).
#pragma once

#include <vector>

namespace ohe {

struct SurfaceParams {
  double atm_short = 0.20;  // ATM vol at very short expiries
  double atm_long = 0.20;   // ATM vol the term structure decays towards
  double term_decay = 0.5;  // years; how fast short-dated vol reverts to long-dated
  double skew = -0.10;      // slope in standardized moneyness (negative: puts richer)
  double smile = 0.02;      // curvature in standardized moneyness
  double floor = 0.01;      // minimum vol anywhere on the surface
};

class VolSurface {
 public:
  struct Slice {
    double expiry;                    // years
    std::vector<double> log_strikes;  // ln(K / ref_spot), increasing
    std::vector<double> vols;
  };

  VolSurface(double ref_spot, std::vector<Slice> slices);

  // Parametric surface: vol(k, t) = atm(t) + skew * x + smile * x^2 with
  // standardized moneyness x = k / (atm(t) sqrt(t)) and
  // atm(t) = long + (short - long) exp(-t / decay). Nodes span +-3 sd.
  static VolSurface parametric(double spot, const SurfaceParams& p, const std::vector<double>& expiries,
                               int strikes_per_slice = 25);

  double ref_spot() const { return ref_spot_; }
  const std::vector<Slice>& slices() const { return slices_; }

  // Implied vol at strike K and expiry tau, for the current spot.
  // Within a slice: linear in log-moneyness, flat beyond the wings.
  // Across slices: linear in total variance sigma^2 t, flat outside the range.
  double iv(double K, double tau, double spot) const;
  double iv(double K, double tau) const { return iv(K, tau, ref_spot_); }

  // Grid points where total variance falls between consecutive expiries
  // (calendar arbitrage). A well-formed surface returns 0.
  int calendar_violations(int samples = 41) const;

  // Same shape, shifted so the ATM vol at `tau` equals `atm_vol`. Used when a
  // single implied-vol quote (e.g. an index like VIX) is all that is known.
  VolSurface rescaled_to_atm(double tau, double atm_vol) const;

 private:
  double slice_iv(const Slice& s, double k) const;

  double ref_spot_;
  std::vector<Slice> slices_;
};

}  // namespace ohe
