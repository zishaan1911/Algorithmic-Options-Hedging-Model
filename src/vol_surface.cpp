#include "ohe/vol_surface.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace ohe {

VolSurface::VolSurface(double ref_spot, std::vector<Slice> slices)
    : ref_spot_(ref_spot), slices_(std::move(slices)) {
  if (ref_spot_ <= 0) throw std::invalid_argument("reference spot must be positive");
  if (slices_.empty()) throw std::invalid_argument("surface needs at least one expiry");
  std::sort(slices_.begin(), slices_.end(), [](const Slice& a, const Slice& b) { return a.expiry < b.expiry; });
  for (const Slice& s : slices_) {
    if (s.expiry <= 0 || s.log_strikes.empty() || s.log_strikes.size() != s.vols.size()) {
      throw std::invalid_argument("malformed surface slice");
    }
    if (!std::is_sorted(s.log_strikes.begin(), s.log_strikes.end())) {
      throw std::invalid_argument("slice strikes must be increasing");
    }
  }
}

VolSurface VolSurface::parametric(double spot, const SurfaceParams& p, const std::vector<double>& expiries,
                                  int strikes_per_slice) {
  std::vector<Slice> slices;
  for (double t : expiries) {
    const double atm = p.atm_long + (p.atm_short - p.atm_long) * std::exp(-t / p.term_decay);
    const double sd = atm * std::sqrt(t);
    Slice s{t, {}, {}};
    for (int i = 0; i < strikes_per_slice; ++i) {
      const double x = -3.0 + 6.0 * i / (strikes_per_slice - 1);  // standardized moneyness
      s.log_strikes.push_back(x * sd);
      s.vols.push_back(std::max(p.floor, atm + p.skew * x + p.smile * x * x));
    }
    slices.push_back(std::move(s));
  }
  return VolSurface(spot, std::move(slices));
}

double VolSurface::slice_iv(const Slice& s, double k) const {
  const auto& ks = s.log_strikes;
  if (k <= ks.front()) return s.vols.front();
  if (k >= ks.back()) return s.vols.back();
  const auto it = std::upper_bound(ks.begin(), ks.end(), k);
  const std::size_t j = static_cast<std::size_t>(it - ks.begin());
  const double w = (k - ks[j - 1]) / (ks[j] - ks[j - 1]);
  return s.vols[j - 1] + w * (s.vols[j] - s.vols[j - 1]);
}

double VolSurface::iv(double K, double tau, double spot) const {
  const double k = std::log(K / spot);
  if (tau <= slices_.front().expiry) return slice_iv(slices_.front(), k);
  if (tau >= slices_.back().expiry) return slice_iv(slices_.back(), k);
  const auto it = std::upper_bound(slices_.begin(), slices_.end(), tau,
                                   [](double t, const Slice& s) { return t < s.expiry; });
  const Slice& b = *it;
  const Slice& a = *(it - 1);
  const double wa = std::pow(slice_iv(a, k), 2) * a.expiry;
  const double wb = std::pow(slice_iv(b, k), 2) * b.expiry;
  const double w = wa + (wb - wa) * (tau - a.expiry) / (b.expiry - a.expiry);
  return std::sqrt(std::max(w, 0.0) / tau);
}

int VolSurface::calendar_violations(int samples) const {
  int bad = 0;
  const double k_lo = slices_.back().log_strikes.front();
  const double k_hi = slices_.back().log_strikes.back();
  for (int i = 0; i < samples; ++i) {
    const double k = k_lo + (k_hi - k_lo) * i / (samples - 1);
    double prev = 0.0;
    for (const Slice& s : slices_) {
      const double w = std::pow(slice_iv(s, k), 2) * s.expiry;
      if (w + 1e-12 < prev) ++bad;
      prev = w;
    }
  }
  return bad;
}

VolSurface VolSurface::rescaled_to_atm(double tau, double atm_vol) const {
  // A parallel shift of every node; the ATM vol at `tau` is monotone in the
  // shift, so bisect for the one that hits the target exactly (interpolation
  // across expiries is in total variance, so the shift is not simply the gap).
  auto shifted = [&](double shift) {
    std::vector<Slice> out = slices_;
    for (Slice& s : out) {
      for (double& v : s.vols) v = std::max(0.01, v + shift);
    }
    return VolSurface(ref_spot_, std::move(out));
  };
  double lo = -2.0, hi = 2.0;
  for (int i = 0; i < 80; ++i) {
    const double mid = 0.5 * (lo + hi);
    if (shifted(mid).iv(ref_spot_, tau) < atm_vol) lo = mid;
    else hi = mid;
  }
  return shifted(0.5 * (lo + hi));
}

}  // namespace ohe
