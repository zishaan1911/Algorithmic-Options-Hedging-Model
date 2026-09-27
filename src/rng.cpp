#include "ohe/rng.hpp"

#include <charconv>

namespace ohe {

namespace {

std::uint64_t splitmix64(std::uint64_t& x) {
  std::uint64_t z = (x += 0x9e3779b97f4a7c15ULL);
  z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
  z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
  return z ^ (z >> 31);
}

}  // namespace

Pcg64 Pcg64::from_seed(std::uint64_t seed, std::uint64_t stream) {
  // PCG's reference seeding (pcg_setseq_128_srandom_r) with 128-bit words
  // expanded from the 64-bit inputs by SplitMix64.
  std::uint64_t s = seed;
  const U128 init_state{splitmix64(s), splitmix64(s)};
  std::uint64_t t = stream ^ 0xda3e39cb94b95bdbULL;
  U128 init_seq{splitmix64(t), splitmix64(t)};
  init_seq = U128{(init_seq.hi << 1) | (init_seq.lo >> 63), (init_seq.lo << 1) | 1ULL};

  Pcg64 rng(U128{0, 0}, init_seq);
  rng.step();
  rng.state_ = add(rng.state_, init_state);
  rng.step();
  return rng;
}

double Pcg64::normal() {
  if (has_spare_) {
    has_spare_ = false;
    return spare_;
  }
  double u, v, s;
  do {
    u = 2.0 * uniform() - 1.0;
    v = 2.0 * uniform() - 1.0;
    s = u * u + v * v;
  } while (s >= 1.0 || s == 0.0);
  const double m = std::sqrt(-2.0 * std::log(s) / s);
  spare_ = v * m;
  has_spare_ = true;
  return u * m;
}

unsigned Pcg64::poisson(double lambda) {
  if (lambda <= 0.0) return 0;
  const double limit = std::exp(-lambda);
  double p = uniform();
  unsigned k = 0;
  while (p > limit) {
    p *= uniform();
    ++k;
  }
  return k;
}

void Pcg64::advance(U128 delta) {
  U128 acc_mult{0, 1}, acc_plus{0, 0};
  U128 cur_mult = kMultiplier, cur_plus = inc_;
  while (delta.hi != 0 || delta.lo != 0) {
    if (delta.lo & 1ULL) {
      acc_mult = mul(acc_mult, cur_mult);
      acc_plus = add(mul(acc_plus, cur_mult), cur_plus);
    }
    cur_plus = mul(add(cur_mult, U128{0, 1}), cur_plus);
    cur_mult = mul(cur_mult, cur_mult);
    delta = U128{delta.hi >> 1, (delta.lo >> 1) | (delta.hi << 63)};
  }
  state_ = add(mul(acc_mult, state_), acc_plus);
}

std::uint64_t normalize_seed(std::string_view seed) {
  std::uint64_t value = 0;
  const auto [ptr, ec] = std::from_chars(seed.data(), seed.data() + seed.size(), value);
  if (ec == std::errc{} && ptr == seed.data() + seed.size() && !seed.empty()) return value;
  std::uint64_t h = 0xcbf29ce484222325ULL;  // FNV-1a
  for (const unsigned char c : seed) {
    h ^= c;
    h *= 0x100000001b3ULL;
  }
  return h;
}

}  // namespace ohe
