// Deterministic random numbers: PCG64 (XSL-RR 128/64), bit-compatible with
// numpy.random.PCG64 for the same (state, increment).
#pragma once

#include <cmath>
#include <cstdint>
#include <string_view>

namespace ohe {

struct U128 {
  std::uint64_t hi = 0;
  std::uint64_t lo = 0;
  friend bool operator==(const U128&, const U128&) = default;
};

// 64 x 64 -> 128-bit product from 32-bit halves; used where the compiler has
// no __int128 (MSVC), and tested against the native path where it does.
inline U128 mul_64x64_portable(std::uint64_t a, std::uint64_t b) {
  const std::uint64_t a_lo = a & 0xffffffffULL, a_hi = a >> 32;
  const std::uint64_t b_lo = b & 0xffffffffULL, b_hi = b >> 32;
  const std::uint64_t p0 = a_lo * b_lo, p1 = a_lo * b_hi, p2 = a_hi * b_lo, p3 = a_hi * b_hi;
  const std::uint64_t mid = (p0 >> 32) + (p1 & 0xffffffffULL) + (p2 & 0xffffffffULL);
  return {p3 + (p1 >> 32) + (p2 >> 32) + (mid >> 32), (p0 & 0xffffffffULL) | (mid << 32)};
}

inline U128 mul_64x64(std::uint64_t a, std::uint64_t b) {
#if defined(__SIZEOF_INT128__)
  __extension__ using u128_native = unsigned __int128;
  const u128_native p = static_cast<u128_native>(a) * b;
  return {static_cast<std::uint64_t>(p >> 64), static_cast<std::uint64_t>(p)};
#else
  return mul_64x64_portable(a, b);
#endif
}

inline U128 add(U128 a, U128 b) {
  U128 r{a.hi + b.hi, a.lo + b.lo};
  r.hi += (r.lo < a.lo);
  return r;
}

inline U128 mul(U128 a, U128 b) {
  U128 r = mul_64x64(a.lo, b.lo);
  r.hi += a.lo * b.hi + a.hi * b.lo;
  return r;
}

class Pcg64 {
 public:
  using result_type = std::uint64_t;
  static constexpr U128 kMultiplier{0x2360ed051fc65da4ULL, 0x4385df649fccf645ULL};

  // Raw state, as numpy's `PCG64.state = {"state": s, "inc": i}` sets it.
  Pcg64(U128 state, U128 increment) : state_(state), inc_{increment.hi, increment.lo | 1ULL} {}

  // Seed + stream id. Different streams never overlap in practice, which is
  // what lets Monte Carlo blocks run on any thread and stay reproducible.
  static Pcg64 from_seed(std::uint64_t seed, std::uint64_t stream = 0);

  result_type next() {
    step();
    return output(state_);
  }
  result_type operator()() { return next(); }
  static constexpr result_type min() { return 0; }
  static constexpr result_type max() { return ~0ULL; }

  // Uniform in [0, 1) with 53 random bits.
  double uniform() { return static_cast<double>(next() >> 11) * 0x1.0p-53; }
  // Standard normal (Marsaglia polar method).
  double normal();
  // Poisson(lambda) by inversion; fine for the small rates used per time step.
  unsigned poisson(double lambda);

  // Jump ahead by `delta` steps in O(log delta) (Brown, "Random number
  // generation with arbitrary strides").
  void advance(U128 delta);
  void advance(std::uint64_t delta) { advance(U128{0, delta}); }

  U128 state() const { return state_; }
  U128 increment() const { return inc_; }

 private:
  void step() { state_ = add(mul(state_, kMultiplier), inc_); }
  static result_type output(U128 s) {
    const std::uint64_t x = s.hi ^ s.lo;
    const unsigned rot = static_cast<unsigned>(s.hi >> 58);
    return (x >> rot) | (x << ((64U - rot) & 63U));
  }

  U128 state_;
  U128 inc_;
  bool has_spare_ = false;
  double spare_ = 0.0;
};

// Turn a user seed into 64 bits: decimal strings parse as integers, anything
// else is hashed (FNV-1a), so "2024-backtest" is a valid, stable seed.
std::uint64_t normalize_seed(std::string_view seed);

}  // namespace ohe
