#include <set>

#include "check.hpp"
#include "ohe/models.hpp"
#include "ohe/rng.hpp"

using namespace ohe;

// Reference values from numpy 2.x:
//   bg = np.random.PCG64(); bg.state = {"bit_generator": "PCG64", "state": {
//       "state": 0x0123456789abcdeffedcba9876543210,
//       "inc": 0x5851f42d4c957f2d14057b7ef767814f}, "has_uint32": 0, "uinteger": 0}
//   bg.random_raw(5); then reset, bg.advance(1000); bg.random_raw()
TEST(pcg64_matches_numpy_bit_for_bit) {
  const U128 state{0x0123456789abcdefULL, 0xfedcba9876543210ULL};
  const U128 inc{0x5851f42d4c957f2dULL, 0x14057b7ef767814fULL};
  Pcg64 rng(state, inc);
  const std::uint64_t expected[] = {0x13c49fecdee35f71ULL, 0x4ee9574cc31f57d2ULL, 0x718b9867b2c7ef05ULL,
                                    0xa9b3898995846d5cULL, 0x48d690c435a20381ULL};
  for (std::uint64_t e : expected) CHECK(rng.next() == e);
}

TEST(portable_128_bit_multiply_matches_native) {
  Pcg64 rng = Pcg64::from_seed(1);
  const std::uint64_t edge[] = {0, 1, 0xffffffffULL, 0x100000000ULL, ~0ULL, 0x8000000000000000ULL};
  for (std::uint64_t a : edge) {
    for (std::uint64_t b : edge) CHECK(mul_64x64_portable(a, b) == mul_64x64(a, b));
  }
  for (int i = 0; i < 1'000'000; ++i) {
    const std::uint64_t a = rng.next(), b = rng.next();
    CHECK(mul_64x64_portable(a, b) == mul_64x64(a, b));
  }
}

TEST(pcg64_advance_matches_numpy) {
  Pcg64 rng(U128{0x0123456789abcdefULL, 0xfedcba9876543210ULL},
            U128{0x5851f42d4c957f2dULL, 0x14057b7ef767814fULL});
  rng.advance(1000);
  CHECK(rng.next() == 0xa32e8e379313e335ULL);
}

TEST(pcg64_advance_equals_stepping) {
  Pcg64 a = Pcg64::from_seed(99), b = Pcg64::from_seed(99);
  for (int i = 0; i < 12345; ++i) a.next();
  b.advance(12345);
  CHECK(a.next() == b.next());
}

TEST(seeded_streams_are_reproducible_and_distinct) {
  Pcg64 a = Pcg64::from_seed(7, 0), b = Pcg64::from_seed(7, 0), c = Pcg64::from_seed(7, 1);
  std::set<std::uint64_t> firsts;
  for (int i = 0; i < 100; ++i) {
    const auto x = a.next();
    CHECK(x == b.next());
    firsts.insert(x);
    CHECK(x != c.next());
  }
  CHECK(firsts.size() == 100);
}

TEST(normal_sampler_moments) {
  Pcg64 rng = Pcg64::from_seed(2024);
  const int n = 1'000'000;
  double s1 = 0, s2 = 0, s4 = 0;
  for (int i = 0; i < n; ++i) {
    const double z = rng.normal();
    s1 += z;
    s2 += z * z;
    s4 += z * z * z * z;
  }
  CHECK_NEAR(s1 / n, 0.0, 0.005);
  CHECK_NEAR(s2 / n, 1.0, 0.005);
  CHECK_NEAR(s4 / n, 3.0, 0.03);
}

TEST(poisson_sampler_mean) {
  Pcg64 rng = Pcg64::from_seed(5);
  double sum = 0;
  for (int i = 0; i < 200'000; ++i) sum += rng.poisson(0.7);
  CHECK_NEAR(sum / 200'000, 0.7, 0.01);
}

TEST(seed_normalization) {
  CHECK(normalize_seed("42") == 42);
  CHECK(normalize_seed("2024-q1") == normalize_seed("2024-q1"));
  CHECK(normalize_seed("2024-q1") != normalize_seed("2024-q2"));
  CHECK(normalize_seed("") != 0);
}

TEST(monte_carlo_is_identical_for_any_thread_count) {
  Model m;
  m.heston = HestonParams{0.04, 2.0, 0.04, 0.5, -0.6};
  McConfig mc;
  mc.paths = 40'000;
  mc.steps_per_year = 100;
  mc.threads = 1;
  const auto one = mc_price(OptionType::Call, 100, 105, 0.02, 0.0, 0.5, m, mc);
  mc.threads = 8;
  const auto eight = mc_price(OptionType::Call, 100, 105, 0.02, 0.0, 0.5, m, mc);
  CHECK(one.price == eight.price);
  CHECK(one.std_error == eight.std_error);
}
