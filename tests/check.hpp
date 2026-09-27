// Minimal self-registering test harness (no external dependencies).
#pragma once

#include <cmath>
#include <sstream>
#include <string>
#include <vector>

namespace check {

struct Case {
  const char* name;
  void (*fn)();
};

inline std::vector<Case>& registry() {
  static std::vector<Case> cases;
  return cases;
}

struct Register {
  Register(const char* name, void (*fn)()) { registry().push_back({name, fn}); }
};

struct Failure {
  std::string message;
};

inline std::string where(const char* file, int line) {
  std::ostringstream os;
  os << file << ":" << line;
  return os.str();
}

}  // namespace check

#define TEST(name)                                          \
  static void name();                                       \
  static const check::Register register_##name(#name, name); \
  static void name()

#define CHECK(cond)                                                              \
  do {                                                                           \
    if (!(cond)) throw check::Failure{check::where(__FILE__, __LINE__) + ": " #cond}; \
  } while (0)

#define CHECK_NEAR(actual, expected, tol)                                                       \
  do {                                                                                          \
    const double a_ = (actual), e_ = (expected), t_ = (tol);                                    \
    if (!(std::abs(a_ - e_) <= t_)) {                                                           \
      std::ostringstream os_;                                                                   \
      os_.precision(12);                                                                        \
      os_ << check::where(__FILE__, __LINE__) << ": " #actual " = " << a_ << ", expected " << e_ \
          << " +- " << t_;                                                                      \
      throw check::Failure{os_.str()};                                                          \
    }                                                                                           \
  } while (0)

#define CHECK_THROWS(expr)                                                                      \
  do {                                                                                          \
    bool threw_ = false;                                                                        \
    try {                                                                                       \
      (void)(expr);                                                                             \
    } catch (...) {                                                                             \
      threw_ = true;                                                                            \
    }                                                                                           \
    if (!threw_) throw check::Failure{check::where(__FILE__, __LINE__) + ": expected throw: " #expr}; \
  } while (0)
