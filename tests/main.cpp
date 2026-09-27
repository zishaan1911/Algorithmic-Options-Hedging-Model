#include <chrono>
#include <cstdio>
#include <cstring>
#include <exception>

#include "check.hpp"

// Usage: ohe_tests [substring]  -- runs the tests whose name contains substring.
int main(int argc, char** argv) {
  const char* filter = argc > 1 ? argv[1] : nullptr;
  int passed = 0, failed = 0;
  for (const auto& c : check::registry()) {
    if (filter && !std::strstr(c.name, filter)) continue;
    const auto t0 = std::chrono::steady_clock::now();
    try {
      c.fn();
      const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
      std::printf("  pass  %-48s %8.1f ms\n", c.name, ms);
      ++passed;
    } catch (const check::Failure& f) {
      std::printf("  FAIL  %s\n        %s\n", c.name, f.message.c_str());
      ++failed;
    } catch (const std::exception& e) {
      std::printf("  FAIL  %s\n        exception: %s\n", c.name, e.what());
      ++failed;
    }
  }
  std::printf("\n%d passed, %d failed\n", passed, failed);
  return failed == 0 ? 0 : 1;
}
