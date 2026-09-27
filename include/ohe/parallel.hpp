// Deterministic parallel Monte Carlo.
//
// Work is cut into fixed-size blocks and block b always draws from
// Pcg64::from_seed(seed, b). Workers pull blocks from a shared counter and the
// partial results are combined in block order, so the answer is identical for
// 1 thread or 64.
#pragma once

#include <algorithm>
#include <atomic>
#include <cstddef>
#include <functional>
#include <thread>
#include <vector>

namespace ohe {

inline unsigned default_threads() {
  const unsigned n = std::thread::hardware_concurrency();
  return n == 0 ? 1 : n;
}

// Calls fn(block_index) for every block in [0, blocks) on up to `threads` threads.
inline void parallel_blocks(std::size_t blocks, unsigned threads,
                            const std::function<void(std::size_t)>& fn) {
  threads = std::max(1U, std::min<unsigned>(threads, static_cast<unsigned>(blocks)));
  if (threads == 1) {
    for (std::size_t b = 0; b < blocks; ++b) fn(b);
    return;
  }
  std::atomic<std::size_t> next{0};
  std::vector<std::thread> pool;
  pool.reserve(threads);
  for (unsigned t = 0; t < threads; ++t) {
    pool.emplace_back([&] {
      for (std::size_t b = next.fetch_add(1); b < blocks; b = next.fetch_add(1)) fn(b);
    });
  }
  for (std::thread& th : pool) th.join();
}

}  // namespace ohe
