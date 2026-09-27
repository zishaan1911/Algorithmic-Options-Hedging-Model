// Trade ledger in SQLite (WAL mode).
//
// Writes go through a queue to a single writer thread that commits in
// batches, so the backtest loop never blocks on disk. WAL mode lets other
// connections (e.g. `ohe ledger` in another terminal) read while a run writes.
#pragma once

#include <condition_variable>
#include <deque>
#include <mutex>
#include <string>
#include <thread>
#include <variant>
#include <vector>

struct sqlite3;

namespace ohe {

struct FillRecord {
  std::string run;
  int day = 0;
  std::string date;
  std::string instrument;  // e.g. "UNDERLYING", "C 450.00 2024-05-17"
  double quantity = 0;     // signed: + buy, - sell
  double price = 0;
  double fee = 0;
  double slippage = 0;     // cost versus mid, in currency
};

struct DayRecord {
  std::string run;
  int day = 0;
  std::string date;
  double spot = 0;
  double implied_vol = 0;
  double equity = 0;
  double option_pnl = 0;
  double hedge_pnl = 0;
  double costs = 0;
  double delta_pnl = 0, gamma_pnl = 0, theta_pnl = 0, vega_pnl = 0;
  double net_delta = 0;
  int contracts = 0;
  bool halted = false;
};

struct EventRecord {
  std::string run;
  int day = 0;
  std::string date;
  std::string kind;
  std::string message;
};

class Ledger {
 public:
  explicit Ledger(const std::string& path);
  ~Ledger();
  Ledger(const Ledger&) = delete;
  Ledger& operator=(const Ledger&) = delete;

  void record(FillRecord r) { push(std::move(r)); }
  void record(DayRecord r) { push(std::move(r)); }
  void record(EventRecord r) { push(std::move(r)); }
  // Blocks until everything queued so far is committed.
  void flush();

  // Read-only helpers using their own connection (safe while a writer runs).
  static std::vector<FillRecord> read_fills(const std::string& path, const std::string& run);
  static std::vector<EventRecord> read_events(const std::string& path, const std::string& run);
  static int count_days(const std::string& path, const std::string& run);

 private:
  using Item = std::variant<FillRecord, DayRecord, EventRecord>;
  void push(Item item);
  void writer_loop();
  void write(const Item& item);

  sqlite3* db_ = nullptr;
  std::mutex mu_;
  std::condition_variable cv_;
  std::condition_variable drained_;
  std::deque<Item> queue_;
  bool stop_ = false;
  bool busy_ = false;
  std::thread writer_;
};

}  // namespace ohe
