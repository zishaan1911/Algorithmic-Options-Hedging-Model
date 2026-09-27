#include "ohe/ledger.hpp"

#include <sqlite3.h>

#include <stdexcept>

namespace ohe {

namespace {

void check(int rc, sqlite3* db, const char* what) {
  if (rc != SQLITE_OK && rc != SQLITE_DONE && rc != SQLITE_ROW) {
    throw std::runtime_error(std::string(what) + ": " + (db ? sqlite3_errmsg(db) : "sqlite error"));
  }
}

void exec(sqlite3* db, const char* sql) {
  char* err = nullptr;
  if (sqlite3_exec(db, sql, nullptr, nullptr, &err) != SQLITE_OK) {
    std::string msg = err ? err : "sqlite error";
    sqlite3_free(err);
    throw std::runtime_error(msg);
  }
}

constexpr const char* kSchema = R"sql(
CREATE TABLE IF NOT EXISTS fills (
  id INTEGER PRIMARY KEY, run TEXT NOT NULL, day INTEGER, date TEXT,
  instrument TEXT, quantity REAL, price REAL, fee REAL, slippage REAL);
CREATE TABLE IF NOT EXISTS days (
  run TEXT NOT NULL, day INTEGER, date TEXT, spot REAL, implied_vol REAL, equity REAL,
  option_pnl REAL, hedge_pnl REAL, costs REAL, delta_pnl REAL, gamma_pnl REAL,
  theta_pnl REAL, vega_pnl REAL, net_delta REAL, contracts INTEGER, halted INTEGER,
  PRIMARY KEY (run, day));
CREATE TABLE IF NOT EXISTS events (
  id INTEGER PRIMARY KEY, run TEXT NOT NULL, day INTEGER, date TEXT, kind TEXT, message TEXT);
CREATE INDEX IF NOT EXISTS fills_run ON fills(run);
CREATE INDEX IF NOT EXISTS events_run ON events(run);
)sql";

struct Stmt {
  sqlite3_stmt* s = nullptr;
  Stmt(sqlite3* db, const char* sql) { check(sqlite3_prepare_v2(db, sql, -1, &s, nullptr), db, sql); }
  ~Stmt() { sqlite3_finalize(s); }
  Stmt& text(int i, const std::string& v) {
    sqlite3_bind_text(s, i, v.c_str(), -1, SQLITE_TRANSIENT);
    return *this;
  }
  Stmt& real(int i, double v) {
    sqlite3_bind_double(s, i, v);
    return *this;
  }
  Stmt& integer(int i, long long v) {
    sqlite3_bind_int64(s, i, v);
    return *this;
  }
};

sqlite3* open_readonly(const std::string& path) {
  sqlite3* db = nullptr;
  if (sqlite3_open_v2(path.c_str(), &db, SQLITE_OPEN_READONLY, nullptr) != SQLITE_OK) {
    const std::string msg = db ? sqlite3_errmsg(db) : "cannot open";
    sqlite3_close(db);
    throw std::runtime_error(path + ": " + msg);
  }
  sqlite3_busy_timeout(db, 5000);
  return db;
}

}  // namespace

Ledger::Ledger(const std::string& path) {
  if (sqlite3_open(path.c_str(), &db_) != SQLITE_OK) {
    const std::string msg = db_ ? sqlite3_errmsg(db_) : "cannot open";
    sqlite3_close(db_);
    throw std::runtime_error(path + ": " + msg);
  }
  sqlite3_busy_timeout(db_, 5000);
  exec(db_, "PRAGMA journal_mode=WAL; PRAGMA synchronous=NORMAL;");
  exec(db_, kSchema);
  writer_ = std::thread([this] { writer_loop(); });
}

Ledger::~Ledger() {
  {
    std::lock_guard lock(mu_);
    stop_ = true;
  }
  cv_.notify_all();
  if (writer_.joinable()) writer_.join();
  sqlite3_close(db_);
}

void Ledger::push(Item item) {
  {
    std::lock_guard lock(mu_);
    queue_.push_back(std::move(item));
  }
  cv_.notify_one();
}

void Ledger::flush() {
  std::unique_lock lock(mu_);
  drained_.wait(lock, [this] { return queue_.empty() && !busy_; });
}

void Ledger::writer_loop() {
  std::unique_lock lock(mu_);
  for (;;) {
    cv_.wait(lock, [this] { return stop_ || !queue_.empty(); });
    if (queue_.empty() && stop_) break;
    std::deque<Item> batch;
    batch.swap(queue_);
    busy_ = true;
    lock.unlock();
    exec(db_, "BEGIN");
    for (const Item& item : batch) write(item);
    exec(db_, "COMMIT");
    lock.lock();
    busy_ = false;
    drained_.notify_all();
  }
}

void Ledger::write(const Item& item) {
  if (const auto* f = std::get_if<FillRecord>(&item)) {
    Stmt st(db_, "INSERT INTO fills(run,day,date,instrument,quantity,price,fee,slippage) "
                 "VALUES(?,?,?,?,?,?,?,?)");
    st.text(1, f->run).integer(2, f->day).text(3, f->date).text(4, f->instrument)
        .real(5, f->quantity).real(6, f->price).real(7, f->fee).real(8, f->slippage);
    check(sqlite3_step(st.s), db_, "insert fill");
  } else if (const auto* d = std::get_if<DayRecord>(&item)) {
    Stmt st(db_, "INSERT OR REPLACE INTO days VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)");
    st.text(1, d->run).integer(2, d->day).text(3, d->date).real(4, d->spot).real(5, d->implied_vol)
        .real(6, d->equity).real(7, d->option_pnl).real(8, d->hedge_pnl).real(9, d->costs)
        .real(10, d->delta_pnl).real(11, d->gamma_pnl).real(12, d->theta_pnl).real(13, d->vega_pnl)
        .real(14, d->net_delta).integer(15, d->contracts).integer(16, d->halted ? 1 : 0);
    check(sqlite3_step(st.s), db_, "insert day");
  } else if (const auto* e = std::get_if<EventRecord>(&item)) {
    Stmt st(db_, "INSERT INTO events(run,day,date,kind,message) VALUES(?,?,?,?,?)");
    st.text(1, e->run).integer(2, e->day).text(3, e->date).text(4, e->kind).text(5, e->message);
    check(sqlite3_step(st.s), db_, "insert event");
  }
}

std::vector<FillRecord> Ledger::read_fills(const std::string& path, const std::string& run) {
  sqlite3* db = open_readonly(path);
  std::vector<FillRecord> out;
  {
    Stmt st(db, "SELECT day,date,instrument,quantity,price,fee,slippage FROM fills WHERE run=? ORDER BY id");
    st.text(1, run);
    while (sqlite3_step(st.s) == SQLITE_ROW) {
      FillRecord f;
      f.run = run;
      f.day = sqlite3_column_int(st.s, 0);
      f.date = reinterpret_cast<const char*>(sqlite3_column_text(st.s, 1));
      f.instrument = reinterpret_cast<const char*>(sqlite3_column_text(st.s, 2));
      f.quantity = sqlite3_column_double(st.s, 3);
      f.price = sqlite3_column_double(st.s, 4);
      f.fee = sqlite3_column_double(st.s, 5);
      f.slippage = sqlite3_column_double(st.s, 6);
      out.push_back(std::move(f));
    }
  }
  sqlite3_close(db);
  return out;
}

std::vector<EventRecord> Ledger::read_events(const std::string& path, const std::string& run) {
  sqlite3* db = open_readonly(path);
  std::vector<EventRecord> out;
  {
    Stmt st(db, "SELECT day,date,kind,message FROM events WHERE run=? ORDER BY id");
    st.text(1, run);
    while (sqlite3_step(st.s) == SQLITE_ROW) {
      EventRecord e;
      e.run = run;
      e.day = sqlite3_column_int(st.s, 0);
      e.date = reinterpret_cast<const char*>(sqlite3_column_text(st.s, 1));
      e.kind = reinterpret_cast<const char*>(sqlite3_column_text(st.s, 2));
      e.message = reinterpret_cast<const char*>(sqlite3_column_text(st.s, 3));
      out.push_back(std::move(e));
    }
  }
  sqlite3_close(db);
  return out;
}

int Ledger::count_days(const std::string& path, const std::string& run) {
  sqlite3* db = open_readonly(path);
  int n = 0;
  {
    Stmt st(db, "SELECT COUNT(*) FROM days WHERE run=?");
    st.text(1, run);
    if (sqlite3_step(st.s) == SQLITE_ROW) n = sqlite3_column_int(st.s, 0);
  }
  sqlite3_close(db);
  return n;
}

}  // namespace ohe
