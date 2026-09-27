cmake_minimum_required(VERSION 3.20)
project(options_hedging_engine VERSION 1.0.0 LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release)
endif()

option(OHE_BUILD_TESTS "Build the test suite" ON)

# SQLite: use the system library, else build the pinned amalgamation.
find_package(SQLite3 QUIET)
if(NOT SQLite3_FOUND)
  include(FetchContent)
  FetchContent_Declare(sqlite_amalgamation
    URL https://www.sqlite.org/2026/sqlite-amalgamation-3530400.zip
    URL_HASH SHA3_256=628a44cfe82c66aed1ccbbe85a562d2e33ebe64b3288981ed76285612227934e
    DOWNLOAD_EXTRACT_TIMESTAMP TRUE)
  FetchContent_MakeAvailable(sqlite_amalgamation)
  add_library(sqlite3_bundled STATIC ${sqlite_amalgamation_SOURCE_DIR}/sqlite3.c)
  target_include_directories(sqlite3_bundled PUBLIC ${sqlite_amalgamation_SOURCE_DIR})
  target_compile_definitions(sqlite3_bundled PRIVATE SQLITE_THREADSAFE=1 SQLITE_OMIT_LOAD_EXTENSION)
  find_package(Threads REQUIRED)
  target_link_libraries(sqlite3_bundled PUBLIC Threads::Threads)
  add_library(SQLite::SQLite3 ALIAS sqlite3_bundled)
endif()

find_package(Threads REQUIRED)

add_library(ohe
  src/rng.cpp
  src/black_scholes.cpp
  src/models.cpp
  src/features.cpp
  src/vol_surface.cpp
  src/risk.cpp
  src/data.cpp
  src/ledger.cpp
  src/backtest.cpp)
target_include_directories(ohe PUBLIC include)
target_link_libraries(ohe PUBLIC Threads::Threads PRIVATE SQLite::SQLite3)
if(MSVC)
  target_compile_options(ohe PRIVATE /W4 /permissive-)
else()
  target_compile_options(ohe PRIVATE -Wall -Wextra -Wpedantic -Wshadow -Wconversion -Wno-sign-conversion)
endif()

add_executable(ohe_cli apps/ohe.cpp)
set_target_properties(ohe_cli PROPERTIES OUTPUT_NAME ohe)
target_link_libraries(ohe_cli PRIVATE ohe)
if(MINGW)
  # Self-contained Windows binaries: no libstdc++/libgcc/winpthread DLLs needed.
  target_link_options(ohe_cli PRIVATE -static)
endif()

if(OHE_BUILD_TESTS)
  enable_testing()
  add_executable(ohe_tests
    tests/main.cpp
    tests/test_rng.cpp
    tests/test_pricing.cpp
    tests/test_surface_features.cpp
    tests/test_risk.cpp
    tests/test_backtest.cpp)
  target_link_libraries(ohe_tests PRIVATE ohe)
  if(MINGW)
    target_link_options(ohe_tests PRIVATE -static)
  endif()
  add_test(NAME ohe_tests COMMAND ohe_tests)
endif()
