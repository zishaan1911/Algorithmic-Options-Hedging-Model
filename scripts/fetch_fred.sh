#!/usr/bin/env sh
# Download the S&P 500 index and VIX (30-day implied vol of S&P 500 options)
# from FRED, the St. Louis Fed's public data service, into ./data.
#
#   sh scripts/fetch_fred.sh
#   ./build/ohe backtest --prices data/SP500.csv --iv data/VIXCLS.csv --iv-scale 0.01
#
# The files are not committed: S&P Dow Jones Indices licenses the SP500 series
# for personal use and does not allow redistribution.
set -eu
mkdir -p data
for series in SP500 VIXCLS; do
  curl -fsSL "https://fred.stlouisfed.org/graph/fredgraph.csv?id=${series}" -o "data/${series}.csv"
  echo "data/${series}.csv: $(($(wc -l < "data/${series}.csv") - 1)) rows"
done
