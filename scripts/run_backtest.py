#!/bin/bash

# Run backtesting for all strategies
# Usage: ./run_backtest.sh [strategy] [symbol] [timeframe]

set -e

# Default parameters
STRATEGY=${1:-"all"}
SYMBOL=${2:-"BTC-USD"}
TIMEFRAME=${3:-"1h"}
START_DATE=${4:-"2023-01-01"}
END_DATE=${5:-"2024-12-31"}

echo "================================================"
echo "Running Backtest"
echo "================================================"
echo "Strategy: $STRATEGY"
echo "Symbol: $SYMBOL"
echo "Timeframe: $TIMEFRAME"
echo "Period: $START_DATE to $END_DATE"
echo "================================================"

# Activate virtual environment
source venv/bin/activate

# Run backtest
python -m src.backtesting.backtest_runner \
    --strategy $STRATEGY \
    --symbol $SYMBOL \
    --timeframe $TIMEFRAME \
    --start-date $START_DATE \
    --end-date $END_DATE \
    --initial-capital 100000 \
    --output-dir backtest_results

# Generate report
python -m src.backtesting.report_generator \
    --results-dir backtest_results \
    --output-format html

echo "================================================"
echo "Backtest completed!"
echo "Results saved to: backtest_results/"
echo "================================================"