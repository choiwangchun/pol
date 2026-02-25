#!/usr/bin/env bash
set -euo pipefail

# Tiny-canary defaults. Override with env vars when needed.
MAX_MARKET_NOTIONAL="${MAX_MARKET_NOTIONAL:-10}"
MAX_DAILY_LOSS="${MAX_DAILY_LOSS:-5}"
F_CAP="${F_CAP:-0.01}"
KELLY_FRACTION="${KELLY_FRACTION:-0.1}"
MIN_ORDER_NOTIONAL="${MIN_ORDER_NOTIONAL:-0.3}"

PYTHONPATH=src uv run --python 3.11 python -m pm1h_edge_trader.main \
  --mode live \
  --max-market-notional "${MAX_MARKET_NOTIONAL}" \
  --max-daily-loss "${MAX_DAILY_LOSS}" \
  --f-cap "${F_CAP}" \
  --kelly-fraction "${KELLY_FRACTION}" \
  --max-entries-per-market 1 \
  --min-order-notional "${MIN_ORDER_NOTIONAL}" \
  --disable-auto-claim \
  --hard-kill-on-daily-loss
