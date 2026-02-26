#!/usr/bin/env bash
set -euo pipefail

PYTHONPATH=src uv run --python 3.11 python -m pm1h_edge_trader.main \
  --mode live \
  --resume \
  --state-dir state \
  --edge-min 1.0 \
  --max-market-notional 1 \
  --max-daily-loss 1 \
  --kelly-fraction 0.0 \
  --f-cap 0.0 \
  --disable-auto-claim \
  --hard-kill-on-daily-loss
