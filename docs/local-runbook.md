# Local Runbook (MVP Validation)

## 1) Prepare local config
```bash
cp .env.example .env
set -a
source .env
set +a
```

## 2) Run integration validation tests
```bash
python3 -m unittest discover -s tests -p "test_*.py" -v
```

## 3) Run one full dry-run cycle (network data)
This runs Gamma + CLOB + Binance feeds and dry-run execution.

```bash
PYTHONPATH=src python3 -m pm1h_edge_trader.main --mode dry-run --max-ticks 1 --disable-websocket
```

과거 실행 로그를 무시하고 완전 새로 시작하려면 `--fresh-start`를 추가하세요.

```bash
PYTHONPATH=src python3 -m pm1h_edge_trader.main --mode dry-run --fresh-start
```

If no active BTC 1H Up/Down market exists at runtime, the process exits at discovery stage.
특정 시장 URL이 있으면 slug를 지정해서 실행할 수 있습니다:

```bash
PYTHONPATH=src python3 -m pm1h_edge_trader.main \
  --mode dry-run \
  --market-slug bitcoin-up-or-down-february-16-1am-et \
  --max-ticks 1
```

## 4) Trigger execution kill-switch scenarios manually (offline)
```bash
# stale data
python3 -m tests.stubs.run_stub_cycle --edge 0.03 --fee 0 --data-age-s 120 --allow-entry true

# non-zero fee
python3 -m tests.stubs.run_stub_cycle --edge 0.03 --fee 0.001 --data-age-s 3 --allow-entry true
```

## 5) Inspect logs
Execution CSV:

```bash
tail -n 20 logs/executions.csv
rg "place|paper_fill|paper_settle|skip|kill_switch_triggered" logs/executions.csv
cat logs/result.json
```

## 6) 24시간 paper 성과 측정 (예: bankroll=$500)
```bash
RUN_ID=$(date -u +%Y%m%dT%H%M%SZ)
LOG_DIR="logs/paper500_${RUN_ID}"
END_TS=$(( $(date +%s) + 24*3600 ))

while [ $(date +%s) -lt $END_TS ]; do
  PYTHONPATH=src python3 -m pm1h_edge_trader.main \
    --mode dry-run \
    --bankroll 500 \
    --tick-seconds 1 \
    --log-dir "$LOG_DIR" \
  || true
  sleep 3
done

echo "$LOG_DIR/executions.csv"
```

정산 반영 후 승률/손익:

```bash
CSV_PATH="$LOG_DIR/executions.csv" PYTHONPATH=src python3 - <<'PY'
import os
from pm1h_edge_trader.logger import summarize_csv

initial_bankroll = 500.0
summary = summarize_csv(os.environ["CSV_PATH"])
roi = (summary.total_pnl / initial_bankroll * 100.0) if initial_bankroll > 0 else 0.0
print(f"trades={summary.trade_count} wins={summary.win_count} losses={summary.loss_count}")
print(f"hit_rate={summary.hit_rate*100:.2f}% total_pnl=${summary.total_pnl:.2f} roi={roi:.2f}%")
PY
```
