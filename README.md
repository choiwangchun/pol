# PM-1H Edge Trader (MVP)

Polymarket BTC 1H Up/Down market에서 Binance 기반 공정확률 `q`를 계산하고,
`ask/bid` 대비 엣지를 기준으로 dry-run 주문 의사결정을 수행하는 MVP입니다.

## 핵심 구현 범위

- Gamma API 기반 활성 BTC 1H 시장 자동 탐색 (`rules/resolution` 검증 포함)
- CLOB 오더북(top-of-book) + Binance 1H `open(K)`/실시간 `S_t` 피드
- 확률 엔진: `q_up = P(S_T >= K)` (로그정규 근사, RV/IV 혼합)
- 엣지 필터 + Fractional Kelly 사이징
- limit intent 실행 엔진(취소/재호가/near-expiry block/kill-switch)
- CSV 실행 로그(`logs/executions.csv`)

## 실행

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
pm1h-edge-trader --help
```

한 사이클 dry-run:

```bash
PYTHONPATH=src python -m pm1h_edge_trader.main \
  --mode dry-run \
  --max-ticks 1 \
  --disable-websocket
```

특정 시장을 강제 지정하려면:

```bash
PYTHONPATH=src python -m pm1h_edge_trader.main \
  --mode dry-run \
  --market-slug bitcoin-up-or-down-february-16-1am-et \
  --max-ticks 1
```

live 모드(실주문):

```bash
export POLYMARKET_PRIVATE_KEY=...
export POLYMARKET_FUNDER=...
PYTHONPATH=src python -m pm1h_edge_trader.main \
  --mode live \
  --max-ticks 1
```

기존 `executions.csv`/`result.json` 누적 상태를 무시하고 새로 시작하려면:

```bash
PYTHONPATH=src python -m pm1h_edge_trader.main \
  --mode dry-run \
  --fresh-start
```

## Paper 성과 집계 (자동 체결/정산)

`dry-run`에서는 실행 로그에 아래 status가 추가됩니다.

- `paper_fill`: limit가 현재 best ask를 충족해 paper 체결로 간주된 주문
- `paper_settle`: Polymarket 결과가 확정되어 PnL/승패가 기록된 체결
- `result.json`: 가상 금고 상태(현재 잔고/실현손익/승률/미정산 체결)를 덮어쓰기 갱신

예: `--bankroll 1000`으로 실행하면 `logs/result.json` 기준으로 아래 값이 누적됩니다.

- `initial_bankroll`
- `current_balance`
- `realized_pnl`
- `settled_trades`, `wins`, `losses`, `win_rate`
- `unsettled_fills`, `unsettled_notional`
- `last_event` (`paper_fill` 또는 `paper_settle`)

승률/손익 집계 예시:

```bash
PYTHONPATH=src python - <<'PY'
from pm1h_edge_trader.logger import summarize_csv

initial_bankroll = 500.0
summary = summarize_csv("logs/executions.csv")
roi = (summary.total_pnl / initial_bankroll * 100.0) if initial_bankroll > 0 else 0.0
print(f"trades={summary.trade_count} wins={summary.win_count} losses={summary.loss_count}")
print(f"hit_rate={summary.hit_rate*100:.2f}% total_pnl=${summary.total_pnl:.2f} roi={roi:.2f}%")
PY
```

## 테스트

```bash
python3 -m unittest -q src/pm1h_edge_trader/engine/tests/test_math_engine.py
python3 -m unittest discover -s tests -p "test_*.py" -v
```

## 참고

- `live` 모드는 `py-clob-client` 기반으로 연결되어 있으며, 실계정/실주문이 발생할 수 있으므로
  운영 키/자금/리스크 한도 관리 후 사용해야 합니다.
- 활성 BTC 1H Up/Down 시장이 없는 시점에는 discovery 단계에서 종료됩니다.
