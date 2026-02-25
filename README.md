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

`--mode live`에서는 CLOB `balance/allowance`를 조회해 bankroll이 자동으로 계좌 기준으로 설정됩니다.

Live 모드에서는 자동 Claim(온체인 `redeemPositions`) 루프가 기본 활성화됩니다.

- 기본 주기: 60초
- 기본 쿨다운: 동일 condition 재시도 600초
- 대상: Data API에서 `redeemable=true` 인 포지션
- 기본 RPC: `https://polygon-rpc.com` (`POLYGON_RPC_URL` 또는 `--polygon-rpc-url`로 변경 가능)

중요: 현재 자동 Claim은 **직접 지갑 모드(= `funder` 주소와 `private key` 주소가 동일한 경우)** 에서만 동작합니다.  
프록시/세이프 지갑(`funder != signer`)은 자동 Claim이 비활성화되고 경고 로그만 출력됩니다.

자동 Claim 옵션:

```bash
--disable-auto-claim
--auto-claim-interval-seconds 60
--auto-claim-size-threshold 0.0001
--auto-claim-cooldown-seconds 600
--auto-claim-tx-timeout-seconds 120
--polygon-rpc-url https://polygon-rpc.com
--data-api-base-url https://data-api.polymarket.com
```

Safety + Recon + Arb 옵션(기본은 안전/옵트인):

```bash
--hard-kill-on-daily-loss         # 기본 ON (비활성화: --no-hard-kill-on-daily-loss)
--position-reconcile-interval-seconds 10
--position-mismatch-policy kill   # kill | block | warn
--position-size-threshold 0.0001

--enable-complete-set-arb
--arb-min-profit 0.0
--arb-max-notional 25
--arb-fill-timeout-seconds 15
```

- `kill-switch reason` 확장: `daily_loss_limit`, `bankroll_depleted`, `market_notional_limit_breach`, `position_mismatch`
- 포지션 리컨실은 Live 지갑 포지션(Data API `/positions`)과 로컬 노출을 비교합니다.
- complete-set arb는 기본 OFF이며 `ask_up + ask_down < 1`일 때만 페어 매수를 시도합니다.

Live 오더 리컨실 정책(기본값 안전 모드):

- 기본: venue에 orphan open order가 보이면 **자동 취소하지 않고** kill-switch latch + 수동 확인
- `--cancel-orphan-orders`: orphan order 자동 취소 허용

Live 체결 집계:

- `live_fill` 이벤트를 `result.json`에 반영하여 `unsettled_notional`/`available_bankroll` 계산에 사용
- open 상태 주문도 `size_matched` 증가분을 주기적으로 폴링하여 partial fill을 누적 반영
- `get_orders`/`get_order` 실패는 재시도 후 fail-safe kill-switch를 트리거
- complete-set 원레그 타임아웃 시 우선 자동 복구(반대편 보강 또는 언와인드 시도) 후 실패 시 kill-switch
- 현재 전략은 UP/DOWN 토큰을 기본적으로 `BUY`로 진입하며, 주문 객체에는 venue side(`BUY/SELL`)를 명시적으로 분리해 전달

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

## Electron GUI

데스크톱 UI에서 아래를 한 번에 제어할 수 있습니다.

- 매매 시작/중지
- 실시간 로그 확인
- 현재 잔고/실현손익/승률/미정산 노출 확인
- 수동 1회 포지션 진입(UP/DOWN)
- Live 모드 선택 시 bankroll 자동 조회/반영(계좌 balance/allowance 기반)

실행:

```bash
cd desktop
npm install
npm run start
```

수동 포지션 진입 CLI만 단독으로 쓰고 싶으면:

```bash
uv run --python 3.11 python -m pm1h_edge_trader.manual_entry \
  --mode dry-run \
  --direction up \
  --usd 50
```

API 크리덴셜 자동 생성/반영:

```bash
# POLYMARKET_PRIVATE_KEY / POLYMARKET_FUNDER 가 환경변수에 설정되어 있어야 함
uv run --python 3.11 pm1h-derive-api-creds --write-env
```

## 참고

- `live` 모드는 `py-clob-client` 기반으로 연결되어 있으며, 실계정/실주문이 발생할 수 있으므로
  운영 키/자금/리스크 한도 관리 후 사용해야 합니다.
- 활성 BTC 1H Up/Down 시장이 없는 시점에는 discovery 단계에서 종료됩니다.
