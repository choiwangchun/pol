# Canary Live Runbook

## Pre-flight
1. `.env`에 `POLYMARKET_PRIVATE_KEY`, `POLYMARKET_FUNDER` 설정
2. 전용 지갑 확인(수동 주문 혼용 금지)
3. 카나리 자금만 계정에 유지
4. 로그 경로 확인: `logs/`
5. 잔액 확인:

```bash
uv run --python 3.11 pm1h-live-balance
```

## Stage 0 (No-Trade)

```bash
scripts/canary_stage0_no_trade.sh
```

검증:
1. 주문이 실제로 나가지 않음
2. heartbeat/reconcile 루프 정상
3. 치명 에러 없이 30~60분 유지

## Stage 1 (Tiny Live)

```bash
scripts/canary_stage1_tiny_live.sh
```

검증:
1. 체결 로그(`live_fill`)가 누적됨
2. `logs/result.json`의 `unsettled_notional`, `available_bankroll`이 정상
3. 손실 한도 도달 시 hard kill 작동
4. `MAX_LIVE_DRAWDOWN=1 scripts/canary_stage1_tiny_live.sh`로 `live_drawdown_limit` kill 재현 가능
5. `MAX_MARKET_NOTIONAL=1`로 market notional 제한 동작 재현 가능

## Resume Drill

목적: 프로세스 중단/재시작에서도 상태/리컨실/안전게이트가 연속 동작하는지 확인.

1. `scripts/canary_stage0_no_trade.sh` 실행
2. 체크포인트 파일 생성 확인:
   - `state/runtime_state.json`
   - `state/policy_state.json`
3. 프로세스 강제 종료(`kill -9 <pid>`)
4. 동일 명령으로 재실행
5. 확인 포인트:
   - startup에서 checkpoint load 로그가 출력됨
   - kill latch 상태가 이전 값대로 복원됨
   - `--require-manual-unlatch` 사용 중이면 `--manual-unlatch` 없이 자동 재개되지 않음
   - orphan order 감지 시 현재 정책(`kill` 또는 `cancel`)대로 동작

Kill 발생 후 운영자 조치:
1. `scripts/emergency_stop.sh` 실행
2. open orders가 0인지 확인
3. `logs/app.log`에서 kill reason 확인 후 원인 제거 전 재가동 금지

## Emergency Stop

```bash
scripts/emergency_stop.sh
```

확인:
1. `open_orders_after`가 0인지 확인
2. 0이 아니면 Polymarket UI/API에서 수동 정리

## Post-check
1. `tail -n 200 logs/app.log`
2. `tail -n 50 logs/executions.csv`
3. `cat logs/result.json`
