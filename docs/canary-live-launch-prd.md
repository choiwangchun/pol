# PM-1H Edge Trader Canary Live Launch PRD

## 1. 목적
- Live 런칭의 1순위는 수익이 아니라 안전성/정합성/운영 가능성 검증이다.
- 아래 3가지를 카나리에서 증명한다.
1. 안전장치가 실제 강제 차단으로 동작한다.
2. 부분체결/오더 불일치/포지션 불일치에서 안전정지가 된다.
3. 운영자가 런북으로 복구/롤백 가능하다.

## 2. 성공 기준 (Exit Criteria)
- Kill-switch 트리거 시 즉시 주문 정리 + 신규 진입 중단 확인.
- `live_fill` + partial fill 누적 반영과 `result.json` 노출 값 정합성 확인.
- 알림/운영 대응이 1분 내 동작하고 Emergency Stop 후 open orders가 0으로 수렴.

## 3. 카나리 단계
### Stage 0: Live 연결 검증 (No-Trade)
- 주문 0개로 heartbeat/reconcile/position-check 루프 안정성 확인.
- 권장 명령:

```bash
PYTHONPATH=src python -m pm1h_edge_trader.main \
  --mode live \
  --edge-min 1.0 \
  --max-market-notional 1 \
  --max-daily-loss 1 \
  --kelly-fraction 0.0 \
  --f-cap 0.0 \
  --disable-auto-claim
```

### Stage 1: 초소액 체결
- 매우 보수적 cap으로 1~N건 체결 정합성 검증.
- 권장 시작 범위:
1. `--max-market-notional 5~25`
2. `--max-daily-loss 2~10`
3. `--f-cap 0.005~0.01`
4. `--kelly-fraction 0.05~0.10`
5. `--max-entries-per-market 1`

### Stage 2: 운영 드릴
- heartbeat 중단, cancel-all, position mismatch 케이스를 강제로 발생시켜 런북 검증.

### Stage 3: 기능 확장 (옵트인)
1. complete-set arb ON (아주 작은 notional, 짧은 timeout)
2. policy bandit shadow ON
3. auto-claim ON

## 4. 운영 정책
- 전용 지갑만 사용(수동 오더와 혼용 금지).
- 카나리 자금만 계정에 유지.
- 초기에는 `--disable-auto-claim` 권장.
- orphan order 기본 정책은 자동 취소가 아니라 kill + 수동확인.

## 5. 모니터링 핵심 지표
- Risk: `available_bankroll`, `unsettled_notional`, `daily_realized_pnl`, `open_orders_count`
- Execution: place/cancel 성공률, fill rate, partial fill 발생률
- Feed: data age, reconnect 횟수, fee-rate guard 결과
- Claim(옵션): claim 시도/성공/실패/timeout

## 6. 알림 우선순위
- Sev0: kill-switch latch, heartbeat 연속 실패, cancel-all 실패, position mismatch kill
- Sev1: reconcile 연속 실패, fee-rate 이상, data stale 반복
- Sev2: spread 확대, IV/RV fallback 과다

## 7. 롤백 전략
### 소프트 롤백
- 즉시 거래 정지: `--edge-min 1.0 --kelly-fraction 0 --f-cap 0`
- 기능 OFF: `--disable-auto-claim`, `--enable-complete-set-arb` 제거, bandit OFF

### 하드 롤백
1. 프로세스 종료
2. emergency stop 실행
3. stable 태그 커밋으로 복귀
4. Stage 0부터 재검증

## 8. Emergency Stop 표준
1. 프로세스 종료(SIGINT)
2. `pm1h-emergency-stop` 실행
3. open orders 0 확인
4. `/positions`로 잔존 포지션 확인
5. 재가동은 Stage 0부터
