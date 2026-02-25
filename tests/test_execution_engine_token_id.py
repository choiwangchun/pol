from __future__ import annotations

import sys
import unittest
from datetime import datetime, timezone
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from pm1h_edge_trader.execution import (  # noqa: E402
    CancelResult,
    ExecutionConfig,
    IntentSignal,
    LimitOrderExecutionEngine,
    OrderHandle,
    OrderStatus,
    SafetySurface,
    Side,
)


class _CapturingAdapter:
    def __init__(self, *, now: datetime) -> None:
        self._now = now
        self.last_request = None

    def place_limit_order(self, request):  # type: ignore[no-untyped-def]
        self.last_request = request
        return OrderHandle(order_id="ord-1", status=OrderStatus.OPEN, acknowledged_at=self._now)

    def cancel_order(self, order_id: str) -> CancelResult:
        return CancelResult(order_id=order_id, canceled=True, acknowledged_at=self._now)


class ExecutionEngineTokenIdTests(unittest.TestCase):
    def test_engine_propagates_signal_token_id_to_order_request_and_active_intent(self) -> None:
        now = datetime(2026, 2, 16, tzinfo=timezone.utc)
        adapter = _CapturingAdapter(now=now)
        engine = LimitOrderExecutionEngine(
            adapter=adapter,
            config=ExecutionConfig(entry_block_window_s=10.0),
            now_fn=lambda: now,
        )
        signal = IntentSignal(
            market_id="mkt-1",
            token_id="token-up",
            side=Side.BUY,
            edge=0.05,
            min_edge=0.01,
            desired_price=0.55,
            size=12.0,
            seconds_to_expiry=120.0,
            signal_ts=now,
            allow_entry=True,
        )
        safety = SafetySurface(
            fee_rate=0.0,
            data_age_s=1.0,
            clock_drift_s=0.0,
            book_is_consistent=True,
        )

        result = engine.process_signal(signal, safety, now=now)

        self.assertFalse(result.kill_switch_active)
        self.assertIsNotNone(adapter.last_request)
        self.assertEqual(adapter.last_request.token_id, "token-up")
        active = engine.active_intents()[("mkt-1", "token-up")]
        self.assertEqual(active.token_id, "token-up")

    def test_engine_tracks_two_tokens_in_same_market_without_key_collision(self) -> None:
        now = datetime(2026, 2, 16, tzinfo=timezone.utc)
        adapter = _CapturingAdapter(now=now)
        engine = LimitOrderExecutionEngine(
            adapter=adapter,
            config=ExecutionConfig(entry_block_window_s=10.0),
            now_fn=lambda: now,
        )
        safety = SafetySurface(
            fee_rate=0.0,
            data_age_s=1.0,
            clock_drift_s=0.0,
            book_is_consistent=True,
        )
        first = IntentSignal(
            market_id="mkt-1",
            token_id="token-up",
            side=Side.BUY,
            edge=0.05,
            min_edge=0.01,
            desired_price=0.55,
            size=12.0,
            seconds_to_expiry=120.0,
            signal_ts=now,
            allow_entry=True,
        )
        second = IntentSignal(
            market_id="mkt-1",
            token_id="token-down",
            side=Side.SELL,
            edge=0.04,
            min_edge=0.01,
            desired_price=0.45,
            size=9.0,
            seconds_to_expiry=120.0,
            signal_ts=now,
            allow_entry=True,
        )

        engine.process_signal(first, safety, now=now)
        engine.process_signal(second, safety, now=now)

        intents = engine.active_intents()
        self.assertIn(("mkt-1", "token-up"), intents)
        self.assertIn(("mkt-1", "token-down"), intents)


if __name__ == "__main__":
    unittest.main()
