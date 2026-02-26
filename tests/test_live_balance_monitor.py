from __future__ import annotations

import sys
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from pm1h_edge_trader.config import PolymarketLiveAuthConfig  # noqa: E402
from pm1h_edge_trader.live_balance import LiveBalanceSnapshot  # noqa: E402
from pm1h_edge_trader.risk.live_balance_monitor import (  # noqa: E402
    LiveBalanceMonitor,
)


class LiveBalanceMonitorTests(unittest.TestCase):
    def test_maybe_refresh_respects_interval(self) -> None:
        now = datetime(2026, 2, 26, 0, 0, tzinfo=timezone.utc)
        calls = {"count": 0}

        def _fetch(*, auth, host, adapter_factory=None):  # type: ignore[no-untyped-def]
            calls["count"] += 1
            return LiveBalanceSnapshot(balance=100.0, allowance=100.0, bankroll=100.0)

        monitor = LiveBalanceMonitor(
            auth=PolymarketLiveAuthConfig(private_key="k", funder="f"),
            host="https://clob.polymarket.com",
            refresh_seconds=30.0,
            max_drawdown=10.0,
            fetch_fn=_fetch,
        )

        first = monitor.maybe_refresh(now=now)
        second = monitor.maybe_refresh(now=now + timedelta(seconds=5))
        third = monitor.maybe_refresh(now=now + timedelta(seconds=31))

        self.assertIsNotNone(first)
        self.assertIsNone(second)
        self.assertIsNotNone(third)
        self.assertEqual(calls["count"], 2)

    def test_drawdown_trigger_uses_baseline_bankroll(self) -> None:
        now = datetime(2026, 2, 26, 0, 0, tzinfo=timezone.utc)
        snapshots = iter(
            [
                LiveBalanceSnapshot(balance=100.0, allowance=100.0, bankroll=100.0),
                LiveBalanceSnapshot(balance=88.0, allowance=88.0, bankroll=88.0),
            ]
        )

        def _fetch(*, auth, host, adapter_factory=None):  # type: ignore[no-untyped-def]
            return next(snapshots)

        monitor = LiveBalanceMonitor(
            auth=PolymarketLiveAuthConfig(private_key="k", funder="f"),
            host="https://clob.polymarket.com",
            refresh_seconds=1.0,
            max_drawdown=10.0,
            fetch_fn=_fetch,
        )
        first = monitor.maybe_refresh(now=now)
        second = monitor.maybe_refresh(now=now + timedelta(seconds=2))

        self.assertIsNotNone(first)
        self.assertIsNotNone(second)
        assert second is not None
        self.assertAlmostEqual(second.drawdown, 12.0)
        self.assertTrue(second.triggered)


if __name__ == "__main__":
    unittest.main()
