from __future__ import annotations

import sys
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from pm1h_edge_trader.feeds.binance_underlying import BinanceUnderlyingAdapter  # noqa: E402


class BinanceUnderlyingTimestampTests(unittest.IsolatedAsyncioTestCase):
    async def test_kline_price_time_uses_event_time_not_candle_close_time(self) -> None:
        adapter = BinanceUnderlyingAdapter(enable_websocket=False)
        now = datetime(2026, 3, 2, 7, 45, 56, tzinfo=timezone.utc)
        event_ms = int(now.timestamp() * 1000)
        future_close_ms = int((now + timedelta(minutes=14)).timestamp() * 1000)
        candle_open_ms = int((now - timedelta(minutes=45)).timestamp() * 1000)

        event = {
            "e": "kline",
            "E": event_ms,
            "k": {
                "i": "1h",
                "o": "65440.01",
                "c": "65528.01",
                "t": candle_open_ms,
                "T": future_close_ms,
            },
        }

        await adapter._apply_ws_event(event)
        snapshot = await adapter.get_snapshot()

        expected_price_time = datetime.fromtimestamp(event_ms / 1000.0, tz=timezone.utc)
        self.assertEqual(snapshot.last_price, 65528.01)
        self.assertEqual(snapshot.price_time, expected_price_time)


if __name__ == "__main__":
    unittest.main()
