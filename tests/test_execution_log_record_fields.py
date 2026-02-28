from __future__ import annotations

import sys
import unittest
from datetime import datetime, timezone
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from pm1h_edge_trader.logger.models import ExecutionLogRecord  # noqa: E402


class ExecutionLogRecordFieldTests(unittest.TestCase):
    def test_csv_fields_include_reason_and_debug_context(self) -> None:
        fields = ExecutionLogRecord.csv_fields()
        self.assertIn("reason", fields)
        self.assertIn("data_ready", fields)
        self.assertIn("entry_block_reason", fields)

    def test_as_dict_includes_reason_and_debug_context(self) -> None:
        record = ExecutionLogRecord(
            timestamp=datetime(2026, 2, 28, 12, 0, tzinfo=timezone.utc),
            market_id="mkt-1",
            K=100000.0,
            S_t=100100.0,
            tau=0.0001,
            sigma=0.5,
            q_up=0.52,
            bid_up=0.49,
            ask_up=0.51,
            bid_down=0.48,
            ask_down=0.5,
            edge=0.01,
            order_id="dry-1",
            side="up",
            price=0.51,
            size=10.0,
            status="skip",
            reason="edge_below_min",
            data_ready=False,
            entry_block_reason="external_kill_switch_latched",
            settlement_outcome=None,
            pnl=None,
        )

        payload = record.as_dict()
        self.assertEqual(payload["reason"], "edge_below_min")
        self.assertIs(payload["data_ready"], False)
        self.assertEqual(payload["entry_block_reason"], "external_kill_switch_latched")


if __name__ == "__main__":
    unittest.main()
