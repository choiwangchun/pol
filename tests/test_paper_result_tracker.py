from __future__ import annotations

import json
import sys
import unittest
from datetime import datetime, timezone
from pathlib import Path
from tempfile import TemporaryDirectory

ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from pm1h_edge_trader.logger import ExecutionLogRecord  # noqa: E402
from pm1h_edge_trader.paper.result_tracker import PaperResultTracker  # noqa: E402


def _record(
    *,
    status: str,
    order_id: str,
    side: str = "up",
    price: float = 0.5,
    size: float = 10.0,
    pnl: float | None = None,
    settlement_outcome: str | None = None,
) -> ExecutionLogRecord:
    return ExecutionLogRecord(
        timestamp=datetime(2026, 2, 20, 12, 0, tzinfo=timezone.utc),
        market_id="m1",
        K=0.0,
        S_t=0.0,
        tau=0.0,
        sigma=0.0,
        q_up=0.5,
        bid_up=0.0,
        ask_up=0.0,
        bid_down=0.0,
        ask_down=0.0,
        edge=0.0,
        order_id=order_id,
        side=side,
        price=price,
        size=size,
        status=status,
        settlement_outcome=settlement_outcome,
        pnl=pnl,
    )


class PaperResultTrackerTests(unittest.TestCase):
    def test_available_bankroll_subtracts_unsettled_notional(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            result_path = Path(tmp_dir) / "result.json"
            tracker = PaperResultTracker(initial_bankroll=1000.0, result_json_path=result_path)
            tracker.write_snapshot()

            self.assertEqual(tracker.available_bankroll(), 1000.0)

            tracker.on_record(_record(status="paper_fill", order_id="ord-1", price=0.6, size=100.0))
            self.assertEqual(tracker.available_bankroll(), 940.0)
            self.assertEqual(tracker.market_entry_count("m1"), 1)
            self.assertEqual(tracker.market_unsettled_notional("m1"), 60.0)

            tracker.on_record(
                _record(
                    status="paper_settle",
                    order_id="ord-1",
                    price=0.6,
                    size=100.0,
                    pnl=25.0,
                    settlement_outcome="win",
                )
            )
            self.assertEqual(tracker.available_bankroll(), 1025.0)
            self.assertEqual(tracker.market_unsettled_notional("m1"), 0.0)
            self.assertEqual(tracker.daily_realized_pnl(datetime(2026, 2, 20, tzinfo=timezone.utc)), 25.0)

    def test_available_bankroll_is_clamped_to_zero(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            result_path = Path(tmp_dir) / "result.json"
            tracker = PaperResultTracker(initial_bankroll=100.0, result_json_path=result_path)
            tracker.write_snapshot()

            tracker.on_record(_record(status="paper_fill", order_id="ord-1", price=0.95, size=200.0))
            self.assertEqual(tracker.available_bankroll(), 0.0)

    def test_updates_balance_on_fill_and_settlement_and_overwrites_json(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            result_path = Path(tmp_dir) / "result.json"
            tracker = PaperResultTracker(
                initial_bankroll=1000.0,
                result_json_path=result_path,
                now_fn=lambda: datetime(2026, 2, 20, 12, 30, tzinfo=timezone.utc),
            )
            tracker.write_snapshot()

            tracker.on_record(_record(status="paper_fill", order_id="ord-1", price=0.4, size=10.0))
            after_fill = json.loads(result_path.read_text(encoding="utf-8"))
            self.assertEqual(after_fill["initial_bankroll"], 1000.0)
            self.assertEqual(after_fill["current_balance"], 1000.0)
            self.assertEqual(after_fill["unsettled_fills"], 1)
            self.assertAlmostEqual(after_fill["unsettled_notional"], 4.0)
            self.assertEqual(after_fill["last_event"]["status"], "paper_fill")

            tracker.on_record(
                _record(
                    status="paper_settle",
                    order_id="ord-1",
                    price=0.4,
                    size=10.0,
                    pnl=6.0,
                    settlement_outcome="win",
                )
            )
            after_settle = json.loads(result_path.read_text(encoding="utf-8"))
            self.assertEqual(after_settle["settled_trades"], 1)
            self.assertEqual(after_settle["wins"], 1)
            self.assertEqual(after_settle["losses"], 0)
            self.assertEqual(after_settle["unsettled_fills"], 0)
            self.assertEqual(after_settle["realized_pnl"], 6.0)
            self.assertEqual(after_settle["current_balance"], 1006.0)
            self.assertEqual(after_settle["last_event"]["status"], "paper_settle")

    def test_ignores_duplicate_settlement_by_order_id(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            result_path = Path(tmp_dir) / "result.json"
            tracker = PaperResultTracker(initial_bankroll=1000.0, result_json_path=result_path)
            tracker.write_snapshot()

            settle = _record(
                status="paper_settle",
                order_id="ord-dup",
                price=0.5,
                size=2.0,
                pnl=1.0,
                settlement_outcome="win",
            )
            tracker.on_record(settle)
            tracker.on_record(settle)

            snapshot = json.loads(result_path.read_text(encoding="utf-8"))
            self.assertEqual(snapshot["settled_trades"], 1)
            self.assertEqual(snapshot["realized_pnl"], 1.0)
            self.assertEqual(snapshot["current_balance"], 1001.0)

    def test_supports_live_fill_and_live_settle_statuses(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            result_path = Path(tmp_dir) / "result.json"
            tracker = PaperResultTracker(
                initial_bankroll=100.0,
                result_json_path=result_path,
                fill_statuses=("live_fill",),
                settle_statuses=("live_settle",),
            )
            tracker.write_snapshot()

            tracker.on_record(_record(status="live_fill", order_id="live-1", price=0.4, size=10.0))
            self.assertAlmostEqual(tracker.unsettled_notional(), 4.0)
            self.assertAlmostEqual(tracker.available_bankroll(), 96.0)

            tracker.on_record(
                _record(
                    status="live_settle",
                    order_id="live-1",
                    price=0.4,
                    size=10.0,
                    pnl=2.5,
                    settlement_outcome="win",
                )
            )
            self.assertAlmostEqual(tracker.current_balance(), 102.5)
            self.assertAlmostEqual(tracker.unsettled_notional(), 0.0)


if __name__ == "__main__":
    unittest.main()
