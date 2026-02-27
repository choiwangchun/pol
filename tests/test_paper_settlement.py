from __future__ import annotations

import csv
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
from pm1h_edge_trader.paper.settlement import (  # noqa: E402
    MarketResolution,
    MarketResolutionResolver,
    PaperSettlementCoordinator,
)


class _MemoryReporter:
    def __init__(self) -> None:
        self.records: list[ExecutionLogRecord] = []

    def append(self, record: ExecutionLogRecord) -> None:
        self.records.append(record)


class _FakeResolver(MarketResolutionResolver):
    def __init__(self, winners: dict[str, str | None]) -> None:
        self._winners = winners

    async def resolve_market(self, market_id: str) -> MarketResolution:
        winner = self._winners.get(market_id)
        return MarketResolution(winner=winner, resolved=winner in {"up", "down"})


def _base_row(
    *,
    timestamp: str,
    market_id: str,
    order_id: str,
    side: str,
    price: float,
    size: float,
    status: str,
    settlement_outcome: str = "",
    pnl: str = "",
) -> dict[str, str]:
    return {
        "timestamp": timestamp,
        "market_id": market_id,
        "K": "0",
        "S_t": "0",
        "tau": "0",
        "sigma": "0",
        "q_up": "0.5",
        "bid_up": "0",
        "ask_up": "0",
        "bid_down": "0",
        "ask_down": "0",
        "edge": "0",
        "order_id": order_id,
        "side": side,
        "price": str(price),
        "size": str(size),
        "status": status,
        "settlement_outcome": settlement_outcome,
        "pnl": pnl,
    }


class PaperSettlementTests(unittest.IsolatedAsyncioTestCase):
    async def test_reconcile_appends_settlement_for_unresolved_paper_fills(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            csv_path = Path(tmp_dir) / "executions.csv"
            rows = [
                _base_row(
                    timestamp="2026-02-20T10:00:00+00:00",
                    market_id="m-win",
                    order_id="o-win",
                    side="up",
                    price=0.40,
                    size=10.0,
                    status="paper_fill",
                ),
                _base_row(
                    timestamp="2026-02-20T10:00:00+00:00",
                    market_id="m-loss",
                    order_id="o-loss",
                    side="down",
                    price=0.20,
                    size=5.0,
                    status="paper_fill",
                ),
                _base_row(
                    timestamp="2026-02-20T10:05:00+00:00",
                    market_id="m-already",
                    order_id="o-already",
                    side="up",
                    price=0.40,
                    size=10.0,
                    status="paper_fill",
                ),
                _base_row(
                    timestamp="2026-02-20T11:10:00+00:00",
                    market_id="m-already",
                    order_id="o-already",
                    side="up",
                    price=0.40,
                    size=10.0,
                    status="paper_settle",
                    settlement_outcome="win",
                    pnl="6.0",
                ),
                _base_row(
                    timestamp="2026-02-20T10:10:00+00:00",
                    market_id="m-unresolved",
                    order_id="o-unresolved",
                    side="up",
                    price=0.30,
                    size=9.0,
                    status="paper_fill",
                ),
            ]

            with csv_path.open("w", newline="", encoding="utf-8") as fp:
                writer = csv.DictWriter(fp, fieldnames=ExecutionLogRecord.csv_fields())
                writer.writeheader()
                writer.writerows(rows)

            reporter = _MemoryReporter()
            resolver = _FakeResolver(
                {
                    "m-win": "up",
                    "m-loss": "up",
                    "m-already": "up",
                    "m-unresolved": None,
                }
            )
            coordinator = PaperSettlementCoordinator(
                execution_csv_path=csv_path,
                reporter=reporter,
                resolver=resolver,
                now_fn=lambda: datetime(2026, 2, 20, 12, 0, tzinfo=timezone.utc),
            )

            count = await coordinator.reconcile()

            self.assertEqual(count, 2)
            self.assertEqual(len(reporter.records), 2)
            by_order = {record.order_id: record for record in reporter.records}
            win = by_order["o-win"]
            self.assertEqual(win.status, "paper_settle")
            self.assertEqual(win.settlement_outcome, "win")
            self.assertAlmostEqual(win.pnl or 0.0, 6.0)

            loss = by_order["o-loss"]
            self.assertEqual(loss.status, "paper_settle")
            self.assertEqual(loss.settlement_outcome, "loss")
            self.assertAlmostEqual(loss.pnl or 0.0, -1.0)

    async def test_reconcile_supports_live_fill_and_live_settle_statuses(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            csv_path = Path(tmp_dir) / "executions.csv"
            rows = [
                _base_row(
                    timestamp="2026-02-20T10:00:00+00:00",
                    market_id="m-live",
                    order_id="o-live",
                    side="up",
                    price=0.40,
                    size=10.0,
                    status="live_fill",
                )
            ]
            with csv_path.open("w", newline="", encoding="utf-8") as fp:
                writer = csv.DictWriter(fp, fieldnames=ExecutionLogRecord.csv_fields())
                writer.writeheader()
                writer.writerows(rows)

            reporter = _MemoryReporter()
            resolver = _FakeResolver({"m-live": "up"})
            coordinator = PaperSettlementCoordinator(
                execution_csv_path=csv_path,
                reporter=reporter,
                resolver=resolver,
                now_fn=lambda: datetime(2026, 2, 20, 12, 0, tzinfo=timezone.utc),
                fill_statuses=("live_fill",),
                settle_statuses=("live_settle",),
            )

            count = await coordinator.reconcile()

            self.assertEqual(count, 1)
            self.assertEqual(len(reporter.records), 1)
            record = reporter.records[0]
            self.assertEqual(record.status, "live_settle")
            self.assertEqual(record.order_id, "o-live")


if __name__ == "__main__":
    unittest.main()
