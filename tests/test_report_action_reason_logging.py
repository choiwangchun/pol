from __future__ import annotations

import sys
import unittest
from datetime import datetime, timezone
from pathlib import Path
from tempfile import TemporaryDirectory

ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from pm1h_edge_trader.config import RuntimeMode, build_config  # noqa: E402
from pm1h_edge_trader.execution import (  # noqa: E402
    ExecutionAction,
    ExecutionActionType,
    IntentSignal,
    OutcomeSide,
)
from pm1h_edge_trader.feeds.models import BestQuote, UnderlyingSnapshot  # noqa: E402
from pm1h_edge_trader.main import PM1HEdgeTraderApp  # noqa: E402


class _CollectingReporter:
    def __init__(self) -> None:
        self.records = []

    def append(self, record):  # type: ignore[no-untyped-def]
        self.records.append(record)


class ReportActionReasonLoggingTests(unittest.TestCase):
    def test_report_action_includes_reason_and_debug_context(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            config = build_config(
                mode=RuntimeMode.DRY_RUN,
                binance_symbol="BTCUSDT",
                market_slug=None,
                tick_seconds=1.0,
                max_ticks=1,
                bankroll=1000.0,
                edge_min=0.015,
                edge_buffer=0.01,
                kelly_fraction=0.25,
                f_cap=0.05,
                min_order_notional=5.0,
                rv_fallback=0.55,
                sigma_weight=1.0,
                iv_override=None,
                log_dir=Path(tmp_dir),
                enable_websocket=False,
            )
            app = PM1HEdgeTraderApp(config)
            app._reporter = _CollectingReporter()  # type: ignore[assignment]
            app._paper_result_tracker = None

            now = datetime(2026, 2, 28, 12, 0, tzinfo=timezone.utc)
            signal_map = {
                "tok-up": IntentSignal(
                    market_id="mkt-1",
                    token_id="tok-up",
                    side=OutcomeSide.UP,
                    edge=0.01,
                    min_edge=0.015,
                    desired_price=0.52,
                    size=0.0,
                    seconds_to_expiry=120.0,
                    signal_ts=now,
                    allow_entry=False,
                )
            }
            action = ExecutionAction(
                action_type=ExecutionActionType.SKIP,
                market_id="mkt-1",
                token_id="tok-up",
                side=OutcomeSide.UP,
                reason="edge_below_min",
            )
            market = type("Market", (), {"market_id": "mkt-1"})()
            underlying = UnderlyingSnapshot(
                symbol="BTCUSDT",
                interval="1h",
                candle_open=100000.0,
                candle_open_time=now,
                last_price=100050.0,
                price_time=now,
                source="test",
            )
            up_quote = BestQuote(
                token_id="tok-up",
                best_bid=0.51,
                best_ask=0.52,
                updated_at=now,
                source="test",
            )
            down_quote = BestQuote(
                token_id="tok-down",
                best_bid=0.48,
                best_ask=0.49,
                updated_at=now,
                source="test",
            )

            app._report_action(
                action=action,
                signal_map=signal_map,
                market=market,
                underlying=underlying,
                up_quote=up_quote,
                down_quote=down_quote,
                tau_years=0.0001,
                sigma=0.5,
                decision=None,
                data_ready=False,
                entry_block_reason="external_kill_switch_latched",
            )

            self.assertGreaterEqual(len(app._reporter.records), 1)  # type: ignore[attr-defined]
            first = app._reporter.records[0]  # type: ignore[attr-defined]
            self.assertEqual(first.reason, "edge_below_min")
            self.assertIs(first.data_ready, False)
            self.assertEqual(first.entry_block_reason, "external_kill_switch_latched")


if __name__ == "__main__":
    unittest.main()
