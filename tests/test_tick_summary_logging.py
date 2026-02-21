from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from pm1h_edge_trader.engine import DecisionOutcome, TradeIntent  # noqa: E402
from pm1h_edge_trader.main import _tick_decision_metrics  # noqa: E402


class TickSummaryLoggingTests(unittest.TestCase):
    def test_tick_decision_metrics_default_to_zero_when_decision_missing(self) -> None:
        metrics = _tick_decision_metrics(None)
        self.assertEqual(
            metrics,
            {
                "up_kelly": 0.0,
                "down_kelly": 0.0,
                "up_notional": 0.0,
                "down_notional": 0.0,
            },
        )

    def test_tick_decision_metrics_expose_kelly_and_notional(self) -> None:
        decision = DecisionOutcome(
            sigma=0.4,
            q_up=0.61,
            q_down=0.39,
            up=TradeIntent(
                side="up",
                should_trade=True,
                probability=0.61,
                ask=0.52,
                edge=0.09,
                edge_after_buffer=0.08,
                kelly_fraction=0.04,
                order_size=20.0,
                reason="trade",
            ),
            down=TradeIntent(
                side="down",
                should_trade=False,
                probability=0.39,
                ask=0.50,
                edge=-0.11,
                edge_after_buffer=-0.12,
                kelly_fraction=0.0,
                order_size=0.0,
                reason="edge_below_min",
            ),
        )

        metrics = _tick_decision_metrics(decision)
        self.assertEqual(metrics["up_kelly"], 0.04)
        self.assertEqual(metrics["down_kelly"], 0.0)
        self.assertEqual(metrics["up_notional"], 20.0)
        self.assertEqual(metrics["down_notional"], 0.0)


if __name__ == "__main__":
    unittest.main()
