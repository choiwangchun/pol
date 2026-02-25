from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from pm1h_edge_trader.strategy.complete_set_arb import decide_complete_set_arb  # noqa: E402


class CompleteSetArbTests(unittest.TestCase):
    def test_no_trade_when_ask_sum_at_or_above_one(self) -> None:
        at_par = decide_complete_set_arb(
            ask_up=0.5,
            ask_down=0.5,
            bankroll=100.0,
            min_profit=0.0,
            max_notional=50.0,
            min_order_notional=1.0,
        )
        above_par = decide_complete_set_arb(
            ask_up=0.6,
            ask_down=0.5,
            bankroll=100.0,
            min_profit=0.0,
            max_notional=50.0,
            min_order_notional=1.0,
        )

        self.assertFalse(at_par.should_trade)
        self.assertFalse(above_par.should_trade)

    def test_trade_when_ask_sum_below_one_and_constraints_met(self) -> None:
        result = decide_complete_set_arb(
            ask_up=0.45,
            ask_down=0.45,
            bankroll=100.0,
            min_profit=0.01,
            max_notional=50.0,
            min_order_notional=1.0,
        )

        self.assertTrue(result.should_trade)
        self.assertGreater(result.pair_tokens, 0.0)
        self.assertGreaterEqual(result.notional_up, 1.0)
        self.assertGreaterEqual(result.notional_down, 1.0)
        self.assertAlmostEqual(result.ask_sum, 0.9)
        self.assertAlmostEqual(result.profit_per_pair, 0.1)

    def test_no_trade_when_min_profit_not_met(self) -> None:
        result = decide_complete_set_arb(
            ask_up=0.48,
            ask_down=0.48,
            bankroll=100.0,
            min_profit=0.05,
            max_notional=50.0,
            min_order_notional=1.0,
        )
        self.assertFalse(result.should_trade)

    def test_no_trade_when_min_order_notional_not_met(self) -> None:
        result = decide_complete_set_arb(
            ask_up=0.45,
            ask_down=0.45,
            bankroll=2.0,
            min_profit=0.0,
            max_notional=2.0,
            min_order_notional=2.0,
        )
        self.assertFalse(result.should_trade)


if __name__ == "__main__":
    unittest.main()
