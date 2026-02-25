from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from pm1h_edge_trader.execution import KillSwitchReason  # noqa: E402
from pm1h_edge_trader.risk.circuit_breaker import (  # noqa: E402
    CircuitBreakerSnapshot,
    evaluate_circuit_breaker,
)


class RiskCircuitBreakerTests(unittest.TestCase):
    def test_triggers_daily_loss_limit(self) -> None:
        snapshot = CircuitBreakerSnapshot(
            daily_realized_pnl=-250.0,
            max_daily_loss=200.0,
            decision_bankroll=100.0,
            market_notional=0.0,
            max_market_notional=500.0,
        )

        result = evaluate_circuit_breaker(snapshot)

        self.assertTrue(result.triggered)
        self.assertIn(KillSwitchReason.DAILY_LOSS_LIMIT, result.reasons)

    def test_triggers_bankroll_depleted(self) -> None:
        snapshot = CircuitBreakerSnapshot(
            daily_realized_pnl=0.0,
            max_daily_loss=200.0,
            decision_bankroll=0.0,
            market_notional=0.0,
            max_market_notional=500.0,
        )

        result = evaluate_circuit_breaker(snapshot)

        self.assertTrue(result.triggered)
        self.assertIn(KillSwitchReason.BANKROLL_DEPLETED, result.reasons)

    def test_triggers_market_notional_limit_breach(self) -> None:
        snapshot = CircuitBreakerSnapshot(
            daily_realized_pnl=0.0,
            max_daily_loss=200.0,
            decision_bankroll=100.0,
            market_notional=510.0,
            max_market_notional=500.0,
        )

        result = evaluate_circuit_breaker(snapshot)

        self.assertTrue(result.triggered)
        self.assertIn(KillSwitchReason.MARKET_NOTIONAL_LIMIT_BREACH, result.reasons)

    def test_triggers_worst_case_loss_limit(self) -> None:
        snapshot = CircuitBreakerSnapshot(
            daily_realized_pnl=0.0,
            max_daily_loss=200.0,
            decision_bankroll=100.0,
            market_notional=100.0,
            max_market_notional=500.0,
            worst_case_loss=120.0,
            max_worst_case_loss=100.0,
        )

        result = evaluate_circuit_breaker(snapshot)

        self.assertTrue(result.triggered)
        self.assertIn(KillSwitchReason.WORST_CASE_LOSS_LIMIT, result.reasons)

    def test_returns_not_triggered_when_within_limits(self) -> None:
        snapshot = CircuitBreakerSnapshot(
            daily_realized_pnl=-50.0,
            max_daily_loss=200.0,
            decision_bankroll=100.0,
            market_notional=150.0,
            max_market_notional=500.0,
        )

        result = evaluate_circuit_breaker(snapshot)

        self.assertFalse(result.triggered)
        self.assertEqual(result.reasons, ())


if __name__ == "__main__":
    unittest.main()
