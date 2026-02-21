from __future__ import annotations

import sys
import unittest
from datetime import datetime, timezone
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from pm1h_edge_trader.execution import (
    DryRunExecutionAdapter,
    ExecutionActionType,
    ExecutionConfig,
    IntentSignal,
    KillSwitchReason,
    LimitOrderExecutionEngine,
    SafetySurface,
    Side,
)


class MvpValidationIntegrationTests(unittest.TestCase):
    def _engine(
        self,
        *,
        max_data_age_s: float = 3.0,
        fee_must_be_zero: bool = True,
    ) -> LimitOrderExecutionEngine:
        config = ExecutionConfig(
            max_data_age_s=max_data_age_s,
            fee_must_be_zero=fee_must_be_zero,
            entry_block_window_s=60,
        )
        now = datetime(2026, 2, 16, tzinfo=timezone.utc)
        adapter = DryRunExecutionAdapter(id_prefix="it", now_fn=lambda: now)
        return LimitOrderExecutionEngine(adapter=adapter, config=config, now_fn=lambda: now)

    @staticmethod
    def _signal(
        *,
        edge: float = 0.03,
        min_edge: float = 0.01,
        allow_entry: bool = True,
    ) -> IntentSignal:
        now = datetime(2026, 2, 16, tzinfo=timezone.utc)
        return IntentSignal(
            market_id="mkt-1",
            token_id="token-up",
            side=Side.BUY,
            edge=edge,
            min_edge=min_edge,
            desired_price=0.55,
            size=10.0,
            seconds_to_expiry=600,
            signal_ts=now,
            allow_entry=allow_entry,
        )

    @staticmethod
    def _safety(
        *,
        fee_rate: float = 0.0,
        data_age_s: float = 1.0,
    ) -> SafetySurface:
        return SafetySurface(
            fee_rate=fee_rate,
            data_age_s=data_age_s,
            clock_drift_s=0.0,
            book_is_consistent=True,
        )

    def test_market_rules_validation_gate_blocks_trade(self) -> None:
        engine = self._engine()
        result = engine.process_signal(
            self._signal(edge=0.05, min_edge=0.01, allow_entry=False),
            self._safety(),
        )

        self.assertEqual(len(result.actions), 1)
        self.assertEqual(result.actions[0].action_type, ExecutionActionType.SKIP)
        self.assertEqual(result.actions[0].reason, "entry_not_allowed")
        self.assertFalse(result.kill_switch_active)
        self.assertEqual(engine.active_intents(), {})

    def test_no_trade_when_edge_below_threshold(self) -> None:
        engine = self._engine()
        result = engine.process_signal(
            self._signal(edge=0.01, min_edge=0.02),
            self._safety(),
        )

        self.assertEqual(len(result.actions), 1)
        self.assertEqual(result.actions[0].action_type, ExecutionActionType.SKIP)
        self.assertEqual(result.actions[0].reason, "edge_below_threshold")
        self.assertFalse(result.kill_switch_active)
        self.assertEqual(engine.active_intents(), {})

    def test_trade_when_edge_positive_and_fee_zero(self) -> None:
        engine = self._engine()
        result = engine.process_signal(
            self._signal(edge=0.03, min_edge=0.01),
            self._safety(fee_rate=0.0),
        )

        self.assertEqual(len(result.actions), 1)
        self.assertEqual(result.actions[0].action_type, ExecutionActionType.PLACE)
        self.assertEqual(result.actions[0].reason, "new_intent")
        self.assertFalse(result.kill_switch_active)
        self.assertEqual(len(engine.active_intents()), 1)

    def test_kill_switch_on_stale_data_or_non_zero_fee(self) -> None:
        scenarios = [
            (
                "stale data",
                self._safety(fee_rate=0.0, data_age_s=10.0),
                KillSwitchReason.DATA_STALE,
            ),
            (
                "non-zero fee",
                self._safety(fee_rate=0.001, data_age_s=1.0),
                KillSwitchReason.FEE_NON_ZERO,
            ),
        ]

        for name, safety, expected_reason in scenarios:
            with self.subTest(name=name):
                engine = self._engine(max_data_age_s=3.0, fee_must_be_zero=True)
                result = engine.process_signal(
                    self._signal(edge=0.03, min_edge=0.01),
                    safety,
                )
                self.assertEqual(len(result.actions), 1)
                self.assertEqual(
                    result.actions[0].action_type,
                    ExecutionActionType.KILL_SWITCH_TRIGGERED,
                )
                self.assertIn(expected_reason, result.kill_switch.reasons)
                self.assertTrue(result.kill_switch_active)
                self.assertEqual(engine.active_intents(), {})

    def test_non_positive_size_is_skipped(self) -> None:
        engine = self._engine()
        signal = self._signal(edge=0.03, min_edge=0.01)
        signal = IntentSignal(
            market_id=signal.market_id,
            token_id=signal.token_id,
            side=signal.side,
            edge=signal.edge,
            min_edge=signal.min_edge,
            desired_price=signal.desired_price,
            size=0.0,
            seconds_to_expiry=signal.seconds_to_expiry,
            signal_ts=signal.signal_ts,
            allow_entry=signal.allow_entry,
        )
        result = engine.process_signal(signal, self._safety())
        self.assertEqual(len(result.actions), 1)
        self.assertEqual(result.actions[0].action_type, ExecutionActionType.SKIP)
        self.assertEqual(result.actions[0].reason, "non_positive_size")

    def test_invalid_desired_price_is_skipped(self) -> None:
        engine = self._engine()
        signal = self._signal(edge=0.03, min_edge=0.01)
        signal = IntentSignal(
            market_id=signal.market_id,
            token_id=signal.token_id,
            side=signal.side,
            edge=signal.edge,
            min_edge=signal.min_edge,
            desired_price=1.0,
            size=signal.size,
            seconds_to_expiry=signal.seconds_to_expiry,
            signal_ts=signal.signal_ts,
            allow_entry=signal.allow_entry,
        )
        result = engine.process_signal(signal, self._safety())
        self.assertEqual(len(result.actions), 1)
        self.assertEqual(result.actions[0].action_type, ExecutionActionType.SKIP)
        self.assertEqual(result.actions[0].reason, "invalid_desired_price")


if __name__ == "__main__":
    unittest.main()
