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
    Side,
)
from pm1h_edge_trader.feeds.models import BestQuote, UnderlyingSnapshot  # noqa: E402
from pm1h_edge_trader.main import PM1HEdgeTraderApp, build_arg_parser  # noqa: E402


class _FakePolicyController:
    def __init__(self) -> None:
        self.fill_calls = []

    def on_fill(self, record, *, selection=None):  # type: ignore[no-untyped-def]
        self.fill_calls.append((record, selection))


class PolicyRuntimeWiringTests(unittest.TestCase):
    def _build_config(self):
        with TemporaryDirectory() as tmp_dir:
            return build_config(
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
                enable_policy_bandit=True,
                policy_shadow_mode=True,
                policy_exploration_epsilon=0.1,
                policy_ucb_c=1.5,
            )

    def test_arg_parser_accepts_policy_flags(self) -> None:
        parser = build_arg_parser()
        args = parser.parse_args(
            [
                "--enable-policy-bandit",
                "--policy-shadow-mode",
                "--policy-exploration-epsilon",
                "0.2",
                "--policy-ucb-c",
                "1.4",
            ]
        )
        self.assertTrue(args.enable_policy_bandit)
        self.assertTrue(args.policy_shadow_mode)
        self.assertAlmostEqual(args.policy_exploration_epsilon, 0.2)
        self.assertAlmostEqual(args.policy_ucb_c, 1.4)

    def test_build_config_maps_policy_fields(self) -> None:
        config = self._build_config()
        self.assertTrue(config.policy.enabled)
        self.assertTrue(config.policy.shadow_mode)
        self.assertAlmostEqual(config.policy.exploration_epsilon, 0.1)
        self.assertAlmostEqual(config.policy.ucb_c, 1.5)

    def test_report_action_for_paper_fill_notifies_policy_controller(self) -> None:
        config = self._build_config()
        app = PM1HEdgeTraderApp(config)
        fake_policy = _FakePolicyController()
        app._policy_controller = fake_policy
        now = datetime(2026, 2, 23, 9, 30, tzinfo=timezone.utc)

        signal_map = {
            Side.BUY: IntentSignal(
                market_id="mkt-1",
                token_id="tok-up",
                side=Side.BUY,
                edge=0.02,
                min_edge=0.01,
                desired_price=0.55,
                size=10.0,
                seconds_to_expiry=1200.0,
                signal_ts=now,
                allow_entry=True,
            ),
            Side.SELL: IntentSignal(
                market_id="mkt-1",
                token_id="tok-down",
                side=Side.SELL,
                edge=0.0,
                min_edge=0.01,
                desired_price=0.45,
                size=0.0,
                seconds_to_expiry=1200.0,
                signal_ts=now,
                allow_entry=False,
            ),
        }
        action = ExecutionAction(
            action_type=ExecutionActionType.PLACE,
            market_id="mkt-1",
            side=Side.BUY,
            order_id="dry-77",
            reason="new_intent",
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
            best_bid=0.54,
            best_ask=0.55,
            updated_at=now,
            source="test",
        )
        down_quote = BestQuote(
            token_id="tok-down",
            best_bid=0.44,
            best_ask=0.45,
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
        )

        self.assertEqual(len(fake_policy.fill_calls), 1)
        self.assertEqual(fake_policy.fill_calls[0][0].order_id, "dry-77")


if __name__ == "__main__":
    unittest.main()
