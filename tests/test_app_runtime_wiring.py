from __future__ import annotations

import sys
import unittest
import csv
import asyncio
from datetime import datetime, timezone
from datetime import timedelta
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace

ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from pm1h_edge_trader.config import (  # noqa: E402
    PolymarketLiveAuthConfig,
    RuntimeMode,
    build_config,
)
from pm1h_edge_trader.engine import DecisionOutcome, TradeIntent  # noqa: E402
from pm1h_edge_trader.execution import DryRunExecutionAdapter  # noqa: E402
from pm1h_edge_trader.execution import (  # noqa: E402
    ActiveIntent,
    CancelResult,
    ExecutionAction,
    ExecutionActionType,
    ExecutionConfig,
    IntentSignal,
    LimitOrderExecutionEngine,
    OrderHandle,
    OrderRequest,
    OrderStatus,
    OutcomeSide,
)
from pm1h_edge_trader.execution.polymarket_live_adapter import (  # noqa: E402
    PolymarketLiveExecutionAdapter,
)
from pm1h_edge_trader.feeds.models import BestQuote, UnderlyingSnapshot  # noqa: E402
from pm1h_edge_trader.main import PM1HEdgeTraderApp  # noqa: E402


class _FakeLiveOpsAdapter:
    def __init__(self, *, now: datetime) -> None:
        self._now = now
        self._counter = 0
        self.open_order_ids: set[str] = set()
        self.canceled_order_ids: list[str] = []
        self.place_requests: list[OrderRequest] = []
        self.cancel_all_calls = 0
        self.heartbeat_calls = 0
        self.heartbeat_ok = True
        self.tick_sizes: dict[str, float] = {}
        self.order_payloads: dict[str, object] = {}
        self.trade_payloads: dict[str, object] = {}
        self.get_order_calls: list[str] = []
        self.get_trade_calls: list[str] = []
        self.raise_from_list_open_order_ids: list[Exception] = []

    def place_limit_order(self, request: OrderRequest) -> OrderHandle:
        self._counter += 1
        order_id = f"live-{self._counter}"
        self.place_requests.append(request)
        self.open_order_ids.add(order_id)
        return OrderHandle(order_id=order_id, status=OrderStatus.OPEN, acknowledged_at=self._now)

    def cancel_order(self, order_id: str) -> CancelResult:
        self.canceled_order_ids.append(order_id)
        canceled = order_id in self.open_order_ids
        self.open_order_ids.discard(order_id)
        return CancelResult(order_id=order_id, canceled=canceled, acknowledged_at=self._now)

    def list_open_order_ids(self) -> set[str]:
        if self.raise_from_list_open_order_ids:
            raise self.raise_from_list_open_order_ids.pop(0)
        return set(self.open_order_ids)

    def cancel_all_orders(self) -> int:
        self.cancel_all_calls += 1
        count = len(self.open_order_ids)
        self.open_order_ids.clear()
        return count

    def send_heartbeat(self, *, heartbeat_id: str | None = None) -> bool:
        self.heartbeat_calls += 1
        return self.heartbeat_ok

    def get_tick_size(self, token_id: str) -> float | None:
        return self.tick_sizes.get(token_id)

    def get_order(self, order_id: str) -> object | None:
        self.get_order_calls.append(order_id)
        payload = self.order_payloads.get(order_id)
        if isinstance(payload, Exception):
            raise payload
        return payload

    def get_trade(self, trade_id: str) -> object | None:
        self.get_trade_calls.append(trade_id)
        payload = self.trade_payloads.get(trade_id)
        if isinstance(payload, Exception):
            raise payload
        return payload


class AppRuntimeWiringTests(unittest.TestCase):
    def _build_config(
        self,
        *,
        mode: RuntimeMode,
        bankroll: float = 10_000.0,
        cancel_orphan_orders: bool = False,
        enable_complete_set_arb: bool = False,
        adopt_existing_positions: bool = False,
        adopt_existing_positions_policy: str = "block",
        position_mismatch_policy: str = "kill",
        max_live_drawdown: float = 0.0,
        live_balance_refresh_seconds: float = 30.0,
    ):
        with TemporaryDirectory() as tmp_dir:
            return build_config(
                mode=mode,
                binance_symbol="BTCUSDT",
                market_slug=None,
                tick_seconds=1.0,
                max_ticks=1,
                bankroll=bankroll,
                edge_min=0.015,
                edge_buffer=0.010,
                kelly_fraction=0.25,
                f_cap=0.05,
                min_order_notional=5.0,
                rv_fallback=0.55,
                sigma_weight=1.0,
                iv_override=None,
                log_dir=Path(tmp_dir),
                enable_websocket=False,
                cancel_orphan_orders=cancel_orphan_orders,
                enable_complete_set_arb=enable_complete_set_arb,
                adopt_existing_positions=adopt_existing_positions,
                adopt_existing_positions_policy=adopt_existing_positions_policy,
                position_mismatch_policy=position_mismatch_policy,
                max_live_drawdown=max_live_drawdown,
                live_balance_refresh_seconds=live_balance_refresh_seconds,
                polymarket_live_auth=PolymarketLiveAuthConfig(
                    private_key="0x1111111111111111111111111111111111111111111111111111111111111111",
                    funder="0x2222222222222222222222222222222222222222",
                ),
            )

    def test_dry_run_mode_uses_dry_adapter(self) -> None:
        config = self._build_config(mode=RuntimeMode.DRY_RUN)
        app = PM1HEdgeTraderApp(config)
        self.assertIsInstance(app._execution_engine._adapter, DryRunExecutionAdapter)

    def test_live_mode_uses_polymarket_live_adapter(self) -> None:
        config = self._build_config(mode=RuntimeMode.LIVE)
        app = PM1HEdgeTraderApp(config)
        self.assertIsInstance(app._execution_engine._adapter, PolymarketLiveExecutionAdapter)
        self.assertIsNotNone(app._paper_settlement)

    def test_build_signal_uses_quote_token_context(self) -> None:
        config = self._build_config(mode=RuntimeMode.DRY_RUN)
        app = PM1HEdgeTraderApp(config)
        now = datetime(2026, 2, 16, tzinfo=timezone.utc)
        quote = BestQuote(
            token_id="token-from-book",
            best_bid=0.45,
            best_ask=0.55,
            updated_at=now,
            source="test",
        )

        signal = app._build_signal(
            market_id="market-1",
            token_id="token-fallback",
            side_label="up",
            now=now,
            seconds_to_expiry=120.0,
            quote=quote,
            decision=None,
            data_ready=False,
        )

        self.assertEqual(signal.token_id, "token-from-book")

    def test_decision_bankroll_uses_tracker_available_bankroll(self) -> None:
        config = self._build_config(mode=RuntimeMode.DRY_RUN)
        app = PM1HEdgeTraderApp(config)

        class _Tracker:
            def available_bankroll(self) -> float:
                return 321.0

        app._paper_result_tracker = _Tracker()
        self.assertEqual(app._decision_bankroll(), 321.0)

    def test_report_action_marks_paper_fill_as_filled_and_clears_intent(self) -> None:
        config = self._build_config(mode=RuntimeMode.DRY_RUN)
        app = PM1HEdgeTraderApp(config)
        now = datetime(2026, 2, 21, 6, 30, tzinfo=timezone.utc)

        adapter = app._execution_engine._adapter
        self.assertIsInstance(adapter, DryRunExecutionAdapter)
        handle = adapter.place_limit_order(
            OrderRequest(
                market_id="mkt-1",
                token_id="tok-up",
                side=OutcomeSide.UP,
                price=0.55,
                size=10.0,
                submitted_at=now,
            )
        )

        app._execution_engine._active_intents[("mkt-1", "tok-up")] = ActiveIntent(
            order_id=handle.order_id,
            market_id="mkt-1",
            token_id="tok-up",
            side=OutcomeSide.UP,
            price=0.55,
            size=10.0,
            edge=0.02,
            created_at=now,
            last_quoted_at=now,
            status=OrderStatus.OPEN,
        )

        signal_map = {
            "tok-up": IntentSignal(
                market_id="mkt-1",
                token_id="tok-up",
                side=OutcomeSide.UP,
                edge=0.02,
                min_edge=0.01,
                desired_price=0.55,
                size=10.0,
                seconds_to_expiry=1200.0,
                signal_ts=now,
                allow_entry=True,
            ),
            "tok-down": IntentSignal(
                market_id="mkt-1",
                token_id="tok-down",
                side=OutcomeSide.DOWN,
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
            token_id="tok-up",
            side=OutcomeSide.UP,
            order_id=handle.order_id,
            reason="new_intent",
        )
        market = SimpleNamespace(market_id="mkt-1")
        underlying = UnderlyingSnapshot(
            symbol="BTCUSDT",
            interval="1h",
            candle_open=100000.0,
            candle_open_time=now,
            last_price=100100.0,
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
            data_ready=True,
            entry_block_reason=None,
        )

        self.assertEqual(adapter.order_status(handle.order_id), OrderStatus.FILLED)
        self.assertNotIn(("mkt-1", "tok-up"), app._execution_engine.active_intents())

    def test_entry_block_reason_stops_trading_when_daily_loss_cap_hit(self) -> None:
        config = self._build_config(mode=RuntimeMode.DRY_RUN, bankroll=1000.0)
        app = PM1HEdgeTraderApp(config)

        class _Tracker:
            def daily_realized_pnl(self, as_of):  # type: ignore[no-untyped-def]
                return -250.0

            def market_entry_count(self, market_id: str) -> int:
                return 0

        app._paper_result_tracker = _Tracker()
        reason = app._entry_block_reason(
            now=datetime(2026, 2, 21, 6, 30, tzinfo=timezone.utc),
            market_id="mkt-1",
        )
        self.assertEqual(reason, "daily_loss_cap_reached")

    def test_entry_block_reason_stops_trading_when_position_mismatch_latched(self) -> None:
        config = self._build_config(mode=RuntimeMode.LIVE, bankroll=1000.0)
        app = PM1HEdgeTraderApp(config)
        app._position_mismatch_blocked = True

        reason = app._entry_block_reason(
            now=datetime(2026, 2, 21, 6, 30, tzinfo=timezone.utc),
            market_id="mkt-1",
        )

        self.assertEqual(reason, "position_mismatch")

    def test_build_signal_caps_notional_by_market_limit(self) -> None:
        config = self._build_config(mode=RuntimeMode.DRY_RUN, bankroll=100.0)
        app = PM1HEdgeTraderApp(config)
        now = datetime(2026, 2, 21, 6, 30, tzinfo=timezone.utc)

        class _Tracker:
            def market_unsettled_notional(self, market_id: str) -> float:
                return 15.0

            def daily_realized_pnl(self, as_of):  # type: ignore[no-untyped-def]
                return 0.0

            def market_entry_count(self, market_id: str) -> int:
                return 0

        app._paper_result_tracker = _Tracker()
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
                order_size=80.0,
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
        quote = BestQuote(
            token_id="tok-up",
            best_bid=0.54,
            best_ask=0.55,
            updated_at=now,
            source="test",
        )

        signal = app._build_signal(
            market_id="mkt-1",
            token_id="tok-up",
            side_label="up",
            now=now,
            seconds_to_expiry=600.0,
            quote=quote,
            decision=decision,
            data_ready=True,
        )

        # bankroll=100, default market cap is 25% => 25.0, with 15.0 already used -> remaining 10.0
        self.assertAlmostEqual(signal.size, 10.0 / 0.55, places=8)

    def test_set_market_tick_sizes_updates_price_epsilon(self) -> None:
        config = self._build_config(mode=RuntimeMode.DRY_RUN)
        app = PM1HEdgeTraderApp(config)

        app._set_market_tick_sizes({"tok-up": 0.01, "tok-down": 0.001})

        self.assertEqual(app._token_tick_sizes["tok-up"], 0.01)
        self.assertEqual(app._execution_engine._config.price_epsilon, 0.0005)

    def test_cap_order_notional_counts_active_open_intents(self) -> None:
        config = self._build_config(mode=RuntimeMode.DRY_RUN, bankroll=100.0)
        app = PM1HEdgeTraderApp(config)
        now = datetime(2026, 2, 21, 8, 0, tzinfo=timezone.utc)

        app._execution_engine._active_intents[("mkt-1", "tok-up")] = ActiveIntent(
            order_id="live-100",
            market_id="mkt-1",
            token_id="tok-up",
            side=OutcomeSide.UP,
            price=0.50,
            size=10.0,
            edge=0.03,
            created_at=now,
            last_quoted_at=now,
            status=OrderStatus.OPEN,
        )

        # cap=25.0, active notional=5.0, so remaining=20.0
        capped = app._cap_order_notional_for_market(market_id="mkt-1", suggested_notional=30.0)
        self.assertAlmostEqual(capped, 20.0)

    def test_reconcile_live_open_orders_latches_kill_switch_without_canceling_orphans_by_default(self) -> None:
        config = self._build_config(mode=RuntimeMode.LIVE)
        app = PM1HEdgeTraderApp(config)
        now = datetime(2026, 2, 21, 8, 0, tzinfo=timezone.utc)
        adapter = _FakeLiveOpsAdapter(now=now)
        engine = LimitOrderExecutionEngine(
            adapter=adapter,
            config=ExecutionConfig(entry_block_window_s=60.0),
            now_fn=lambda: now,
        )
        app._execution_engine = engine

        engine._active_intents[("mkt-1", "tok-up")] = ActiveIntent(
            order_id="live-100",
            market_id="mkt-1",
            token_id="tok-up",
            side=OutcomeSide.UP,
            price=0.50,
            size=10.0,
            edge=0.03,
            created_at=now,
            last_quoted_at=now,
            status=OrderStatus.OPEN,
        )
        adapter.open_order_ids = {"ghost-1"}

        app._reconcile_live_open_orders(now=now, force=True)

        self.assertNotIn("ghost-1", adapter.canceled_order_ids)
        self.assertIn("live-100", adapter.canceled_order_ids)
        self.assertEqual(adapter.cancel_all_calls, 0)
        self.assertTrue(app._external_kill_switch_latched)
        self.assertTrue(engine.kill_switch.active)

    def test_reconcile_live_open_orders_can_cancel_orphans_when_enabled(self) -> None:
        config = self._build_config(mode=RuntimeMode.LIVE, cancel_orphan_orders=True)
        app = PM1HEdgeTraderApp(config)
        now = datetime(2026, 2, 21, 8, 0, tzinfo=timezone.utc)
        adapter = _FakeLiveOpsAdapter(now=now)
        engine = LimitOrderExecutionEngine(
            adapter=adapter,
            config=ExecutionConfig(entry_block_window_s=60.0),
            now_fn=lambda: now,
        )
        app._execution_engine = engine
        adapter.open_order_ids = {"ghost-1"}

        app._reconcile_live_open_orders(now=now, force=True)

        self.assertIn("ghost-1", adapter.canceled_order_ids)

    def test_reconcile_live_open_orders_appends_live_fill_when_missing_order_is_filled(self) -> None:
        config = self._build_config(mode=RuntimeMode.LIVE)
        app = PM1HEdgeTraderApp(config)
        now = datetime(2026, 2, 21, 8, 0, tzinfo=timezone.utc)
        adapter = _FakeLiveOpsAdapter(now=now)
        engine = LimitOrderExecutionEngine(
            adapter=adapter,
            config=ExecutionConfig(entry_block_window_s=60.0),
            now_fn=lambda: now,
        )
        app._execution_engine = engine

        engine._active_intents[("mkt-1", "tok-up")] = ActiveIntent(
            order_id="live-100",
            market_id="mkt-1",
            token_id="tok-up",
            side=OutcomeSide.UP,
            price=0.50,
            size=10.0,
            edge=0.03,
            created_at=now,
            last_quoted_at=now,
            status=OrderStatus.OPEN,
        )
        adapter.open_order_ids = set()
        adapter.order_payloads["live-100"] = {
            "order": {
                "id": "live-100",
                "status": "MATCHED",
                "size_matched": "3.5",
                "associate_trades": ["trade-77"],
                "price": "0.52",
            }
        }
        adapter.trade_payloads["trade-77"] = {
            "id": "trade-77",
            "price": "0.53",
            "maker_orders": [{"order_id": "live-100", "matched_amount": "3.5"}],
        }

        app._reconcile_live_open_orders(now=now, force=True)

        self.assertEqual(engine.active_intents(), {})
        self.assertEqual(adapter.get_order_calls, ["live-100"])
        self.assertEqual(adapter.get_trade_calls, ["trade-77"])
        self.assertIsNone(engine._filled_entries_by_market.get("mkt-1"))

        with config.logging.execution_csv_path.open("r", encoding="utf-8", newline="") as fp:
            rows = list(csv.DictReader(fp))
        live_fills = [row for row in rows if row.get("status") == "live_fill" and row.get("order_id") == "live-100"]
        self.assertEqual(len(live_fills), 1)
        self.assertEqual(live_fills[0]["side"], "up")
        self.assertEqual(float(live_fills[0]["size"]), 3.5)
        self.assertEqual(float(live_fills[0]["price"]), 0.53)

    def test_reconcile_live_open_orders_tracks_partial_fill_progress_for_open_order(self) -> None:
        config = self._build_config(mode=RuntimeMode.LIVE)
        app = PM1HEdgeTraderApp(config)
        now = datetime(2026, 2, 21, 9, 0, tzinfo=timezone.utc)
        adapter = _FakeLiveOpsAdapter(now=now)
        engine = LimitOrderExecutionEngine(
            adapter=adapter,
            config=ExecutionConfig(entry_block_window_s=60.0),
            now_fn=lambda: now,
        )
        app._execution_engine = engine

        engine._active_intents[("mkt-1", "tok-up")] = ActiveIntent(
            order_id="live-100",
            market_id="mkt-1",
            token_id="tok-up",
            side=OutcomeSide.UP,
            price=0.50,
            size=10.0,
            edge=0.03,
            created_at=now,
            last_quoted_at=now,
            status=OrderStatus.OPEN,
        )
        adapter.open_order_ids = {"live-100"}
        adapter.order_payloads["live-100"] = {
            "order": {
                "id": "live-100",
                "status": "LIVE",
                "size_matched": "2.0",
                "price": "0.50",
            }
        }

        app._reconcile_live_open_orders(now=now, force=True)
        with config.logging.execution_csv_path.open("r", encoding="utf-8", newline="") as fp:
            rows = list(csv.DictReader(fp))
        live_fills = [row for row in rows if row.get("status") == "live_fill" and row.get("order_id") == "live-100"]
        self.assertEqual(len(live_fills), 1)
        self.assertAlmostEqual(float(live_fills[0]["size"]), 2.0)
        self.assertAlmostEqual(app._paper_result_tracker.unsettled_notional(), 1.0)

        app._reconcile_live_open_orders(now=now, force=True)
        with config.logging.execution_csv_path.open("r", encoding="utf-8", newline="") as fp:
            rows = list(csv.DictReader(fp))
        live_fills = [row for row in rows if row.get("status") == "live_fill" and row.get("order_id") == "live-100"]
        self.assertEqual(len(live_fills), 1)

        adapter.order_payloads["live-100"] = {
            "order": {
                "id": "live-100",
                "status": "LIVE",
                "size_matched": "3.0",
                "price": "0.50",
            }
        }
        app._reconcile_live_open_orders(now=now, force=True)
        with config.logging.execution_csv_path.open("r", encoding="utf-8", newline="") as fp:
            rows = list(csv.DictReader(fp))
        live_fills = [row for row in rows if row.get("status") == "live_fill" and row.get("order_id") == "live-100"]
        self.assertEqual(len(live_fills), 2)
        self.assertAlmostEqual(float(live_fills[-1]["size"]), 3.0)
        self.assertAlmostEqual(app._paper_result_tracker.unsettled_notional(), 1.5)

    def test_reconcile_live_open_orders_retries_order_poll_then_latches_kill_switch(self) -> None:
        config = self._build_config(mode=RuntimeMode.LIVE)
        app = PM1HEdgeTraderApp(config)
        now = datetime(2026, 2, 21, 9, 30, tzinfo=timezone.utc)
        adapter = _FakeLiveOpsAdapter(now=now)
        engine = LimitOrderExecutionEngine(
            adapter=adapter,
            config=ExecutionConfig(entry_block_window_s=60.0),
            now_fn=lambda: now,
        )
        app._execution_engine = engine
        adapter.open_order_ids = {"live-100"}
        engine._active_intents[("mkt-1", "tok-up")] = ActiveIntent(
            order_id="live-100",
            market_id="mkt-1",
            token_id="tok-up",
            side=OutcomeSide.UP,
            price=0.50,
            size=10.0,
            edge=0.03,
            created_at=now,
            last_quoted_at=now,
            status=OrderStatus.OPEN,
        )
        adapter.order_payloads["live-100"] = RuntimeError("401 Unauthorized")

        app._reconcile_live_open_orders(now=now, force=True)

        self.assertTrue(app._external_kill_switch_latched)
        self.assertTrue(engine.kill_switch.active)
        self.assertEqual(adapter.cancel_all_calls, 1)
        self.assertGreaterEqual(len(adapter.get_order_calls), 3)

    def test_heartbeat_failure_latches_emergency_and_cancels_all_live_orders(self) -> None:
        config = self._build_config(mode=RuntimeMode.LIVE)
        app = PM1HEdgeTraderApp(config)
        now = datetime(2026, 2, 21, 8, 0, tzinfo=timezone.utc)
        adapter = _FakeLiveOpsAdapter(now=now)
        adapter.heartbeat_ok = False
        adapter.open_order_ids = {"ghost-1"}
        engine = LimitOrderExecutionEngine(
            adapter=adapter,
            config=ExecutionConfig(entry_block_window_s=60.0),
            now_fn=lambda: now,
        )
        app._execution_engine = engine

        app._run_live_heartbeat(now=now, force=True)

        self.assertEqual(adapter.heartbeat_calls, 1)
        self.assertEqual(adapter.cancel_all_calls, 1)
        self.assertTrue(app._external_kill_switch_latched)
        self.assertTrue(engine.kill_switch.active)

    def test_live_drawdown_guard_latches_kill_switch(self) -> None:
        config = self._build_config(mode=RuntimeMode.LIVE, max_live_drawdown=5.0)
        app = PM1HEdgeTraderApp(config)
        now = datetime(2026, 2, 21, 8, 5, tzinfo=timezone.utc)
        adapter = _FakeLiveOpsAdapter(now=now)
        engine = LimitOrderExecutionEngine(
            adapter=adapter,
            config=ExecutionConfig(entry_block_window_s=60.0),
            now_fn=lambda: now,
        )
        app._execution_engine = engine
        app._live_balance_monitor = SimpleNamespace(
            maybe_refresh=lambda *, now: SimpleNamespace(
                triggered=True,
                drawdown=6.0,
                baseline=SimpleNamespace(bankroll=100.0),
                snapshot=SimpleNamespace(bankroll=94.0),
            )
        )

        app._run_live_drawdown_guard(now=now)

        self.assertTrue(app._external_kill_switch_latched)
        self.assertTrue(engine.kill_switch.active)
        self.assertEqual(adapter.cancel_all_calls, 1)

    def test_complete_set_one_leg_timeout_attempts_recovery_before_kill(self) -> None:
        config = self._build_config(mode=RuntimeMode.LIVE, enable_complete_set_arb=True)
        app = PM1HEdgeTraderApp(config)
        now = datetime(2026, 2, 21, 8, 0, tzinfo=timezone.utc)
        adapter = _FakeLiveOpsAdapter(now=now)
        engine = LimitOrderExecutionEngine(
            adapter=adapter,
            config=ExecutionConfig(entry_block_window_s=60.0),
            now_fn=lambda: now,
        )
        app._execution_engine = engine
        app._token_tick_sizes = {"tok-up": 0.01, "tok-down": 0.01}
        app._arb_one_leg_started_at = now - timedelta(seconds=120)
        app._apply_live_market_side_exposure(market_id="mkt-1", side=OutcomeSide.UP, delta_tokens=3.0)
        engine._active_intents[("mkt-1", "tok-up")] = ActiveIntent(
            order_id="live-100",
            market_id="mkt-1",
            token_id="tok-up",
            side=OutcomeSide.UP,
            price=0.60,
            size=3.0,
            edge=0.03,
            created_at=now,
            last_quoted_at=now,
            status=OrderStatus.OPEN,
        )
        adapter.open_order_ids = {"live-100"}
        up_quote = BestQuote(
            token_id="tok-up",
            best_bid=0.59,
            best_ask=0.60,
            updated_at=now,
            source="test",
        )
        down_quote = BestQuote(
            token_id="tok-down",
            best_bid=0.40,
            best_ask=0.41,
            updated_at=now,
            source="test",
        )

        app._guard_complete_set_partial_state(
            now=now,
            market_id="mkt-1",
            seconds_to_expiry=600.0,
            up_quote=up_quote,
            down_quote=down_quote,
            safety=SimpleNamespace(
                fee_rate=0.0,
                data_age_s=0.0,
                clock_drift_s=0.0,
                book_is_consistent=True,
            ),
        )

        self.assertFalse(app._external_kill_switch_latched)
        self.assertGreaterEqual(len(adapter.place_requests), 1)
        self.assertEqual(adapter.place_requests[-1].token_id, "tok-down")
        self.assertEqual(adapter.place_requests[-1].order_side.value, "BUY")

    def test_reconcile_live_positions_adopt_mode_blocks_without_kill(self) -> None:
        config = self._build_config(
            mode=RuntimeMode.LIVE,
            adopt_existing_positions=True,
            adopt_existing_positions_policy="block",
            position_mismatch_policy="kill",
        )
        app = PM1HEdgeTraderApp(config)
        now = datetime(2026, 2, 26, 1, 0, tzinfo=timezone.utc)
        app._market = SimpleNamespace(
            market_id="mkt-1",
            token_ids=SimpleNamespace(up_token_id="tok-up", down_token_id="tok-down"),
        )

        class _PositionsClient:
            async def fetch_positions(self, *, user_address: str, size_threshold: float):  # type: ignore[no-untyped-def]
                return [{"asset": "tok-up", "size": "1.0"}]

        app._positions_client = _PositionsClient()
        asyncio.run(app._reconcile_live_positions(now=now, force=True))

        self.assertTrue(app._position_mismatch_blocked)
        self.assertFalse(app._external_kill_switch_latched)
        self.assertEqual(app._entry_block_reason(now=now, market_id="mkt-1"), "position_mismatch")

    def test_near_expiry_cleanup_cancels_existing_intents_once(self) -> None:
        config = self._build_config(mode=RuntimeMode.DRY_RUN)
        app = PM1HEdgeTraderApp(config)
        now = datetime(2026, 2, 21, 8, 0, tzinfo=timezone.utc)
        adapter = app._execution_engine._adapter
        assert isinstance(adapter, DryRunExecutionAdapter)

        handle = adapter.place_limit_order(
            OrderRequest(
                market_id="mkt-1",
                token_id="tok-up",
                side=OutcomeSide.UP,
                price=0.55,
                size=10.0,
                submitted_at=now,
            )
        )
        app._execution_engine._active_intents[("mkt-1", "tok-up")] = ActiveIntent(
            order_id=handle.order_id,
            market_id="mkt-1",
            token_id="tok-up",
            side=OutcomeSide.UP,
            price=0.55,
            size=10.0,
            edge=0.02,
            created_at=now,
            last_quoted_at=now,
            status=OrderStatus.OPEN,
        )

        cleaned = app._enforce_near_expiry_cleanup(
            market_id="mkt-1",
            seconds_to_expiry=30.0,
            now=now,
        )

        self.assertTrue(cleaned)
        self.assertEqual(app._execution_engine.active_intents(), {})


if __name__ == "__main__":
    unittest.main()
