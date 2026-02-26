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

from pm1h_edge_trader.config import RuntimeMode, build_config  # noqa: E402
from pm1h_edge_trader.main import PM1HEdgeTraderApp  # noqa: E402


class AppResumeStateTests(unittest.IsolatedAsyncioTestCase):
    async def test_bootstrap_restores_kill_latch_when_manual_unlatch_required(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            runtime_path = Path(tmp_dir) / "state" / "runtime_state.json"
            runtime_path.parent.mkdir(parents=True, exist_ok=True)
            runtime_path.write_text(
                json.dumps(
                    {
                        "state_version": 1,
                        "kill_switch_latched": True,
                        "kill_reason": "order_reconciliation_failed",
                        "order_last_matched_size": {"live-11": 1.25},
                    },
                    ensure_ascii=True,
                ),
                encoding="utf-8",
            )
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
                resume=True,
                state_dir=Path(tmp_dir) / "state",
                require_manual_unlatch=True,
                manual_unlatch=False,
            )
            app = PM1HEdgeTraderApp(config)

            await app._bootstrap_runtime()

            self.assertTrue(app._external_kill_switch_latched)
            self.assertEqual(app._live_order_matched_sizes["live-11"], 1.25)
            reason = app._entry_block_reason(
                now=datetime(2026, 2, 26, 9, 0, tzinfo=timezone.utc),
                market_id="mkt-1",
            )
            self.assertEqual(reason, "external_kill_switch_latched")

    async def test_manual_unlatch_clears_restored_latch(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            runtime_path = Path(tmp_dir) / "state" / "runtime_state.json"
            runtime_path.parent.mkdir(parents=True, exist_ok=True)
            runtime_path.write_text(
                json.dumps(
                    {
                        "state_version": 1,
                        "kill_switch_latched": True,
                        "kill_reason": "order_reconciliation_failed",
                    },
                    ensure_ascii=True,
                ),
                encoding="utf-8",
            )
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
                resume=True,
                state_dir=Path(tmp_dir) / "state",
                require_manual_unlatch=True,
                manual_unlatch=True,
            )
            app = PM1HEdgeTraderApp(config)

            await app._bootstrap_runtime()

            self.assertFalse(app._external_kill_switch_latched)


if __name__ == "__main__":
    unittest.main()
