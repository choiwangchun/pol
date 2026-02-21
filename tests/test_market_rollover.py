from __future__ import annotations

import asyncio
import sys
import types
import unittest
from pathlib import Path
from types import SimpleNamespace
from tempfile import TemporaryDirectory

ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from pm1h_edge_trader.config import RuntimeMode, build_config  # noqa: E402
from pm1h_edge_trader.main import PM1HEdgeTraderApp  # noqa: E402


class MarketRolloverTests(unittest.IsolatedAsyncioTestCase):
    async def test_run_reactivates_market_after_expiry(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            config = build_config(
                mode=RuntimeMode.DRY_RUN,
                binance_symbol="BTCUSDT",
                market_slug=None,
                tick_seconds=0.01,
                max_ticks=2,
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
            events: list[str] = []

            async def fake_bootstrap(self):  # type: ignore[no-untyped-def]
                events.append("bootstrap")

            async def fake_activate(self):  # type: ignore[no-untyped-def]
                events.append("activate")
                self._market = SimpleNamespace(market_id="m-1", slug="slug-1")
                return True

            async def fake_deactivate(self):  # type: ignore[no-untyped-def]
                events.append("deactivate")
                self._market = None

            async def fake_tick_once(self, *, tick_count: int):  # type: ignore[no-untyped-def]
                events.append(f"tick{tick_count}")
                return tick_count == 1

            async def fake_reconcile(self, *, force=False, now=None):  # type: ignore[no-untyped-def]
                return None

            app._bootstrap_runtime = types.MethodType(fake_bootstrap, app)  # type: ignore[attr-defined]
            app._activate_market = types.MethodType(fake_activate, app)  # type: ignore[attr-defined]
            app._deactivate_market = types.MethodType(fake_deactivate, app)  # type: ignore[attr-defined]
            app._tick_once = types.MethodType(fake_tick_once, app)
            app._reconcile_paper_settlements = types.MethodType(fake_reconcile, app)

            await app.run(asyncio.Event())

            self.assertEqual(
                events,
                ["bootstrap", "activate", "tick1", "deactivate", "activate", "tick2", "deactivate"],
            )


if __name__ == "__main__":
    unittest.main()
