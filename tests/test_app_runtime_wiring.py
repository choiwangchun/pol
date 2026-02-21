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

from pm1h_edge_trader.config import (  # noqa: E402
    PolymarketLiveAuthConfig,
    RuntimeMode,
    build_config,
)
from pm1h_edge_trader.execution import DryRunExecutionAdapter  # noqa: E402
from pm1h_edge_trader.execution.polymarket_live_adapter import (  # noqa: E402
    PolymarketLiveExecutionAdapter,
)
from pm1h_edge_trader.feeds.models import BestQuote  # noqa: E402
from pm1h_edge_trader.main import PM1HEdgeTraderApp  # noqa: E402


class AppRuntimeWiringTests(unittest.TestCase):
    def _build_config(self, *, mode: RuntimeMode):
        with TemporaryDirectory() as tmp_dir:
            return build_config(
                mode=mode,
                binance_symbol="BTCUSDT",
                market_slug=None,
                tick_seconds=1.0,
                max_ticks=1,
                bankroll=10_000.0,
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


if __name__ == "__main__":
    unittest.main()
