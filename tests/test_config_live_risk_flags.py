from __future__ import annotations

import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from pm1h_edge_trader.config import RuntimeMode, build_config  # noqa: E402
from pm1h_edge_trader.main import build_arg_parser  # noqa: E402


class LiveRiskFlagsConfigTests(unittest.TestCase):
    def test_arg_parser_accepts_live_drawdown_flags(self) -> None:
        parser = build_arg_parser()
        args = parser.parse_args(
            [
                "--max-live-drawdown",
                "15",
                "--live-balance-refresh-seconds",
                "12",
            ]
        )
        self.assertAlmostEqual(args.max_live_drawdown, 15.0)
        self.assertAlmostEqual(args.live_balance_refresh_seconds, 12.0)

    def test_build_config_maps_live_drawdown_fields(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            config = build_config(
                mode=RuntimeMode.LIVE,
                binance_symbol="BTCUSDT",
                market_slug=None,
                tick_seconds=1.0,
                max_ticks=0,
                bankroll=500.0,
                edge_min=0.015,
                edge_buffer=0.01,
                kelly_fraction=0.25,
                f_cap=0.05,
                min_order_notional=1.0,
                rv_fallback=0.55,
                sigma_weight=1.0,
                iv_override=None,
                log_dir=Path(tmp_dir),
                enable_websocket=False,
                max_live_drawdown=25.0,
                live_balance_refresh_seconds=20.0,
            )
        self.assertAlmostEqual(config.risk.max_live_drawdown, 25.0)
        self.assertAlmostEqual(config.risk.live_balance_refresh_seconds, 20.0)


if __name__ == "__main__":
    unittest.main()
