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


class ResumePolicyFlagTests(unittest.TestCase):
    def test_arg_parser_accepts_resume_and_policy_mode_flags(self) -> None:
        parser = build_arg_parser()
        args = parser.parse_args(
            [
                "--no-resume",
                "--state-dir",
                "state-alt",
                "--require-manual-unlatch",
                "--manual-unlatch",
                "--policy-mode",
                "apply_calibrator_cost",
            ]
        )
        self.assertFalse(args.resume)
        self.assertEqual(args.state_dir, Path("state-alt"))
        self.assertTrue(args.require_manual_unlatch)
        self.assertTrue(args.manual_unlatch)
        self.assertEqual(args.policy_mode, "apply_calibrator_cost")

    def test_build_config_maps_runtime_resume_flags(self) -> None:
        with TemporaryDirectory() as tmp_dir:
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
                resume=False,
                state_dir=Path(tmp_dir) / "state-dir",
                require_manual_unlatch=True,
                manual_unlatch=True,
                policy_mode="full",
            )

            self.assertFalse(config.runtime_state.resume)
            self.assertEqual(config.runtime_state.state_dir, Path(tmp_dir) / "state-dir")
            self.assertTrue(config.runtime_state.require_manual_unlatch)
            self.assertTrue(config.runtime_state.manual_unlatch)
            self.assertEqual(config.policy.mode, "full")


if __name__ == "__main__":
    unittest.main()
