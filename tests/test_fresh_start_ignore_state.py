from __future__ import annotations

import csv
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from pm1h_edge_trader.config import RuntimeMode, build_config  # noqa: E402
from pm1h_edge_trader.main import PM1HEdgeTraderApp, build_arg_parser  # noqa: E402


class _FakeTracker:
    def __init__(self) -> None:
        self.bootstrap_calls: list[Path] = []
        self.write_calls = 0

    def bootstrap_from_csv(self, path: Path) -> None:
        self.bootstrap_calls.append(path)

    def write_snapshot(self) -> None:
        self.write_calls += 1


class FreshStartIgnoreStateTests(unittest.IsolatedAsyncioTestCase):
    def test_cli_parses_fresh_start_ignore_state_flag(self) -> None:
        parser = build_arg_parser()
        args = parser.parse_args(["--fresh-start", "--fresh-start-ignore-state"])
        self.assertTrue(args.fresh_start)
        self.assertTrue(args.fresh_start_ignore_state)

    async def test_bootstrap_skips_checkpoint_load_when_fresh_start_ignore_state_enabled(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            log_dir = Path(tmp_dir) / "logs"
            csv_path = log_dir / "executions.csv"
            csv_path.parent.mkdir(parents=True, exist_ok=True)
            with csv_path.open("w", newline="", encoding="utf-8") as fp:
                writer = csv.DictWriter(fp, fieldnames=["timestamp", "status"])
                writer.writeheader()
                writer.writerow({"timestamp": "2026-02-20T12:00:00+00:00", "status": "paper_fill"})

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
                log_dir=log_dir,
                enable_websocket=False,
                resume=True,
                fresh_start=True,
                fresh_start_ignore_state=True,
            )
            app = PM1HEdgeTraderApp(config)
            fake = _FakeTracker()
            app._paper_result_tracker = fake  # type: ignore[assignment]

            called = {"value": False}

            def _mark_called() -> None:
                called["value"] = True

            app._load_checkpoint_state = _mark_called  # type: ignore[assignment]

            await app._bootstrap_runtime()

            self.assertFalse(called["value"])
            self.assertEqual(len(fake.bootstrap_calls), 0)
            self.assertEqual(fake.write_calls, 1)

            with csv_path.open("r", newline="", encoding="utf-8") as fp:
                rows = list(csv.DictReader(fp))
            self.assertEqual(rows, [])


if __name__ == "__main__":
    unittest.main()
