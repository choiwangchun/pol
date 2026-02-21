from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from pm1h_edge_trader.main import _should_mark_dry_run_fill  # noqa: E402


class PaperFillLogicTests(unittest.TestCase):
    def test_marks_fill_when_limit_price_crosses_or_matches_best_ask(self) -> None:
        self.assertTrue(_should_mark_dry_run_fill(limit_price=0.55, best_ask=0.55))
        self.assertTrue(_should_mark_dry_run_fill(limit_price=0.56, best_ask=0.55))

    def test_does_not_mark_fill_when_not_marketable(self) -> None:
        self.assertFalse(_should_mark_dry_run_fill(limit_price=0.5499, best_ask=0.55))
        self.assertFalse(_should_mark_dry_run_fill(limit_price=0.55, best_ask=None))


if __name__ == "__main__":
    unittest.main()
