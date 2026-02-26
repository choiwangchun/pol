from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from pm1h_edge_trader.execution import OutcomeSide, parse_outcome_side  # noqa: E402


class OutcomeSideTests(unittest.TestCase):
    def test_parse_outcome_side_accepts_up_aliases(self) -> None:
        for raw in ("up", "UP", "u", "yes", "y"):
            with self.subTest(raw=raw):
                self.assertEqual(parse_outcome_side(raw), OutcomeSide.UP)

    def test_parse_outcome_side_accepts_down_aliases(self) -> None:
        for raw in ("down", "DOWN", "d", "no", "n"):
            with self.subTest(raw=raw):
                self.assertEqual(parse_outcome_side(raw), OutcomeSide.DOWN)

    def test_parse_outcome_side_returns_none_for_unknown_value(self) -> None:
        self.assertIsNone(parse_outcome_side("buy"))


if __name__ == "__main__":
    unittest.main()
