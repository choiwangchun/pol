from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from pm1h_edge_trader.positions.reconciler import (  # noqa: E402
    evaluate_position_mismatch,
    extract_position_snapshot,
)


class PositionReconcilerTests(unittest.TestCase):
    def test_extract_position_snapshot_reads_up_down_sizes(self) -> None:
        payload = [
            {
                "asset": "tok-up",
                "size": "2.5",
                "currentValue": "1.25",
            },
            {
                "asset": "tok-down",
                "size": "1.0",
                "currentValue": "0.40",
            },
        ]

        snapshot = extract_position_snapshot(
            payload,
            up_token_id="tok-up",
            down_token_id="tok-down",
            size_threshold=0.0001,
        )

        self.assertAlmostEqual(snapshot.up_size, 2.5)
        self.assertAlmostEqual(snapshot.down_size, 1.0)
        self.assertAlmostEqual(snapshot.up_value, 1.25)
        self.assertAlmostEqual(snapshot.down_value, 0.40)
        self.assertTrue(snapshot.has_any_position_above_threshold)

    def test_extract_position_snapshot_ignores_small_sizes(self) -> None:
        payload = [
            {"asset": "tok-up", "size": "0.00001"},
            {"asset": "tok-down", "size": "0.00002"},
        ]
        snapshot = extract_position_snapshot(
            payload,
            up_token_id="tok-up",
            down_token_id="tok-down",
            size_threshold=0.001,
        )
        self.assertFalse(snapshot.has_any_position_above_threshold)

    def test_evaluate_position_mismatch_true_when_wallet_only_exposure_exists(self) -> None:
        payload = [
            {"asset": "tok-up", "size": "1.2"},
        ]
        snapshot = extract_position_snapshot(
            payload,
            up_token_id="tok-up",
            down_token_id="tok-down",
            size_threshold=0.01,
        )
        result = evaluate_position_mismatch(
            snapshot=snapshot,
            local_up_size=0.0,
            local_down_size=0.0,
            size_threshold=0.01,
            relative_tolerance=0.05,
        )
        self.assertTrue(result.mismatch)
        self.assertEqual(result.reason, "wallet_only_exposure")

    def test_evaluate_position_mismatch_true_when_local_only_exposure_exists(self) -> None:
        payload: list[dict[str, str]] = []
        snapshot = extract_position_snapshot(
            payload,
            up_token_id="tok-up",
            down_token_id="tok-down",
            size_threshold=0.01,
        )
        result = evaluate_position_mismatch(
            snapshot=snapshot,
            local_up_size=0.5,
            local_down_size=0.0,
            size_threshold=0.01,
            relative_tolerance=0.05,
        )
        self.assertTrue(result.mismatch)
        self.assertEqual(result.reason, "local_only_exposure")

    def test_evaluate_position_mismatch_true_when_size_diverges(self) -> None:
        payload = [
            {"asset": "tok-up", "size": "1.2"},
        ]
        snapshot = extract_position_snapshot(
            payload,
            up_token_id="tok-up",
            down_token_id="tok-down",
            size_threshold=0.01,
        )
        result = evaluate_position_mismatch(
            snapshot=snapshot,
            local_up_size=0.8,
            local_down_size=0.0,
            size_threshold=0.01,
            relative_tolerance=0.05,
        )
        self.assertTrue(result.mismatch)
        self.assertEqual(result.reason, "size_divergence")

    def test_evaluate_position_mismatch_false_within_tolerance(self) -> None:
        payload = [
            {"asset": "tok-up", "size": "1.2"},
        ]
        snapshot = extract_position_snapshot(
            payload,
            up_token_id="tok-up",
            down_token_id="tok-down",
            size_threshold=0.01,
        )
        result = evaluate_position_mismatch(
            snapshot=snapshot,
            local_up_size=1.18,
            local_down_size=0.0,
            size_threshold=0.01,
            relative_tolerance=0.05,
        )
        self.assertFalse(result.mismatch)
        self.assertEqual(result.reason, "within_tolerance")


if __name__ == "__main__":
    unittest.main()
