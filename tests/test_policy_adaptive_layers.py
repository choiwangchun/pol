from __future__ import annotations

import sys
import unittest
from datetime import datetime, timezone
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from pm1h_edge_trader.policy.calibration import (  # noqa: E402
    OnlineLogisticCalibrator,
    build_calibrator_features,
)
from pm1h_edge_trader.policy.cost import OnlineCostEstimator  # noqa: E402


class AdaptivePolicyLayerTests(unittest.TestCase):
    def test_calibrator_update_moves_probability_toward_label(self) -> None:
        calibrator = OnlineLogisticCalibrator(learning_rate=1e-2, l2=0.0, max_weight_norm=10.0)
        features = build_calibrator_features(
            q_model_up=0.55,
            spread=0.03,
            rv_long=0.4,
            rv_short=0.6,
            tte_seconds=1800.0,
        )
        before = calibrator.predict(features)

        for _ in range(100):
            calibrator.update(features=features, label_up=1.0, freeze=False)
        after = calibrator.predict(features)

        self.assertGreater(after, before)

    def test_calibrator_freeze_prevents_parameter_updates(self) -> None:
        calibrator = OnlineLogisticCalibrator(learning_rate=1e-2, l2=0.0, max_weight_norm=10.0)
        features = build_calibrator_features(
            q_model_up=0.45,
            spread=0.02,
            rv_long=0.5,
            rv_short=0.4,
            tte_seconds=2700.0,
        )
        state_before = calibrator.export_state()

        calibrator.update(features=features, label_up=1.0, freeze=True)
        state_after = calibrator.export_state()

        self.assertEqual(state_before, state_after)

    def test_cost_estimator_ema_updates_and_freeze(self) -> None:
        estimator = OnlineCostEstimator(alpha=0.5, decay=0.999, min_cost=0.0, max_cost=0.2)
        now = datetime(2026, 2, 26, 9, 0, tzinfo=timezone.utc)
        estimator.update(
            decision_price=0.50,
            fill_price=0.52,
            spread=0.02,
            matched_size=10.0,
            ts=now,
            freeze=False,
        )
        updated = estimator.get()
        self.assertGreater(updated, 0.0)

        frozen_before = estimator.export_state()
        estimator.update(
            decision_price=0.50,
            fill_price=0.60,
            spread=0.10,
            matched_size=20.0,
            ts=now,
            freeze=True,
        )
        frozen_after = estimator.export_state()
        self.assertEqual(frozen_before, frozen_after)


if __name__ == "__main__":
    unittest.main()
