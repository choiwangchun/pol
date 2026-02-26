from __future__ import annotations

import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from pm1h_edge_trader.runtime.state import RuntimeStateManager  # noqa: E402


class RuntimeStateManagerTests(unittest.TestCase):
    def test_runtime_state_roundtrip_uses_atomic_replace(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            manager = RuntimeStateManager(state_dir=Path(tmp_dir))
            payload = {
                "state_version": 1,
                "kill_switch_latched": True,
                "last_seen_market_id": "mkt-1",
                "order_last_matched_size": {"live-1": 2.5},
            }

            manager.save_runtime(payload)
            loaded = manager.load_runtime()

            self.assertEqual(loaded["kill_switch_latched"], True)
            self.assertEqual(loaded["order_last_matched_size"]["live-1"], 2.5)
            self.assertFalse((Path(tmp_dir) / "runtime_state.json.tmp").exists())

    def test_policy_state_roundtrip_uses_atomic_replace(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            manager = RuntimeStateManager(state_dir=Path(tmp_dir))
            payload = {
                "state_version": 1,
                "freeze_learning": False,
                "calibrator": {"weights": [0.1, 0.2], "bias": 0.0},
                "cost_estimator": {"ema_cost": 0.01},
            }

            manager.save_policy(payload)
            loaded = manager.load_policy()

            self.assertEqual(loaded["calibrator"]["weights"], [0.1, 0.2])
            self.assertAlmostEqual(loaded["cost_estimator"]["ema_cost"], 0.01)
            self.assertFalse((Path(tmp_dir) / "policy_state.json.tmp").exists())


if __name__ == "__main__":
    unittest.main()
