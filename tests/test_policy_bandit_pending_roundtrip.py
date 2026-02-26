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

from pm1h_edge_trader.engine import DecisionConfig  # noqa: E402
from pm1h_edge_trader.logger.models import ExecutionLogRecord  # noqa: E402
from pm1h_edge_trader.policy.bandit import (  # noqa: E402
    PolicyBanditConfig,
    PolicyBanditController,
)


class PolicyBanditPendingRoundtripTests(unittest.TestCase):
    def _build_controller(self, *, tmp_dir: str) -> PolicyBanditController:
        return PolicyBanditController(
            config=PolicyBanditConfig(
                enabled=True,
                shadow_mode=False,
                exploration_epsilon=0.0,
                ucb_c=1.0,
                reward_turnover_lambda=0.1,
                reward_risk_penalty=1.0,
                vol_ratio_threshold=1.1,
                spread_tight_threshold=0.03,
                dataset_path=Path(tmp_dir) / "policy_dataset.csv",
            ),
            base_decision_config=DecisionConfig(
                sigma_weight=1.0,
                edge_min=0.02,
                edge_buffer_up=0.01,
                edge_buffer_down=0.01,
                cost_rate_up=0.005,
                cost_rate_down=0.005,
                kelly_fraction=0.25,
                f_cap=0.05,
                min_order_size=5.0,
            ),
        )

    @staticmethod
    def _fill_record(order_id: str) -> ExecutionLogRecord:
        return ExecutionLogRecord(
            timestamp=datetime(2026, 2, 23, 9, 0, tzinfo=timezone.utc),
            market_id="mkt-1",
            K=100000.0,
            S_t=100100.0,
            tau=0.0001,
            sigma=0.55,
            q_up=0.58,
            bid_up=0.55,
            ask_up=0.56,
            bid_down=0.42,
            ask_down=0.43,
            edge=0.02,
            order_id=order_id,
            side="up",
            price=0.56,
            size=20.0,
            status="paper_fill",
            settlement_outcome=None,
            pnl=None,
        )

    @staticmethod
    def _settle_record(order_id: str) -> ExecutionLogRecord:
        return ExecutionLogRecord(
            timestamp=datetime(2026, 2, 23, 10, 0, tzinfo=timezone.utc),
            market_id="mkt-1",
            K=100000.0,
            S_t=100050.0,
            tau=0.0,
            sigma=0.55,
            q_up=0.0,
            bid_up=0.0,
            ask_up=0.0,
            bid_down=0.0,
            ask_down=0.0,
            edge=0.0,
            order_id=order_id,
            side="up",
            price=0.56,
            size=20.0,
            status="paper_settle",
            settlement_outcome="win",
            pnl=8.0,
        )

    def test_pending_roundtrip_restores_reward_update_path(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            controller = self._build_controller(tmp_dir=tmp_dir)
            selection = controller.select_profile(
                timestamp=datetime(2026, 2, 23, 9, 0, tzinfo=timezone.utc),
                market_id="mkt-1",
                seconds_to_expiry=1800.0,
                rv_long=0.4,
                rv_short=0.6,
                sigma=0.5,
                iv=0.55,
                spot=100000.0,
                strike=100020.0,
                bid_up=0.48,
                ask_up=0.49,
                bid_down=0.50,
                ask_down=0.51,
                bankroll_at_entry=1000.0,
                allow_exploration=False,
            )
            controller.on_fill(self._fill_record(order_id="dry-22"), selection=selection)

            snapshot = controller.export_state()
            self.assertEqual(len(snapshot.get("pending", [])), 1)

            restored = self._build_controller(tmp_dir=tmp_dir)
            restored.import_state(snapshot)
            reward = restored.on_settlement(self._settle_record(order_id="dry-22"))

            self.assertIsNotNone(reward)
            state = restored.export_state()
            context_bucket = selection.context_id
            profile_bucket = selection.applied_profile_id
            self.assertEqual(state["contexts"][context_bucket][profile_bucket]["count"], 1)


if __name__ == "__main__":
    unittest.main()
