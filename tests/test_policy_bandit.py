from __future__ import annotations

import csv
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
    build_context_id,
)


class _StubRandom:
    def __init__(self, draw: float = 0.0) -> None:
        self._draw = draw

    def random(self) -> float:
        return self._draw

    def choice(self, values):  # type: ignore[no-untyped-def]
        return values[-1]


class PolicyBanditTests(unittest.TestCase):
    def _build_controller(
        self,
        *,
        tmp_dir: str,
        enabled: bool = True,
        shadow_mode: bool = False,
        epsilon: float = 0.0,
        random_draw: float = 1.0,
    ) -> PolicyBanditController:
        return PolicyBanditController(
            config=PolicyBanditConfig(
                enabled=enabled,
                shadow_mode=shadow_mode,
                exploration_epsilon=epsilon,
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
            random_source=_StubRandom(draw=random_draw),
        )

    def _fill_record(self, *, order_id: str = "dry-1") -> ExecutionLogRecord:
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

    def _settle_record(self, *, order_id: str = "dry-1", pnl: float = 4.0) -> ExecutionLogRecord:
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
            pnl=pnl,
        )

    def test_build_context_id_buckets_time_vol_and_spread(self) -> None:
        context = build_context_id(
            seconds_to_expiry=600.0,
            rv_long=0.4,
            rv_short=0.6,
            max_spread=0.04,
            vol_ratio_threshold=1.1,
            spread_tight_threshold=0.03,
        )
        self.assertEqual(context, "tte_lt_15m|vol_high|spread_wide")

    def test_disabled_controller_keeps_baseline_profile(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            controller = self._build_controller(tmp_dir=tmp_dir, enabled=False)
            selection = controller.select_profile(
                timestamp=datetime(2026, 2, 23, 9, 0, tzinfo=timezone.utc),
                market_id="mkt-1",
                seconds_to_expiry=3600.0,
                rv_long=0.4,
                rv_short=0.5,
                sigma=0.45,
                iv=0.5,
                spot=100000.0,
                strike=99900.0,
                bid_up=0.49,
                ask_up=0.50,
                bid_down=0.49,
                ask_down=0.50,
                bankroll_at_entry=1000.0,
                allow_exploration=True,
            )
            self.assertEqual(selection.chosen_profile_id, "P2_BALANCED")
            self.assertEqual(selection.applied_profile_id, "P2_BALANCED")
            self.assertFalse(selection.exploration)

    def test_shadow_mode_applies_balanced_profile_even_when_choice_differs(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            controller = self._build_controller(
                tmp_dir=tmp_dir,
                enabled=True,
                shadow_mode=True,
                epsilon=1.0,
                random_draw=0.0,
            )
            selection = controller.select_profile(
                timestamp=datetime(2026, 2, 23, 9, 0, tzinfo=timezone.utc),
                market_id="mkt-1",
                seconds_to_expiry=3600.0,
                rv_long=0.4,
                rv_short=0.5,
                sigma=0.45,
                iv=0.5,
                spot=100000.0,
                strike=99900.0,
                bid_up=0.49,
                ask_up=0.50,
                bid_down=0.49,
                ask_down=0.50,
                bankroll_at_entry=1000.0,
                allow_exploration=True,
            )
            self.assertNotEqual(selection.chosen_profile_id, "")
            self.assertEqual(selection.applied_profile_id, "P2_BALANCED")

    def test_settlement_updates_bandit_and_writes_dataset(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            controller = self._build_controller(tmp_dir=tmp_dir, enabled=True, shadow_mode=False)
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
                allow_exploration=True,
            )
            fill_record = self._fill_record(order_id="dry-11")
            controller.on_fill(fill_record, selection=selection)

            reward = controller.on_settlement(self._settle_record(order_id="dry-11", pnl=10.0))

            self.assertIsNotNone(reward)
            self.assertGreater(reward or 0.0, 0.0)

            state = controller.export_state()
            context_bucket = selection.context_id
            profile_bucket = selection.applied_profile_id
            self.assertEqual(
                state["contexts"][context_bucket][profile_bucket]["count"],
                1,
            )

            with (Path(tmp_dir) / "policy_dataset.csv").open("r", newline="", encoding="utf-8") as fp:
                rows = list(csv.DictReader(fp))
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["order_id"], "dry-11")
            self.assertEqual(rows[0]["profile_id"], selection.applied_profile_id)


if __name__ == "__main__":
    unittest.main()
