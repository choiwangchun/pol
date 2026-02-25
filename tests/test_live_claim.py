from __future__ import annotations

import sys
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest import mock

from eth_account import Account

ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from pm1h_edge_trader.config import PolymarketLiveAuthConfig  # noqa: E402
from pm1h_edge_trader.live_claim import build_live_auto_claim_worker  # noqa: E402
from pm1h_edge_trader import live_claim as live_claim_module  # noqa: E402
from pm1h_edge_trader.live_claim import (  # noqa: E402
    ClaimAttempt,
    LiveAutoClaimWorker,
    RedeemTarget,
    extract_redeemable_targets,
)


class _FakePositionSource:
    def __init__(self, positions: list[dict[str, object]]) -> None:
        self.positions = positions
        self.calls = 0
        self.last_user: str | None = None
        self.last_threshold: float | None = None

    def fetch_redeemable_positions(self, *, user_address: str, size_threshold: float) -> list[dict[str, object]]:
        self.calls += 1
        self.last_user = user_address
        self.last_threshold = size_threshold
        return list(self.positions)


class _FakeRedeemExecutor:
    def __init__(self) -> None:
        self.targets: list[RedeemTarget] = []

    def redeem_positions(self, target: RedeemTarget) -> ClaimAttempt:
        self.targets.append(target)
        tx_hash = f"0x{len(self.targets):064x}"
        return ClaimAttempt(
            condition_id=target.condition_id,
            index_sets=target.index_sets,
            success=True,
            tx_hash=tx_hash,
            error=None,
        )


class _FakeAddressAdapter:
    def __init__(self) -> None:
        self.conditional = "0x4D97DCd97eC945f40cF65F87097ACe5EA0476045"
        self.collateral = "0x2791Bca1f2de4661ED88A30C99A7a9449Aa84174"

    def get_conditional_address(self) -> str:
        return self.conditional

    def get_collateral_address(self) -> str:
        return self.collateral


class LiveClaimTests(unittest.TestCase):
    def test_extract_redeemable_targets_groups_condition_and_index_sets(self) -> None:
        positions = [
            {
                "conditionId": "0x" + "11" * 32,
                "redeemable": True,
                "size": 3.25,
                "outcomeIndex": 0,
            },
            {
                "conditionId": "0x" + "11" * 32,
                "redeemable": True,
                "size": 2.75,
                "outcomeIndex": 1,
            },
            {
                "conditionId": "0x" + "22" * 32,
                "redeemable": True,
                "size": 9.5,
                "outcomeIndex": 1,
            },
            {
                "conditionId": "0x" + "33" * 32,
                "redeemable": False,
                "size": 8.0,
                "outcomeIndex": 0,
            },
            {
                "conditionId": "0x" + "44" * 32,
                "redeemable": True,
                "size": 0.0,
                "outcomeIndex": 0,
            },
        ]

        targets = extract_redeemable_targets(positions, size_threshold=0.0001)

        self.assertEqual(
            targets,
            [
                RedeemTarget(condition_id="0x" + "11" * 32, index_sets=(1, 2)),
                RedeemTarget(condition_id="0x" + "22" * 32, index_sets=(2,)),
            ],
        )

    def test_live_auto_claim_worker_applies_per_condition_cooldown(self) -> None:
        now_holder = {"value": datetime(2026, 2, 24, 16, 0, tzinfo=timezone.utc)}

        def _now() -> datetime:
            return now_holder["value"]

        positions = [
            {
                "conditionId": "0x" + "aa" * 32,
                "redeemable": True,
                "size": 1.0,
                "outcomeIndex": 0,
            }
        ]
        source = _FakePositionSource(positions)
        executor = _FakeRedeemExecutor()
        worker = LiveAutoClaimWorker(
            user_address="0x1111111111111111111111111111111111111111",
            position_source=source,
            redeem_executor=executor,
            size_threshold=0.0001,
            cooldown_seconds=60.0,
            now_fn=_now,
        )

        first = worker.run_once()
        self.assertEqual(first.targets, 1)
        self.assertEqual(first.attempted, 1)
        self.assertEqual(first.claimed, 1)
        self.assertEqual(first.skipped_cooldown, 0)

        now_holder["value"] = now_holder["value"] + timedelta(seconds=10)
        second = worker.run_once()
        self.assertEqual(second.targets, 1)
        self.assertEqual(second.attempted, 0)
        self.assertEqual(second.claimed, 0)
        self.assertEqual(second.skipped_cooldown, 1)

        now_holder["value"] = now_holder["value"] + timedelta(seconds=80)
        third = worker.run_once()
        self.assertEqual(third.targets, 1)
        self.assertEqual(third.attempted, 1)
        self.assertEqual(third.claimed, 1)
        self.assertEqual(third.skipped_cooldown, 0)

    def test_build_live_auto_claim_worker_rejects_proxy_wallet_mode(self) -> None:
        auth = PolymarketLiveAuthConfig(
            private_key="0x1111111111111111111111111111111111111111111111111111111111111111",
            funder="0x2222222222222222222222222222222222222222",
            chain_id=137,
        )
        with self.assertRaisesRegex(RuntimeError, "direct-wallet mode"):
            build_live_auto_claim_worker(
                auth=auth,
                adapter=_FakeAddressAdapter(),
                rpc_url="https://polygon-rpc.com",
                data_api_base_url="https://data-api.polymarket.com",
                size_threshold=0.0001,
                cooldown_seconds=600.0,
                tx_timeout_seconds=120.0,
            )

    def test_build_live_auto_claim_worker_builds_when_funder_matches_signer(self) -> None:
        private_key = "0x1111111111111111111111111111111111111111111111111111111111111111"
        signer = Account.from_key(private_key).address
        auth = PolymarketLiveAuthConfig(
            private_key=private_key,
            funder=signer,
            chain_id=137,
        )
        fake_executor = _FakeRedeemExecutor()
        with mock.patch.object(
            live_claim_module,
            "OnChainRedeemExecutor",
            return_value=fake_executor,
        ) as constructor:
            worker = build_live_auto_claim_worker(
                auth=auth,
                adapter=_FakeAddressAdapter(),
                rpc_url="https://polygon-rpc.com",
                data_api_base_url="https://data-api.polymarket.com",
                size_threshold=0.0001,
                cooldown_seconds=600.0,
                tx_timeout_seconds=120.0,
            )
        self.assertIsInstance(worker, LiveAutoClaimWorker)
        constructor.assert_called_once()


if __name__ == "__main__":
    unittest.main()
