from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from pm1h_edge_trader.live_balance import fetch_live_balance_snapshot  # noqa: E402


class _FakeAdapter:
    def __init__(self, *, host: str, auth) -> None:  # type: ignore[no-untyped-def]
        self.host = host
        self.auth = auth
        self.called = 0
        self.response = (100.0, 80.0)

    def get_collateral_balance_allowance(self, *, refresh: bool = True):
        self.called += 1
        return self.response


class _Auth:
    pass


class LiveBalanceTests(unittest.TestCase):
    def test_fetch_uses_min_of_balance_and_allowance(self) -> None:
        adapter_holder = {}

        def _factory(*, host: str, auth):  # type: ignore[no-untyped-def]
            adapter = _FakeAdapter(host=host, auth=auth)
            adapter.response = (321.0, 120.5)
            adapter_holder["adapter"] = adapter
            return adapter

        snapshot = fetch_live_balance_snapshot(
            auth=_Auth(),
            host="https://clob.polymarket.com",
            adapter_factory=_factory,
        )

        self.assertEqual(snapshot.balance, 321.0)
        self.assertEqual(snapshot.allowance, 120.5)
        self.assertEqual(snapshot.bankroll, 120.5)
        self.assertEqual(adapter_holder["adapter"].called, 1)

    def test_fetch_falls_back_to_balance_when_allowance_missing(self) -> None:
        def _factory(*, host: str, auth):  # type: ignore[no-untyped-def]
            adapter = _FakeAdapter(host=host, auth=auth)
            adapter.response = (45.0, None)
            return adapter

        snapshot = fetch_live_balance_snapshot(
            auth=_Auth(),
            host="https://clob.polymarket.com",
            adapter_factory=_factory,
        )

        self.assertEqual(snapshot.bankroll, 45.0)

    def test_fetch_raises_when_no_positive_balance_available(self) -> None:
        def _factory(*, host: str, auth):  # type: ignore[no-untyped-def]
            adapter = _FakeAdapter(host=host, auth=auth)
            adapter.response = (None, 0.0)
            return adapter

        with self.assertRaises(RuntimeError):
            fetch_live_balance_snapshot(
                auth=_Auth(),
                host="https://clob.polymarket.com",
                adapter_factory=_factory,
            )


if __name__ == "__main__":
    unittest.main()
