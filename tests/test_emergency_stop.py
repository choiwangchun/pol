from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from pm1h_edge_trader.config import PolymarketLiveAuthConfig  # noqa: E402
from pm1h_edge_trader.emergency_stop import perform_emergency_stop  # noqa: E402


class _FakeAdapter:
    def __init__(self, *, open_before: set[str], open_after: set[str], fail_list_times: int = 0) -> None:
        self._open_before = set(open_before)
        self._open_after = set(open_after)
        self._list_calls = 0
        self._cancel_all_calls = 0
        self._fail_list_times = max(0, fail_list_times)

    def list_open_order_ids(self) -> set[str]:
        self._list_calls += 1
        if self._list_calls <= self._fail_list_times:
            raise RuntimeError("temporary failure")
        if self._cancel_all_calls <= 0:
            return set(self._open_before)
        return set(self._open_after)

    def cancel_all_orders(self) -> int:
        self._cancel_all_calls += 1
        return len(self._open_before)

    @property
    def list_calls(self) -> int:
        return self._list_calls

    @property
    def cancel_all_calls(self) -> int:
        return self._cancel_all_calls


class EmergencyStopTests(unittest.TestCase):
    def test_perform_emergency_stop_cancels_and_verifies_zero_open_orders(self) -> None:
        auth = PolymarketLiveAuthConfig(private_key="0xabc", funder="0xdef")
        adapter = _FakeAdapter(open_before={"o1", "o2"}, open_after=set())

        result = perform_emergency_stop(
            auth=auth,
            host="https://clob.polymarket.com",
            retries=2,
            adapter_factory=lambda **_: adapter,
        )

        self.assertTrue(result["ok"])
        self.assertEqual(result["open_orders_before"], 2)
        self.assertEqual(result["open_orders_after"], 0)
        self.assertEqual(result["cancel_all_reported"], 2)
        self.assertEqual(adapter.cancel_all_calls, 1)

    def test_perform_emergency_stop_reports_residual_orders(self) -> None:
        auth = PolymarketLiveAuthConfig(private_key="0xabc", funder="0xdef")
        adapter = _FakeAdapter(open_before={"o1", "o2"}, open_after={"o2"})

        result = perform_emergency_stop(
            auth=auth,
            host="https://clob.polymarket.com",
            retries=2,
            adapter_factory=lambda **_: adapter,
        )

        self.assertFalse(result["ok"])
        self.assertEqual(result["open_orders_after"], 1)
        self.assertEqual(result["residual_order_ids"], ["o2"])

    def test_perform_emergency_stop_retries_open_order_listing(self) -> None:
        auth = PolymarketLiveAuthConfig(private_key="0xabc", funder="0xdef")
        adapter = _FakeAdapter(open_before={"o1"}, open_after=set(), fail_list_times=1)

        result = perform_emergency_stop(
            auth=auth,
            host="https://clob.polymarket.com",
            retries=3,
            adapter_factory=lambda **_: adapter,
        )

        self.assertTrue(result["ok"])
        self.assertGreaterEqual(adapter.list_calls, 3)

    def test_result_payload_is_json_serializable(self) -> None:
        auth = PolymarketLiveAuthConfig(private_key="0xabc", funder="0xdef")
        adapter = _FakeAdapter(open_before={"o1"}, open_after=set())
        result = perform_emergency_stop(
            auth=auth,
            host="https://clob.polymarket.com",
            retries=1,
            adapter_factory=lambda **_: adapter,
        )
        json.dumps(result, ensure_ascii=True)


if __name__ == "__main__":
    unittest.main()
