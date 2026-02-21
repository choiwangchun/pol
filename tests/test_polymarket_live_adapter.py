from __future__ import annotations

import sys
import unittest
from datetime import datetime, timezone
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from pm1h_edge_trader.config import PolymarketLiveAuthConfig  # noqa: E402
from pm1h_edge_trader.execution import OrderRequest, Side  # noqa: E402
from pm1h_edge_trader.execution.polymarket_live_adapter import (  # noqa: E402
    PolymarketLiveExecutionAdapter,
)


class _FakeClient:
    def __init__(self) -> None:
        self.create_or_derive_api_creds_calls = 0
        self.set_api_creds_calls: list[object] = []
        self.create_order_calls: list[object] = []
        self.post_order_calls: list[object] = []
        self.cancel_calls: list[str] = []
        self.post_order_response: object = {"id": "oid-1"}
        self.cancel_response: object = {"canceled": ["oid-1"]}
        self.raise_from_create: Exception | None = None
        self.raise_from_post: Exception | None = None
        self.raise_from_cancel: Exception | None = None

    def create_or_derive_api_creds(self) -> object:
        self.create_or_derive_api_creds_calls += 1
        return object()

    def set_api_creds(self, creds: object) -> None:
        self.set_api_creds_calls.append(creds)

    def create_order(self, order_args) -> object:  # type: ignore[no-untyped-def]
        if self.raise_from_create is not None:
            raise self.raise_from_create
        self.create_order_calls.append(order_args)
        return {"signed": "order"}

    def post_order(self, order) -> object:  # type: ignore[no-untyped-def]
        if self.raise_from_post is not None:
            raise self.raise_from_post
        self.post_order_calls.append(order)
        return self.post_order_response

    def cancel(self, order_id: str) -> object:
        if self.raise_from_cancel is not None:
            raise self.raise_from_cancel
        self.cancel_calls.append(order_id)
        return self.cancel_response


class PolymarketLiveAdapterTests(unittest.TestCase):
    def _adapter(
        self,
        *,
        fake_client: _FakeClient,
        auth: PolymarketLiveAuthConfig | None = None,
    ) -> tuple[PolymarketLiveExecutionAdapter, dict[str, object]]:
        kwargs_holder: dict[str, object] = {}

        def _factory(**kwargs: object) -> object:
            kwargs_holder.update(kwargs)
            return fake_client

        adapter = PolymarketLiveExecutionAdapter(
            host="https://clob.polymarket.com",
            auth=auth
            or PolymarketLiveAuthConfig(
                private_key="private-key-secret",
                funder="0xFunder",
            ),
            client_factory=_factory,
            now_fn=lambda: datetime(2026, 2, 16, tzinfo=timezone.utc),
        )
        return adapter, kwargs_holder

    def _request(self) -> OrderRequest:
        return OrderRequest(
            market_id="market-1",
            token_id="token-down",
            side=Side.SELL,
            price=0.47,
            size=21.0,
            submitted_at=datetime(2026, 2, 16, tzinfo=timezone.utc),
        )

    def test_place_limit_order_posts_buy_on_selected_token(self) -> None:
        fake_client = _FakeClient()
        adapter, kwargs = self._adapter(fake_client=fake_client)

        handle = adapter.place_limit_order(self._request())

        self.assertEqual(kwargs["host"], "https://clob.polymarket.com")
        self.assertEqual(kwargs["chain_id"], 137)
        self.assertEqual(kwargs["signature_type"], 1)
        self.assertEqual(kwargs["funder"], "0xFunder")
        self.assertEqual(kwargs["key"], "private-key-secret")
        self.assertEqual(fake_client.create_or_derive_api_creds_calls, 1)
        self.assertEqual(len(fake_client.create_order_calls), 1)
        order_args = fake_client.create_order_calls[0]
        self.assertEqual(order_args.token_id, "token-down")
        self.assertEqual(order_args.price, 0.47)
        self.assertEqual(order_args.size, 21.0)
        self.assertEqual(order_args.side, 0)
        self.assertEqual(handle.order_id, "oid-1")

    def test_place_limit_order_reads_order_id_variants(self) -> None:
        responses = [
            {"id": "id-1"},
            {"orderID": "id-2"},
            {"order_id": "id-3"},
        ]
        for response in responses:
            with self.subTest(response=response):
                fake_client = _FakeClient()
                fake_client.post_order_response = response
                adapter, _ = self._adapter(fake_client=fake_client)
                handle = adapter.place_limit_order(self._request())
                self.assertEqual(handle.order_id, next(iter(response.values())))

    def test_cancel_order_parses_success_conservatively(self) -> None:
        cases = [
            ({"canceled": ["oid-1"]}, True),
            ({"cancelledOrderIDs": ["oid-1"]}, True),
            ({"success": True}, True),
            ({"status": "ok"}, False),
            ({}, False),
        ]
        for payload, expected in cases:
            with self.subTest(payload=payload):
                fake_client = _FakeClient()
                fake_client.cancel_response = payload
                adapter, _ = self._adapter(fake_client=fake_client)
                result = adapter.cancel_order("oid-1")
                self.assertEqual(result.canceled, expected)

    def test_place_limit_order_wraps_external_exception_without_secret_echo(self) -> None:
        fake_client = _FakeClient()
        fake_client.raise_from_post = RuntimeError("failed with private-key-secret")
        adapter, _ = self._adapter(fake_client=fake_client)

        with self.assertRaises(RuntimeError) as context:
            adapter.place_limit_order(self._request())

        message = str(context.exception)
        self.assertIn("place_limit_order failed", message)
        self.assertNotIn("private-key-secret", message)


if __name__ == "__main__":
    unittest.main()
