from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from datetime import datetime
from typing import Any

from ..config import PolymarketLiveAuthConfig
from .adapters import ExecutionAdapter
from .types import CancelResult, OrderHandle, OrderRequest, OrderStatus, utc_now

try:
    from py_clob_client.client import ClobClient
    from py_clob_client.clob_types import ApiCreds, OrderArgs
except Exception:  # pragma: no cover - optional runtime dependency during import
    ClobClient = None  # type: ignore[assignment]
    ApiCreds = None  # type: ignore[assignment]
    OrderArgs = None  # type: ignore[assignment]

CLOB_SIDE_BUY = 0


class PolymarketLiveExecutionAdapter(ExecutionAdapter):
    """Real Polymarket CLOB adapter for authenticated live orders."""

    def __init__(
        self,
        *,
        host: str,
        auth: PolymarketLiveAuthConfig,
        client_factory: Callable[..., object] | None = None,
        now_fn: Callable[[], datetime] = utc_now,
    ) -> None:
        if client_factory is None:
            if ClobClient is None:
                raise RuntimeError("py-clob-client is required for live mode.")
            client_factory = ClobClient
        self._now_fn = now_fn

        try:
            self._client = client_factory(
                host=host,
                chain_id=auth.chain_id,
                key=auth.private_key,
                signature_type=auth.signature_type,
                funder=auth.funder,
            )
            self._api_creds_ready = False
            self._set_static_api_creds(auth)
        except Exception as exc:
            raise RuntimeError(
                f"Polymarket live adapter initialization failed ({exc.__class__.__name__})."
            ) from exc

    def place_limit_order(self, request: OrderRequest) -> OrderHandle:
        self._ensure_authenticated_api_creds()
        try:
            order_args = OrderArgs(  # type: ignore[misc,operator]
                token_id=request.token_id,
                price=float(request.price),
                size=float(request.size),
                side=CLOB_SIDE_BUY,
            )
            signed_order = self._client.create_order(order_args)
            response = self._client.post_order(signed_order)
        except Exception as exc:
            raise RuntimeError(
                f"Polymarket place_limit_order failed ({exc.__class__.__name__})."
            ) from exc

        order_id = _extract_order_id(response)
        if order_id is None:
            raise RuntimeError("Polymarket place_limit_order failed (missing order id).")

        return OrderHandle(
            order_id=order_id,
            status=OrderStatus.OPEN,
            acknowledged_at=self._now_fn(),
        )

    def cancel_order(self, order_id: str) -> CancelResult:
        self._ensure_authenticated_api_creds()
        try:
            response = self._client.cancel(order_id)
        except Exception as exc:
            raise RuntimeError(
                f"Polymarket cancel_order failed ({exc.__class__.__name__})."
            ) from exc

        return CancelResult(
            order_id=order_id,
            canceled=_parse_cancel_success(response, order_id),
            acknowledged_at=self._now_fn(),
        )

    def _set_static_api_creds(self, auth: PolymarketLiveAuthConfig) -> None:
        if _has_text(auth.api_key) and _has_text(auth.api_secret) and _has_text(auth.api_passphrase):
            if ApiCreds is None:
                raise RuntimeError("py-clob-client ApiCreds type is unavailable")
            creds = ApiCreds(  # type: ignore[misc,operator]
                api_key=auth.api_key,
                api_secret=auth.api_secret,
                api_passphrase=auth.api_passphrase,
            )
            self._client.set_api_creds(creds)
            self._api_creds_ready = True

    def _ensure_authenticated_api_creds(self) -> None:
        if self._api_creds_ready:
            return
        try:
            creds = self._client.create_or_derive_api_creds()
            if creds is None:
                raise RuntimeError("received empty API credentials")
            self._client.set_api_creds(creds)
        except Exception as exc:
            raise RuntimeError(
                f"Polymarket auth credential setup failed ({exc.__class__.__name__})."
            ) from exc
        self._api_creds_ready = True


def _extract_order_id(payload: object) -> str | None:
    order_id = _extract_first_str_by_keys(payload, ("id", "orderID", "order_id", "orderId"))
    if order_id is None:
        return None
    normalized = order_id.strip()
    return normalized or None


def _extract_first_str_by_keys(payload: object, keys: Sequence[str]) -> str | None:
    if isinstance(payload, Mapping):
        for key in keys:
            candidate = payload.get(key)
            if isinstance(candidate, str):
                return candidate
        for nested_key in ("data", "result", "order"):
            nested = payload.get(nested_key)
            nested_id = _extract_first_str_by_keys(nested, keys)
            if nested_id is not None:
                return nested_id
    elif isinstance(payload, Sequence) and not isinstance(payload, (str, bytes, bytearray)):
        for item in payload:
            nested_id = _extract_first_str_by_keys(item, keys)
            if nested_id is not None:
                return nested_id
    return None


def _parse_cancel_success(payload: object, order_id: str) -> bool:
    if isinstance(payload, bool):
        return payload
    if isinstance(payload, Mapping):
        for key in ("success", "canceled", "cancelled"):
            marker = payload.get(key)
            if isinstance(marker, bool):
                return marker
    return _contains_order_id(payload, order_id)


def _contains_order_id(payload: object, order_id: str) -> bool:
    if isinstance(payload, Mapping):
        for key, value in payload.items():
            if key in {
                "orderID",
                "orderId",
                "order_id",
                "id",
                "canceled",
                "cancelled",
                "orderIDs",
                "order_ids",
                "cancelledOrderIDs",
                "canceledOrderIDs",
            } and _contains_order_id(value, order_id):
                return True
            if _contains_order_id(value, order_id):
                return True
        return False
    if isinstance(payload, Sequence) and not isinstance(payload, (str, bytes, bytearray)):
        return any(_contains_order_id(item, order_id) for item in payload)
    return isinstance(payload, str) and payload.strip() == order_id


def _has_text(value: str | None) -> bool:
    return value is not None and bool(value.strip())
