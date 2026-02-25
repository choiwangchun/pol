from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from datetime import datetime
from typing import Any

from ..config import PolymarketLiveAuthConfig
from .adapters import ExecutionAdapter
from .types import CancelResult, OrderHandle, OrderRequest, OrderStatus, utc_now

try:
    from py_clob_client.client import ClobClient
    from py_clob_client.clob_types import ApiCreds, BalanceAllowanceParams, OrderArgs, TradeParams
    from py_clob_client.order_builder.constants import BUY as CLOB_BUY
    from py_clob_client.order_builder.constants import SELL as CLOB_SELL
except Exception:  # pragma: no cover - optional runtime dependency during import
    ClobClient = None  # type: ignore[assignment]
    ApiCreds = None  # type: ignore[assignment]
    BalanceAllowanceParams = None  # type: ignore[assignment]
    OrderArgs = None  # type: ignore[assignment]
    TradeParams = None  # type: ignore[assignment]
    CLOB_BUY = "BUY"  # type: ignore[assignment]
    CLOB_SELL = "SELL"  # type: ignore[assignment]

COLLATERAL_DECIMALS = 6
COLLATERAL_SCALE = float(10**COLLATERAL_DECIMALS)


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
            self._heartbeat_id: str | None = None
            self._set_static_api_creds(auth)
        except Exception as exc:
            raise RuntimeError(
                f"Polymarket live adapter initialization failed ({exc.__class__.__name__})."
            ) from exc

    def place_limit_order(self, request: OrderRequest) -> OrderHandle:
        self._ensure_authenticated_api_creds()
        try:
            venue_side = str(getattr(request.order_side, "value", request.order_side or CLOB_BUY)).strip().upper()
            if venue_side not in {"BUY", "SELL"}:
                venue_side = CLOB_BUY
            order_args = OrderArgs(  # type: ignore[misc,operator]
                token_id=request.token_id,
                price=float(request.price),
                size=float(request.size),
                side=CLOB_BUY if venue_side == "BUY" else CLOB_SELL,
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

    def list_open_order_ids(self) -> set[str]:
        self._ensure_authenticated_api_creds()
        try:
            response = self._client.get_orders()
        except Exception as exc:
            raise RuntimeError(
                f"Polymarket list_open_order_ids failed ({exc.__class__.__name__})."
            ) from exc
        return _collect_open_order_ids(response)

    def cancel_all_orders(self) -> int:
        self._ensure_authenticated_api_creds()
        try:
            response = self._client.cancel_all()
        except Exception as exc:
            raise RuntimeError(
                f"Polymarket cancel_all_orders failed ({exc.__class__.__name__})."
            ) from exc
        return _count_canceled_orders(response)

    def send_heartbeat(self, *, heartbeat_id: str | None = None) -> bool:
        self._ensure_authenticated_api_creds()
        requested_id = heartbeat_id.strip() if isinstance(heartbeat_id, str) else None
        current_id = requested_id or self._heartbeat_id
        try:
            response = self._client.post_heartbeat(current_id)
        except Exception as exc:
            recovered = _extract_heartbeat_id_from_exception_error(exc)
            if recovered is None:
                return False
            self._heartbeat_id = recovered
            try:
                response = self._client.post_heartbeat(self._heartbeat_id)
            except Exception:
                return False
        next_id = _extract_heartbeat_id_from_payload(response)
        if next_id is not None:
            self._heartbeat_id = next_id
        return True

    def get_tick_size(self, token_id: str) -> float | None:
        normalized = token_id.strip()
        if not normalized:
            return None
        try:
            value = self._client.get_tick_size(normalized)
        except Exception:
            return None
        parsed = _safe_positive_float(value)
        return parsed

    def get_conditional_address(self) -> str | None:
        try:
            value = self._client.get_conditional_address()
        except Exception:
            return None
        return _normalize_hex_address(value)

    def get_collateral_address(self) -> str | None:
        try:
            value = self._client.get_collateral_address()
        except Exception:
            return None
        return _normalize_hex_address(value)

    def get_collateral_balance_allowance(self, *, refresh: bool = True) -> tuple[float | None, float | None]:
        self._ensure_authenticated_api_creds()
        params = _build_balance_allowance_params()
        try:
            if refresh:
                self._client.update_balance_allowance(params)
        except Exception:
            # Best-effort refresh; continue with read.
            pass
        try:
            payload = self._client.get_balance_allowance(params)
        except Exception as exc:
            raise RuntimeError(
                f"Polymarket get_collateral_balance_allowance failed ({exc.__class__.__name__})."
            ) from exc
        balance = _extract_collateral_by_keys(payload, ("balance", "available", "available_balance"))
        allowance = _extract_collateral_by_keys(payload, ("allowance",))
        if allowance is None:
            allowance = _extract_max_collateral_allowance(payload)
        return balance, allowance

    def get_order(self, order_id: str) -> Mapping[str, Any] | None:
        self._ensure_authenticated_api_creds()
        normalized = order_id.strip()
        if not normalized:
            return None
        try:
            payload = self._client.get_order(normalized)
        except Exception as exc:
            if _is_not_found_error(exc):
                return None
            raise RuntimeError(
                f"Polymarket get_order failed ({exc.__class__.__name__})."
            ) from exc
        return payload if isinstance(payload, Mapping) else None

    def get_trade(self, trade_id: str) -> Mapping[str, Any] | None:
        self._ensure_authenticated_api_creds()
        normalized = trade_id.strip()
        if not normalized or TradeParams is None:
            return None
        try:
            params = TradeParams(id=normalized)  # type: ignore[misc,operator]
            payload = self._client.get_trades(params=params)
        except Exception as exc:
            if _is_not_found_error(exc):
                return None
            raise RuntimeError(
                f"Polymarket get_trade failed ({exc.__class__.__name__})."
            ) from exc
        trade = _find_trade_by_id(payload, normalized)
        if trade is not None:
            return trade
        return payload if isinstance(payload, Mapping) else None

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


def _extract_heartbeat_id_from_payload(payload: object) -> str | None:
    if isinstance(payload, Mapping):
        candidate = payload.get("heartbeat_id")
        if isinstance(candidate, str):
            normalized = candidate.strip()
            if normalized:
                return normalized
        for nested_key in ("data", "result"):
            nested = payload.get(nested_key)
            nested_id = _extract_heartbeat_id_from_payload(nested)
            if nested_id is not None:
                return nested_id
    elif isinstance(payload, Sequence) and not isinstance(payload, (str, bytes, bytearray)):
        for item in payload:
            nested_id = _extract_heartbeat_id_from_payload(item)
            if nested_id is not None:
                return nested_id
    return None


def _extract_heartbeat_id_from_exception_error(exc: Exception) -> str | None:
    status_code = getattr(exc, "status_code", None)
    if status_code != 400:
        return None
    error_msg = getattr(exc, "error_msg", None)
    if error_msg is None:
        return None
    return _extract_heartbeat_id_from_payload(error_msg)


def _find_trade_by_id(payload: object, trade_id: str) -> Mapping[str, Any] | None:
    normalized_id = trade_id.strip()
    if not normalized_id:
        return None
    if isinstance(payload, Mapping):
        candidate = _extract_first_str_by_keys(payload, ("id", "trade_id", "tradeID"))
        if candidate is not None and candidate.strip() == normalized_id:
            return payload
        for key in ("data", "result", "trade"):
            nested = payload.get(key)
            found = _find_trade_by_id(nested, normalized_id)
            if found is not None:
                return found
    elif isinstance(payload, Sequence) and not isinstance(payload, (str, bytes, bytearray)):
        for item in payload:
            found = _find_trade_by_id(item, normalized_id)
            if found is not None:
                return found
    return None


def _is_not_found_error(exc: Exception) -> bool:
    status_code = getattr(exc, "status_code", None)
    if status_code == 404:
        return True
    text = str(exc).strip().lower()
    if not text:
        return False
    return ("not found" in text) or ("status=404" in text) or ("404" in text and "unauthorized" not in text)


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


def _collect_open_order_ids(payload: object) -> set[str]:
    found: set[str] = set()
    if isinstance(payload, Mapping):
        candidate_id = _extract_direct_str_by_keys(payload, ("id", "orderID", "orderId", "order_id"))
        status_text = _extract_direct_str_by_keys(payload, ("status", "order_status", "state"))
        if candidate_id is not None and _is_order_mapping(payload):
            normalized = candidate_id.strip()
            if normalized and _is_open_order_status(status_text):
                found.add(normalized)
        for value in payload.values():
            found.update(_collect_open_order_ids(value))
        return found
    if isinstance(payload, Sequence) and not isinstance(payload, (str, bytes, bytearray)):
        for item in payload:
            found.update(_collect_open_order_ids(item))
        return found
    return found


def _count_canceled_orders(payload: object) -> int:
    if isinstance(payload, Mapping):
        for key in (
            "canceled",
            "cancelled",
            "canceledOrderIDs",
            "cancelledOrderIDs",
            "orderIDs",
            "order_ids",
        ):
            value = payload.get(key)
            if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
                return len([item for item in value if isinstance(item, str) and item.strip()])
        if isinstance(payload.get("success"), bool):
            return 1 if payload.get("success") is True else 0
    return len(_collect_open_order_ids(payload))


def _is_order_mapping(payload: Mapping[str, Any]) -> bool:
    if any(key in payload for key in ("id", "order_id", "orderID", "orderId")):
        return True
    if any(key in payload for key in ("token_id", "tokenID", "asset_id", "market_id", "price", "size", "side")):
        return True
    return any(key in payload for key in ("order_id", "orderID", "orderId"))


def _extract_direct_str_by_keys(payload: Mapping[str, Any], keys: Sequence[str]) -> str | None:
    for key in keys:
        candidate = payload.get(key)
        if isinstance(candidate, str):
            normalized = candidate.strip()
            if normalized:
                return normalized
    return None


def _is_open_order_status(status: str | None) -> bool:
    if status is None:
        return True
    normalized = status.strip().lower()
    if not normalized:
        return True
    closed_markers = {
        "matched",
        "filled",
        "executed",
        "canceled",
        "cancelled",
        "rejected",
        "expired",
        "closed",
        "completed",
    }
    return normalized not in closed_markers


def _safe_positive_float(value: object) -> float | None:
    if isinstance(value, (float, int)):
        parsed = float(value)
        return parsed if parsed > 0.0 else None
    if not isinstance(value, str):
        return None
    text = value.strip()
    if not text:
        return None
    try:
        parsed = float(text)
    except ValueError:
        return None
    return parsed if parsed > 0.0 else None


def _normalize_hex_address(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    normalized = value.strip()
    if len(normalized) != 42 or not normalized.startswith("0x"):
        return None
    try:
        int(normalized[2:], 16)
    except ValueError:
        return None
    return normalized


def _build_balance_allowance_params() -> object:
    if BalanceAllowanceParams is None:
        raise RuntimeError("py-clob-client BalanceAllowanceParams type is unavailable")
    params = BalanceAllowanceParams()  # type: ignore[misc,operator]
    setattr(params, "asset_type", "COLLATERAL")
    setattr(params, "signature_type", -1)
    return params


def _extract_collateral_by_keys(payload: object, keys: Sequence[str]) -> float | None:
    if isinstance(payload, Mapping):
        for key in keys:
            value = payload.get(key)
            parsed = _parse_collateral_amount(value)
            if parsed is not None:
                return parsed
        for value in payload.values():
            nested = _extract_collateral_by_keys(value, keys)
            if nested is not None:
                return nested
        return None
    if isinstance(payload, Sequence) and not isinstance(payload, (str, bytes, bytearray)):
        for item in payload:
            nested = _extract_collateral_by_keys(item, keys)
            if nested is not None:
                return nested
    return None


def _extract_max_collateral_allowance(payload: object) -> float | None:
    if isinstance(payload, Mapping):
        allowances = payload.get("allowances")
        if isinstance(allowances, Mapping):
            best: float | None = None
            for value in allowances.values():
                parsed = _parse_collateral_amount(value)
                if parsed is None:
                    continue
                if best is None or parsed > best:
                    best = parsed
            if best is not None:
                return best
        for value in payload.values():
            nested = _extract_max_collateral_allowance(value)
            if nested is not None:
                return nested
        return None
    if isinstance(payload, Sequence) and not isinstance(payload, (str, bytes, bytearray)):
        for item in payload:
            nested = _extract_max_collateral_allowance(item)
            if nested is not None:
                return nested
    return None


def _parse_collateral_amount(value: object) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        if value <= 0:
            return None
        return float(value) / COLLATERAL_SCALE
    if isinstance(value, float):
        if value <= 0.0:
            return None
        return value
    if not isinstance(value, str):
        return None

    text = value.strip()
    if not text:
        return None
    if any(marker in text for marker in (".", "e", "E")):
        try:
            parsed_float = float(text)
        except ValueError:
            return None
        return parsed_float if parsed_float > 0.0 else None
    try:
        parsed_int = int(text)
    except ValueError:
        try:
            parsed_float = float(text)
        except ValueError:
            return None
        return parsed_float if parsed_float > 0.0 else None
    if parsed_int <= 0:
        return None
    return float(parsed_int) / COLLATERAL_SCALE
