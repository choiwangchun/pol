from __future__ import annotations

from typing import Any, Iterable, Mapping

from .http_client import AsyncJsonHttpClient
from .models import FeeRateStatus


class FeeRateGuardClient:
    """Checks whether token fees are non-zero and therefore trade-blocking."""

    def __init__(
        self,
        *,
        base_url: str = "https://clob.polymarket.com",
        http_client: AsyncJsonHttpClient | None = None,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._http = http_client or AsyncJsonHttpClient()

    async def get_fee_rate_status(self, token_id: str) -> FeeRateStatus:
        payload = await self._http.get_json(
            f"{self._base_url}/fee-rate",
            params={"token_id": token_id},
        )
        fee_rate, fee_rate_bps = _extract_fee_rate(payload)
        return FeeRateStatus(
            token_id=token_id,
            fee_rate=fee_rate,
            fee_rate_bps=fee_rate_bps,
            trade_blocked=fee_rate > 0.0,
            raw_response=payload if isinstance(payload, Mapping) else {"payload": payload},
        )

    async def any_trade_blocking_fee(
        self,
        token_ids: Iterable[str],
    ) -> tuple[bool, dict[str, FeeRateStatus]]:
        statuses: dict[str, FeeRateStatus] = {}
        for token_id in token_ids:
            status = await self.get_fee_rate_status(token_id)
            statuses[token_id] = status
        blocked = any(item.trade_blocked for item in statuses.values())
        return blocked, statuses


def _extract_fee_rate(payload: Any) -> tuple[float, float]:
    mapping = _coerce_mapping(payload)

    candidate_rate_keys = (
        "fee_rate",
        "feeRate",
        "makerFeeRate",
        "takerFeeRate",
        "makerBaseFee",
        "takerBaseFee",
    )
    candidate_bps_keys = (
        "fee_rate_bps",
        "feeRateBps",
        "makerFeeBps",
        "takerFeeBps",
    )

    rate_values: list[float] = []
    for key in candidate_rate_keys:
        value = _to_float(mapping.get(key))
        if value is not None:
            rate_values.append(value)

    bps_values: list[float] = []
    for key in candidate_bps_keys:
        value = _to_float(mapping.get(key))
        if value is not None:
            bps_values.append(value)

    fee_rate_from_rates = max(rate_values) if rate_values else 0.0
    fee_rate_from_bps = (max(bps_values) / 10_000.0) if bps_values else 0.0

    fee_rate = max(fee_rate_from_rates, fee_rate_from_bps)
    fee_rate_bps = max(bps_values) if bps_values else fee_rate * 10_000.0
    return fee_rate, fee_rate_bps


def _coerce_mapping(payload: Any) -> Mapping[str, Any]:
    if isinstance(payload, Mapping):
        nested = payload.get("data")
        if isinstance(nested, Mapping):
            return nested
        return payload
    return {}


def _to_float(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return None
        try:
            return float(stripped)
        except ValueError:
            return None
    return None
