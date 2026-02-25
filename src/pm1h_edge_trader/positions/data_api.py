from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from ..feeds.http_client import AsyncJsonHttpClient


class DataApiPositionsClient:
    def __init__(
        self,
        *,
        base_url: str = "https://data-api.polymarket.com",
        http_client: AsyncJsonHttpClient | None = None,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._http = http_client or AsyncJsonHttpClient()

    async def fetch_positions(
        self,
        *,
        user_address: str,
        size_threshold: float = 0.0,
        limit: int = 500,
    ) -> list[Mapping[str, Any]]:
        user = str(user_address).strip()
        if not user:
            return []
        page_limit = min(500, max(1, int(limit)))
        rows: list[Mapping[str, Any]] = []
        offset = 0
        while True:
            payload = await self._http.get_json(
                f"{self._base_url}/positions",
                params={
                    "user": user,
                    "sizeThreshold": f"{max(0.0, float(size_threshold)):.8f}",
                    "limit": str(page_limit),
                    "offset": str(offset),
                },
            )
            if not isinstance(payload, list):
                break
            page = [item for item in payload if isinstance(item, Mapping)]
            if not page:
                break
            rows.extend(page)
            if len(page) < page_limit:
                break
            offset += page_limit
        return rows
