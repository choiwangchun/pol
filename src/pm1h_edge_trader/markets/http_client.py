from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from typing import Any, Mapping
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen


@dataclass(frozen=True)
class HttpResponseError(Exception):
    url: str
    status: int | None
    message: str

    def __str__(self) -> str:
        status = "unknown" if self.status is None else str(self.status)
        return f"HTTP request failed ({status}) for {self.url}: {self.message}"


class AsyncJsonHttpClient:
    """Small async JSON-over-HTTP client using only stdlib dependencies."""

    def __init__(
        self,
        *,
        default_headers: Mapping[str, str] | None = None,
        timeout_seconds: float = 10.0,
    ) -> None:
        self._default_headers = {
            "Accept": "application/json",
            "User-Agent": "pm1h-edge-trader/1.0",
            **(dict(default_headers) if default_headers else {}),
        }
        self._timeout_seconds = timeout_seconds

    async def get_json(
        self,
        url: str,
        *,
        params: Mapping[str, Any] | None = None,
        headers: Mapping[str, str] | None = None,
        timeout_seconds: float | None = None,
    ) -> Any:
        return await asyncio.to_thread(
            self._get_json_sync,
            url,
            params,
            headers,
            timeout_seconds,
        )

    def _get_json_sync(
        self,
        url: str,
        params: Mapping[str, Any] | None,
        headers: Mapping[str, str] | None,
        timeout_seconds: float | None,
    ) -> Any:
        merged_headers = {**self._default_headers, **(dict(headers) if headers else {})}
        final_url = self._build_url(url, params)
        request = Request(final_url, headers=merged_headers, method="GET")
        timeout = timeout_seconds or self._timeout_seconds

        try:
            with urlopen(request, timeout=timeout) as response:
                payload = response.read().decode("utf-8")
        except HTTPError as exc:
            body = self._safe_read_error_body(exc)
            raise HttpResponseError(final_url, exc.code, body) from exc
        except URLError as exc:
            raise HttpResponseError(final_url, None, str(exc.reason)) from exc

        try:
            return json.loads(payload)
        except json.JSONDecodeError as exc:
            raise HttpResponseError(final_url, None, "response is not valid JSON") from exc

    @staticmethod
    def _build_url(url: str, params: Mapping[str, Any] | None) -> str:
        if not params:
            return url
        filtered = {k: v for k, v in params.items() if v is not None}
        if not filtered:
            return url
        query = urlencode(filtered, doseq=True)
        separator = "&" if "?" in url else "?"
        return f"{url}{separator}{query}"

    @staticmethod
    def _safe_read_error_body(exc: HTTPError) -> str:
        try:
            body = exc.read().decode("utf-8", errors="replace")
            return body[:500] if body else exc.reason
        except Exception:
            return str(exc.reason)
