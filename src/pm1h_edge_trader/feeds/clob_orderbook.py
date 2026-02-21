from __future__ import annotations

import asyncio
import json
import random
from datetime import datetime, timezone
from typing import Any, Iterable, Mapping, Sequence

from .http_client import AsyncJsonHttpClient, HttpResponseError
from .models import BestQuote, OrderbookSnapshot
from .websocket_client import OptionalWebsocketsClient, WebSocketClient, WebSocketSession


class ClobOrderbookAdapter:
    """
    Maintains best bid/ask for configured CLOB outcome token IDs.

    Websocket payload formats vary by deployment/channel. This adapter keeps
    parsing logic permissive and always runs REST polling as a safe fallback.
    """

    def __init__(
        self,
        *,
        token_ids: Sequence[str],
        rest_base_url: str = "https://clob.polymarket.com",
        ws_url: str = "wss://ws-subscriptions-clob.polymarket.com/ws/market",
        http_client: AsyncJsonHttpClient | None = None,
        ws_client: WebSocketClient | None = None,
        enable_websocket: bool = True,
        rest_poll_interval_seconds: float = 2.0,
        reconnect_base_delay_seconds: float = 1.0,
        reconnect_max_delay_seconds: float = 30.0,
        websocket_idle_timeout_seconds: float = 30.0,
    ) -> None:
        unique_tokens = tuple(dict.fromkeys(token_ids))
        if not unique_tokens:
            raise ValueError("token_ids cannot be empty")

        self._token_ids = unique_tokens
        self._rest_base_url = rest_base_url.rstrip("/")
        self._ws_url = ws_url
        self._http = http_client or AsyncJsonHttpClient()
        self._ws_client = ws_client or OptionalWebsocketsClient()
        self._enable_websocket = enable_websocket

        self._rest_poll_interval_seconds = max(rest_poll_interval_seconds, 0.0)
        self._reconnect_base_delay_seconds = max(reconnect_base_delay_seconds, 0.1)
        self._reconnect_max_delay_seconds = max(
            reconnect_max_delay_seconds,
            self._reconnect_base_delay_seconds,
        )
        self._websocket_idle_timeout_seconds = max(websocket_idle_timeout_seconds, 1.0)

        self._quotes: dict[str, BestQuote] = {
            token_id: BestQuote(
                token_id=token_id,
                best_bid=None,
                best_ask=None,
                updated_at=_utcnow(),
                source="init",
            )
            for token_id in self._token_ids
        }
        self._last_updated_at: datetime | None = None

        self._state_lock = asyncio.Lock()
        self._stop_event = asyncio.Event()
        self._tasks: set[asyncio.Task[Any]] = set()
        self._running = False

    async def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._stop_event.clear()

        self._tasks.add(asyncio.create_task(self._run_rest_poll_loop(), name="clob-rest-poll"))

        if self._enable_websocket and getattr(self._ws_client, "available", True):
            self._tasks.add(asyncio.create_task(self._run_websocket_loop(), name="clob-ws"))

    async def stop(self) -> None:
        if not self._running:
            return

        self._running = False
        self._stop_event.set()
        tasks = tuple(self._tasks)
        self._tasks.clear()
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    async def get_snapshot(self) -> OrderbookSnapshot:
        async with self._state_lock:
            return OrderbookSnapshot(
                quotes=dict(self._quotes),
                last_updated_at=self._last_updated_at,
            )

    async def get_best_quote(self, token_id: str) -> BestQuote | None:
        async with self._state_lock:
            return self._quotes.get(token_id)

    async def wait_until_ready(self, timeout_seconds: float = 10.0) -> bool:
        loop = asyncio.get_running_loop()
        deadline = loop.time() + timeout_seconds
        while loop.time() < deadline:
            snapshot = await self.get_snapshot()
            ready = all(
                quote.best_bid is not None and quote.best_ask is not None
                for quote in snapshot.quotes.values()
            )
            if ready:
                return True
            await asyncio.sleep(0.1)
        return False

    async def refresh_once(self) -> None:
        for token_id in self._token_ids:
            try:
                payload = await self._fetch_orderbook(token_id)
            except HttpResponseError:
                continue

            best_bid, best_ask = _extract_best_prices(payload)
            if best_bid is None and best_ask is None:
                continue
            await self._update_quote(token_id, best_bid, best_ask, source="rest")

    async def _run_rest_poll_loop(self) -> None:
        while not self._stop_event.is_set():
            try:
                await self.refresh_once()
            except Exception:
                # Keep loop alive and rely on next poll cycle.
                pass
            await self._wait_or_stop(self._rest_poll_interval_seconds)

    async def _run_websocket_loop(self) -> None:
        backoff = self._reconnect_base_delay_seconds

        while not self._stop_event.is_set():
            session: WebSocketSession | None = None
            try:
                session = await self._ws_client.connect(self._ws_url)
                await self._send_subscribe_message(session)
                backoff = self._reconnect_base_delay_seconds

                while not self._stop_event.is_set():
                    raw = await asyncio.wait_for(
                        session.recv(),
                        timeout=self._websocket_idle_timeout_seconds,
                    )
                    await self._handle_ws_message(raw)
            except Exception:
                # Keep running; REST polling remains active as fallback.
                await self._wait_or_stop(_with_jitter(backoff))
                backoff = min(backoff * 2.0, self._reconnect_max_delay_seconds)
            finally:
                if session is not None:
                    try:
                        await session.close()
                    except Exception:
                        pass

    async def _send_subscribe_message(self, session: WebSocketSession) -> None:
        # TODO: Confirm exact subscription schema for the target CLOB deployment.
        # This payload is intentionally simple and may need to be adjusted by orchestrator wiring.
        payload = {
            "type": "subscribe",
            "channel": "book",
            "token_ids": list(self._token_ids),
        }
        await session.send(json.dumps(payload))

    async def _handle_ws_message(self, raw_message: str) -> None:
        try:
            payload = json.loads(raw_message)
        except json.JSONDecodeError:
            return

        for message in _iter_messages(payload):
            candidates = _iter_book_like_payloads(message)
            for candidate in candidates:
                token_id = _extract_token_id(candidate) or _extract_token_id(message)
                if not token_id or token_id not in self._quotes:
                    continue
                best_bid, best_ask = _extract_best_prices(candidate)
                if best_bid is None and best_ask is None:
                    continue
                await self._update_quote(token_id, best_bid, best_ask, source="ws")

    async def _update_quote(
        self,
        token_id: str,
        best_bid: float | None,
        best_ask: float | None,
        *,
        source: str,
    ) -> None:
        async with self._state_lock:
            previous = self._quotes[token_id]
            self._quotes[token_id] = BestQuote(
                token_id=token_id,
                best_bid=best_bid if best_bid is not None else previous.best_bid,
                best_ask=best_ask if best_ask is not None else previous.best_ask,
                updated_at=_utcnow(),
                source=source,
            )
            self._last_updated_at = self._quotes[token_id].updated_at

    async def _fetch_orderbook(self, token_id: str) -> Mapping[str, Any]:
        attempts = (
            (f"{self._rest_base_url}/book", {"token_id": token_id}),
            (f"{self._rest_base_url}/orderbook", {"token_id": token_id}),
            (f"{self._rest_base_url}/book/{token_id}", None),
        )

        last_error: Exception | None = None
        for url, params in attempts:
            try:
                payload = await self._http.get_json(url, params=params)
                if isinstance(payload, Mapping):
                    return payload
            except Exception as exc:
                last_error = exc

        if last_error is not None:
            raise HttpResponseError(
                url=f"{self._rest_base_url}/book",
                status=None,
                message=str(last_error),
            ) from last_error

        raise HttpResponseError(
            url=f"{self._rest_base_url}/book",
            status=None,
            message=f"unexpected orderbook response for token_id={token_id}",
        )

    async def _wait_or_stop(self, delay_seconds: float) -> None:
        if delay_seconds <= 0:
            return
        try:
            await asyncio.wait_for(self._stop_event.wait(), timeout=delay_seconds)
        except asyncio.TimeoutError:
            return


def _iter_messages(payload: Any) -> Iterable[Mapping[str, Any]]:
    if isinstance(payload, Mapping):
        data = payload.get("data")
        if isinstance(data, list):
            yield payload
            for item in data:
                if isinstance(item, Mapping):
                    yield item
            return
        if isinstance(data, Mapping):
            yield payload
            yield data
            return
        yield payload
        return

    if isinstance(payload, list):
        for item in payload:
            if isinstance(item, Mapping):
                yield item


def _iter_book_like_payloads(message: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    payloads: list[Mapping[str, Any]] = [message]
    for key in ("books", "updates", "entries", "payload", "orderbook"):
        nested = message.get(key)
        if isinstance(nested, Mapping):
            payloads.append(nested)
        elif isinstance(nested, list):
            payloads.extend(item for item in nested if isinstance(item, Mapping))
    return payloads


def _extract_token_id(message: Mapping[str, Any]) -> str | None:
    keys = (
        "token_id",
        "tokenId",
        "asset_id",
        "assetId",
        "clob_token_id",
        "clobTokenId",
    )
    for key in keys:
        value = message.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def _extract_best_prices(message: Mapping[str, Any]) -> tuple[float | None, float | None]:
    best_bid = _first_float(message, "best_bid", "bestBid", "bid", "b")
    best_ask = _first_float(message, "best_ask", "bestAsk", "ask", "a")

    bids = message.get("bids")
    asks = message.get("asks")

    bid_prices = [
        price for price in (_extract_level_price(level) for level in _iter_levels(bids)) if price is not None
    ]
    ask_prices = [
        price for price in (_extract_level_price(level) for level in _iter_levels(asks)) if price is not None
    ]

    if best_bid is None and bid_prices:
        best_bid = max(bid_prices)
    if best_ask is None and ask_prices:
        best_ask = min(ask_prices)

    return best_bid, best_ask


def _iter_levels(levels: Any) -> Iterable[Any]:
    if isinstance(levels, list):
        for level in levels:
            yield level


def _extract_level_price(level: Any) -> float | None:
    if isinstance(level, Mapping):
        return _first_float(level, "price", "p", "rate")
    if isinstance(level, list) and level:
        return _to_float(level[0])
    if isinstance(level, tuple) and level:
        return _to_float(level[0])
    return None


def _first_float(mapping: Mapping[str, Any], *keys: str) -> float | None:
    for key in keys:
        value = _to_float(mapping.get(key))
        if value is not None:
            return value
    return None


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


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _with_jitter(delay: float) -> float:
    jitter = delay * 0.2
    return max(0.05, delay + random.uniform(0.0, jitter))
