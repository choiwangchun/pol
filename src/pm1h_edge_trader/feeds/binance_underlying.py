from __future__ import annotations

import asyncio
import json
import random
from dataclasses import replace
from datetime import datetime, timezone
from typing import Any, Mapping, Sequence

from .http_client import AsyncJsonHttpClient
from .models import UnderlyingSnapshot
from .websocket_client import OptionalWebsocketsClient, WebSocketClient, WebSocketSession


class BinanceUnderlyingAdapter:
    """
    Tracks Binance BTCUSDT 1H in-progress candle open (K) and live price (S_t).

    Uses websocket when available and keeps REST polling enabled as fallback.
    """

    def __init__(
        self,
        *,
        symbol: str = "BTCUSDT",
        interval: str = "1h",
        rest_base_url: str = "https://api.binance.com",
        ws_url: str | None = None,
        http_client: AsyncJsonHttpClient | None = None,
        ws_client: WebSocketClient | None = None,
        enable_websocket: bool = True,
        rest_poll_interval_seconds: float = 2.0,
        reconnect_base_delay_seconds: float = 1.0,
        reconnect_max_delay_seconds: float = 30.0,
        websocket_idle_timeout_seconds: float = 30.0,
    ) -> None:
        self._symbol = symbol.upper()
        self._interval = interval
        self._rest_base_url = rest_base_url.rstrip("/")
        self._http = http_client or AsyncJsonHttpClient()
        self._ws_client = ws_client or OptionalWebsocketsClient()
        self._enable_websocket = enable_websocket

        self._ws_url = ws_url or self._default_ws_url(self._symbol, self._interval)

        self._rest_poll_interval_seconds = max(rest_poll_interval_seconds, 0.0)
        self._reconnect_base_delay_seconds = max(reconnect_base_delay_seconds, 0.1)
        self._reconnect_max_delay_seconds = max(
            reconnect_max_delay_seconds,
            self._reconnect_base_delay_seconds,
        )
        self._websocket_idle_timeout_seconds = max(websocket_idle_timeout_seconds, 1.0)

        self._snapshot = UnderlyingSnapshot(
            symbol=self._symbol,
            interval=self._interval,
            candle_open=None,
            candle_open_time=None,
            last_price=None,
            price_time=None,
            source="init",
        )

        self._state_lock = asyncio.Lock()
        self._stop_event = asyncio.Event()
        self._tasks: set[asyncio.Task[Any]] = set()
        self._running = False

    async def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._stop_event.clear()

        self._tasks.add(asyncio.create_task(self._run_rest_poll_loop(), name="binance-rest-poll"))

        if self._enable_websocket and getattr(self._ws_client, "available", True):
            self._tasks.add(asyncio.create_task(self._run_websocket_loop(), name="binance-ws"))

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

    async def get_snapshot(self) -> UnderlyingSnapshot:
        async with self._state_lock:
            return self._snapshot

    async def get_open_and_price(self) -> tuple[float | None, float | None]:
        snapshot = await self.get_snapshot()
        return snapshot.candle_open, snapshot.last_price

    async def get_open_and_live_price(
        self,
        *,
        ensure_fresh: bool = False,
    ) -> tuple[float | None, float | None]:
        if ensure_fresh:
            await self.refresh_from_rest()
        return await self.get_open_and_price()

    async def wait_until_ready(self, timeout_seconds: float = 10.0) -> bool:
        loop = asyncio.get_running_loop()
        deadline = loop.time() + timeout_seconds
        while loop.time() < deadline:
            snapshot = await self.get_snapshot()
            if snapshot.candle_open is not None and snapshot.last_price is not None:
                return True
            await asyncio.sleep(0.1)
        return False

    async def refresh_from_rest(self) -> UnderlyingSnapshot:
        ticker_task = asyncio.create_task(self._fetch_ticker_price())
        candle_task = asyncio.create_task(self._fetch_current_candle_open())

        ticker_result, candle_result = await asyncio.gather(
            ticker_task,
            candle_task,
            return_exceptions=True,
        )

        price: float | None = None
        if not isinstance(ticker_result, Exception):
            price = ticker_result

        candle_open: float | None = None
        candle_open_time: datetime | None = None
        if not isinstance(candle_result, Exception):
            candle_open, candle_open_time = candle_result

        await self._apply_update(
            price=price,
            price_time=_utcnow() if price is not None else None,
            candle_open=candle_open,
            candle_open_time=candle_open_time,
            source="rest",
        )
        return await self.get_snapshot()

    async def _run_rest_poll_loop(self) -> None:
        while not self._stop_event.is_set():
            try:
                await self.refresh_from_rest()
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
                backoff = self._reconnect_base_delay_seconds

                while not self._stop_event.is_set():
                    raw = await asyncio.wait_for(
                        session.recv(),
                        timeout=self._websocket_idle_timeout_seconds,
                    )
                    await self._handle_ws_message(raw)
            except Exception:
                await self._wait_or_stop(_with_jitter(backoff))
                backoff = min(backoff * 2.0, self._reconnect_max_delay_seconds)
            finally:
                if session is not None:
                    try:
                        await session.close()
                    except Exception:
                        pass

    async def _handle_ws_message(self, raw_message: str) -> None:
        try:
            payload = json.loads(raw_message)
        except json.JSONDecodeError:
            return

        events = _extract_events(payload)
        for event in events:
            await self._apply_ws_event(event)

    async def _apply_ws_event(self, event: Mapping[str, Any]) -> None:
        event_type = str(event.get("e", "")).lower()

        if event_type in {"trade", "aggtrade"}:
            price = _to_float(event.get("p"))
            price_time = _parse_timestamp(event.get("T") or event.get("E")) or _utcnow()
            await self._apply_update(price=price, price_time=price_time, source="ws")
            return

        if event_type in {"24hrticker", "miniticker"}:
            price = _to_float(event.get("c"))
            price_time = _parse_timestamp(event.get("E")) or _utcnow()
            await self._apply_update(price=price, price_time=price_time, source="ws")
            return

        if event_type == "kline":
            kline = event.get("k")
            if not isinstance(kline, Mapping):
                return
            interval = str(kline.get("i", "")).lower()
            if interval != self._interval.lower():
                return
            candle_open = _to_float(kline.get("o"))
            candle_open_time = _parse_timestamp(kline.get("t"))
            # Use kline close as a supplemental live price if trade stream is delayed.
            price = _to_float(kline.get("c"))
            price_time = _parse_timestamp(kline.get("T")) or _utcnow()
            await self._apply_update(
                price=price,
                price_time=price_time,
                candle_open=candle_open,
                candle_open_time=candle_open_time,
                source="ws",
            )

    async def _apply_update(
        self,
        *,
        price: float | None = None,
        price_time: datetime | None = None,
        candle_open: float | None = None,
        candle_open_time: datetime | None = None,
        source: str,
    ) -> None:
        async with self._state_lock:
            snapshot = self._snapshot
            if price is not None:
                snapshot = replace(
                    snapshot,
                    last_price=price,
                    price_time=price_time or _utcnow(),
                    source=source,
                )
            if candle_open is not None:
                snapshot = replace(
                    snapshot,
                    candle_open=candle_open,
                    candle_open_time=candle_open_time or _utcnow(),
                    source=source,
                )
            self._snapshot = snapshot

    async def _fetch_ticker_price(self) -> float:
        payload = await self._http.get_json(
            f"{self._rest_base_url}/api/v3/ticker/price",
            params={"symbol": self._symbol},
        )
        if not isinstance(payload, Mapping):
            raise ValueError("unexpected ticker payload")
        price = _to_float(payload.get("price"))
        if price is None:
            raise ValueError("ticker payload missing price")
        return price

    async def _fetch_current_candle_open(self) -> tuple[float, datetime]:
        payload = await self._http.get_json(
            f"{self._rest_base_url}/api/v3/klines",
            params={"symbol": self._symbol, "interval": self._interval, "limit": 2},
        )
        row = _select_current_kline(payload)
        if row is None:
            raise ValueError("kline payload missing rows")

        open_price = _to_float(row[1])
        open_time = _parse_timestamp(row[0])
        if open_price is None or open_time is None:
            raise ValueError("kline payload missing open price/time")
        return open_price, open_time

    async def _wait_or_stop(self, delay_seconds: float) -> None:
        if delay_seconds <= 0:
            return
        try:
            await asyncio.wait_for(self._stop_event.wait(), timeout=delay_seconds)
        except asyncio.TimeoutError:
            return

    @staticmethod
    def _default_ws_url(symbol: str, interval: str) -> str:
        stream_symbol = symbol.lower()
        # Combined stream carries both trade ticks and 1h kline updates.
        return (
            "wss://stream.binance.com:9443/stream"
            f"?streams={stream_symbol}@trade/{stream_symbol}@kline_{interval}"
        )


def _extract_events(payload: Any) -> Sequence[Mapping[str, Any]]:
    if isinstance(payload, Mapping):
        data = payload.get("data")
        if isinstance(data, Mapping):
            return [data]
        if isinstance(data, list):
            return [item for item in data if isinstance(item, Mapping)]
        return [payload]

    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, Mapping)]

    return []


def _select_current_kline(payload: Any) -> Sequence[Any] | None:
    if not isinstance(payload, list):
        return None

    rows = [row for row in payload if isinstance(row, list) and len(row) >= 7]
    if not rows:
        return None

    now_ms = int(_utcnow().timestamp() * 1000)
    for row in rows:
        open_ms = _to_int(row[0])
        close_ms = _to_int(row[6])
        if open_ms is None or close_ms is None:
            continue
        if open_ms <= now_ms < close_ms:
            return row

    return rows[-1]


def _parse_timestamp(value: Any) -> datetime | None:
    millis = _to_int(value)
    if millis is None:
        return None
    seconds = millis / 1000.0 if millis > 1_000_000_000_000 else float(millis)
    return datetime.fromtimestamp(seconds, tz=timezone.utc)


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


def _to_int(value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return None
        try:
            return int(stripped)
        except ValueError:
            return None
    return None


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _with_jitter(delay: float) -> float:
    jitter = delay * 0.2
    return max(0.05, delay + random.uniform(0.0, jitter))
