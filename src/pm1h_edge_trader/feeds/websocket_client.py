from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Protocol


class WebSocketSession(Protocol):
    async def send(self, payload: str) -> None:
        ...

    async def recv(self) -> str:
        ...

    async def close(self) -> None:
        ...


class WebSocketClient(Protocol):
    async def connect(
        self,
        url: str,
        *,
        headers: Mapping[str, str] | None = None,
    ) -> WebSocketSession:
        ...


@dataclass
class _WebsocketsSession:
    _connection: Any

    async def send(self, payload: str) -> None:
        await self._connection.send(payload)

    async def recv(self) -> str:
        message = await self._connection.recv()
        if isinstance(message, bytes):
            return message.decode("utf-8", errors="replace")
        return str(message)

    async def close(self) -> None:
        await self._connection.close()


class OptionalWebsocketsClient:
    """Thin wrapper around the optional `websockets` dependency."""

    def __init__(
        self,
        *,
        open_timeout_seconds: float = 15.0,
        ping_interval_seconds: float = 20.0,
        ping_timeout_seconds: float = 20.0,
    ) -> None:
        self._open_timeout_seconds = open_timeout_seconds
        self._ping_interval_seconds = ping_interval_seconds
        self._ping_timeout_seconds = ping_timeout_seconds
        try:
            import websockets  # type: ignore
        except Exception:
            self._websockets = None
        else:
            self._websockets = websockets

    @property
    def available(self) -> bool:
        return self._websockets is not None

    async def connect(
        self,
        url: str,
        *,
        headers: Mapping[str, str] | None = None,
    ) -> WebSocketSession:
        if self._websockets is None:
            raise RuntimeError(
                "websockets package is not installed; websocket streaming is unavailable"
            )

        connect_kwargs = {
            "open_timeout": self._open_timeout_seconds,
            "ping_interval": self._ping_interval_seconds,
            "ping_timeout": self._ping_timeout_seconds,
        }

        if headers:
            try:
                connection = await self._websockets.connect(
                    url,
                    extra_headers=dict(headers),
                    **connect_kwargs,
                )
            except TypeError:
                connection = await self._websockets.connect(
                    url,
                    additional_headers=dict(headers),
                    **connect_kwargs,
                )
        else:
            connection = await self._websockets.connect(url, **connect_kwargs)

        return _WebsocketsSession(connection)
