from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Mapping, Protocol


@dataclass(frozen=True)
class BestQuote:
    token_id: str
    best_bid: float | None
    best_ask: float | None
    updated_at: datetime
    source: str


@dataclass(frozen=True)
class OrderbookSnapshot:
    quotes: Mapping[str, BestQuote]
    last_updated_at: datetime | None


@dataclass(frozen=True)
class UnderlyingSnapshot:
    symbol: str
    interval: str
    candle_open: float | None
    candle_open_time: datetime | None
    last_price: float | None
    price_time: datetime | None
    source: str


class OrderbookFeed(Protocol):
    async def start(self) -> None:
        ...

    async def stop(self) -> None:
        ...

    async def get_snapshot(self) -> OrderbookSnapshot:
        ...


class UnderlyingFeed(Protocol):
    async def start(self) -> None:
        ...

    async def stop(self) -> None:
        ...

    async def get_snapshot(self) -> UnderlyingSnapshot:
        ...
