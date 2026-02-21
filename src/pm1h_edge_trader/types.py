"""Shared domain types and gateway interfaces."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from typing import Protocol


@dataclass(slots=True, frozen=True)
class MarketMeta:
    """Metadata for one binary prediction market."""

    market_id: str
    question: str
    token_up_id: str
    token_down_id: str
    close_time: datetime


@dataclass(slots=True, frozen=True)
class TokenBookTop:
    token_id: str
    bid: float
    ask: float
    bid_size: float
    ask_size: float
    timestamp: datetime

    @property
    def mid(self) -> float:
        return (self.bid + self.ask) / 2.0


@dataclass(slots=True, frozen=True)
class CandleState:
    open: float
    current: float
    close_time: datetime


@dataclass(slots=True, frozen=True)
class ProbabilityState:
    q_up: float
    q_down: float
    sigma: float
    tau: float


class DecisionAction(StrEnum):
    HOLD = "hold"
    BUY_UP = "buy-up"
    BUY_DOWN = "buy-down"


class OrderSide(StrEnum):
    BUY = "buy"
    SELL = "sell"


@dataclass(slots=True, frozen=True)
class DecisionState:
    action: DecisionAction
    edge_bps: float
    confidence: float
    reason: str
    intent: OrderIntent | None = None


@dataclass(slots=True, frozen=True)
class OrderIntent:
    market_id: str
    token_id: str
    side: OrderSide
    size: float
    limit_price: float
    created_at: datetime
    dry_run: bool
    client_order_id: str


@dataclass(slots=True, frozen=True)
class FillEvent:
    order_id: str
    token_id: str
    side: OrderSide
    fill_price: float
    fill_size: float
    fee_paid: float
    timestamp: datetime


@dataclass(slots=True, frozen=True)
class PnLEvent:
    timestamp: datetime
    realized_usd: float
    unrealized_usd: float
    total_usd: float
    note: str = ""


class PolymarketClient(Protocol):
    """Abstract gateway for Polymarket-facing operations."""

    async def get_market_meta(self, market_id: str) -> MarketMeta: ...

    async def get_token_book_top(self, token_id: str) -> TokenBookTop: ...

    async def place_order(self, intent: OrderIntent) -> str: ...


class BinanceClient(Protocol):
    """Abstract gateway for Binance market-state reads."""

    async def get_candle_state(self, symbol: str, interval: str = "1h") -> CandleState: ...
