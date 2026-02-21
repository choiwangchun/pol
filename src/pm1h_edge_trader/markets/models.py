from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Mapping, Protocol


@dataclass(frozen=True)
class MarketTiming:
    start: datetime
    end: datetime


@dataclass(frozen=True)
class OutcomeTokenIds:
    up_token_id: str
    down_token_id: str


@dataclass(frozen=True)
class ResolutionRuleCheck:
    allowed: bool
    has_binance_reference: bool
    has_btcusdt_reference: bool
    has_one_hour_reference: bool
    has_open_close_reference: bool
    reasons: tuple[str, ...]
    source_text: str


@dataclass(frozen=True)
class MarketCandidate:
    market_id: str
    question: str
    slug: str | None
    token_ids: OutcomeTokenIds
    timing: MarketTiming
    rule_check: ResolutionRuleCheck
    raw_market: Mapping[str, Any]


@dataclass(frozen=True)
class FeeRateStatus:
    token_id: str
    fee_rate: float
    fee_rate_bps: float
    trade_blocked: bool
    raw_response: Mapping[str, Any]


class MarketDiscovery(Protocol):
    async def find_active_btc_1h_up_down_market(
        self,
        *,
        require_rule_match: bool = True,
        preferred_slug: str | None = None,
    ) -> MarketCandidate | None:
        ...


class FeeRateChecker(Protocol):
    async def get_fee_rate_status(self, token_id: str) -> FeeRateStatus:
        ...

    async def any_trade_blocking_fee(
        self,
        token_ids: list[str],
    ) -> tuple[bool, dict[str, FeeRateStatus]]:
        ...
