from __future__ import annotations

import re
from typing import Any, Iterable, Mapping

from .models import ResolutionRuleCheck


_BINANCE_RE = re.compile(r"\bbinance\b", re.IGNORECASE)
_BTCUSDT_RE = re.compile(r"\bbtc\s*/\s*usdt\b|\bbtcusdt\b", re.IGNORECASE)
_ONE_HOUR_RE = re.compile(
    r"\b1\s*h\b|\b1h\b|\b1\s*hr\b|\b1-hour\b|\bone\s+hour\b",
    re.IGNORECASE,
)
_OPEN_RE = re.compile(r"\bopen(?:ing)?\b", re.IGNORECASE)
_CLOSE_RE = re.compile(r"\bclose(?:d|ing)?\b", re.IGNORECASE)


def collect_rule_text(market: Mapping[str, Any]) -> str:
    fields = (
        "rules",
        "resolutionRules",
        "resolutionSource",
        "description",
        "question",
        "title",
    )
    chunks: list[str] = []

    for field in fields:
        value = market.get(field)
        if isinstance(value, str) and value.strip():
            chunks.append(value.strip())

    resolution = market.get("resolution")
    if isinstance(resolution, Mapping):
        for key in ("rules", "source", "description"):
            value = resolution.get(key)
            if isinstance(value, str) and value.strip():
                chunks.append(value.strip())

    return "\n".join(chunks)


def verify_binance_btcusdt_1h_rule(text: str | Iterable[str]) -> ResolutionRuleCheck:
    source_text = _coerce_to_text(text)
    has_binance = bool(_BINANCE_RE.search(source_text))
    has_btcusdt = bool(_BTCUSDT_RE.search(source_text))
    has_one_hour = bool(_ONE_HOUR_RE.search(source_text))
    has_open_close = bool(_OPEN_RE.search(source_text) and _CLOSE_RE.search(source_text))

    reasons: list[str] = []
    if not has_binance:
        reasons.append("missing Binance reference")
    if not has_btcusdt:
        reasons.append("missing BTC/USDT reference")
    if not has_one_hour:
        reasons.append("missing 1H/one-hour timeframe reference")
    if not has_open_close:
        reasons.append("missing open+close price reference")

    return ResolutionRuleCheck(
        allowed=not reasons,
        has_binance_reference=has_binance,
        has_btcusdt_reference=has_btcusdt,
        has_one_hour_reference=has_one_hour,
        has_open_close_reference=has_open_close,
        reasons=tuple(reasons),
        source_text=source_text,
    )


def _coerce_to_text(text: str | Iterable[str]) -> str:
    if isinstance(text, str):
        return text
    return "\n".join(chunk for chunk in text if isinstance(chunk, str))
