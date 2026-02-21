from __future__ import annotations

import json
import re
from datetime import datetime, timedelta, timezone
from typing import Any, Mapping, Sequence
from zoneinfo import ZoneInfo

from .http_client import AsyncJsonHttpClient
from .models import MarketCandidate, MarketTiming, OutcomeTokenIds
from .rule_parser import collect_rule_text, verify_binance_btcusdt_1h_rule


_BTC_RE = re.compile(r"\bbtc\b|\bbitcoin\b", re.IGNORECASE)
_ONE_HOUR_RE = re.compile(
    r"\b1\s*h\b|\b1h\b|\b1\s*hr\b|\b1-hour\b|\bone\s+hour\b",
    re.IGNORECASE,
)
_UP_RE = re.compile(r"\bup\b", re.IGNORECASE)
_DOWN_RE = re.compile(r"\bdown\b", re.IGNORECASE)


class GammaMarketDiscoveryClient:
    """Discover active BTC 1H Up/Down markets from the Gamma API."""

    def __init__(
        self,
        *,
        base_url: str = "https://gamma-api.polymarket.com",
        page_size: int = 200,
        max_pages: int = 8,
        slug_probe_lookback_hours: int = 48,
        slug_probe_lookahead_hours: int = 2,
        http_client: AsyncJsonHttpClient | None = None,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._page_size = page_size
        self._max_pages = max_pages
        self._slug_probe_lookback_hours = max(1, slug_probe_lookback_hours)
        self._slug_probe_lookahead_hours = max(0, slug_probe_lookahead_hours)
        self._http = http_client or AsyncJsonHttpClient()

    async def find_active_btc_1h_up_down_market(
        self,
        *,
        require_rule_match: bool = True,
        preferred_slug: str | None = None,
    ) -> MarketCandidate | None:
        # 1) Fast path: explicit slug, then generated hourly BTC Up/Down slugs around now (ET).
        for slug in _build_slug_probe_list(
            preferred_slug=preferred_slug,
            lookback_hours=self._slug_probe_lookback_hours,
            lookahead_hours=self._slug_probe_lookahead_hours,
        ):
            market = await self._find_by_slug(slug=slug, require_rule_match=require_rule_match)
            if market is not None:
                return market

        # 2) Fallback: paginated scan of active markets.
        offset = 0

        for _ in range(self._max_pages):
            payload = await self._http.get_json(
                f"{self._base_url}/markets",
                params={
                    "active": "true",
                    "closed": "false",
                    "archived": "false",
                    "limit": self._page_size,
                    "offset": offset,
                },
            )
            markets = _extract_markets(payload)
            if not markets:
                return None

            for market in markets:
                candidate = _build_candidate(market)
                if candidate is None:
                    continue
                if require_rule_match and not candidate.rule_check.allowed:
                    continue
                return candidate

            if len(markets) < self._page_size:
                return None
            offset += len(markets)

        return None

    async def _find_by_slug(
        self,
        *,
        slug: str,
        require_rule_match: bool,
    ) -> MarketCandidate | None:
        market_payload = await self._http.get_json(
            f"{self._base_url}/markets",
            params={"slug": slug},
        )
        market_rows = _extract_markets(market_payload)
        for market in market_rows:
            candidate = _build_candidate(market)
            if candidate is None:
                continue
            if require_rule_match and not candidate.rule_check.allowed:
                continue
            return candidate

        # Some payload variants may only expose markets through event expansions.
        event_payload = await self._http.get_json(
            f"{self._base_url}/events",
            params={"slug": slug},
        )
        for event_market in _extract_event_markets(event_payload):
            candidate = _build_candidate(event_market)
            if candidate is None:
                continue
            if require_rule_match and not candidate.rule_check.allowed:
                continue
            return candidate
        return None


def _build_candidate(market: Mapping[str, Any]) -> MarketCandidate | None:
    if not _is_active_market(market):
        return None

    question = _first_non_empty_str(
        market,
        "question",
        "title",
        "name",
    )
    slug = _first_non_empty_str(market, "slug") or None
    text = _market_text_blob(market)

    if not _looks_like_btc_1h_up_down(text, market):
        return None

    try:
        token_ids = _extract_up_down_token_ids(market)
        timing = _extract_timing(market)
    except ValueError:
        return None
    if timing.end <= datetime.now(timezone.utc):
        return None

    rule_text = collect_rule_text(market)
    rule_check = verify_binance_btcusdt_1h_rule(rule_text)

    market_id = _first_non_empty_str(market, "id", "marketId", "conditionId")
    if not market_id:
        return None

    return MarketCandidate(
        market_id=market_id,
        question=question,
        slug=slug,
        token_ids=token_ids,
        timing=timing,
        rule_check=rule_check,
        raw_market=dict(market),
    )


def _extract_markets(payload: Any) -> list[Mapping[str, Any]]:
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, Mapping)]
    if isinstance(payload, Mapping):
        for key in ("markets", "data", "results"):
            value = payload.get(key)
            if isinstance(value, list):
                return [item for item in value if isinstance(item, Mapping)]
    return []


def _extract_event_markets(payload: Any) -> list[Mapping[str, Any]]:
    event_rows = _extract_markets(payload)
    markets: list[Mapping[str, Any]] = []
    for event in event_rows:
        nested = event.get("markets")
        if isinstance(nested, list):
            markets.extend(item for item in nested if isinstance(item, Mapping))
    return markets


def _build_slug_probe_list(
    *,
    preferred_slug: str | None,
    lookback_hours: int,
    lookahead_hours: int,
) -> list[str]:
    candidates: list[str] = []
    if isinstance(preferred_slug, str):
        slug = preferred_slug.strip().lower()
        if slug:
            candidates.append(slug)

    now_et = datetime.now(timezone.utc).astimezone(ZoneInfo("America/New_York"))
    base_hour = now_et.replace(minute=0, second=0, microsecond=0)
    deltas = [0]
    horizon = max(lookback_hours, lookahead_hours)
    for step in range(1, horizon + 1):
        if step <= lookback_hours:
            deltas.append(-step)
        if step <= lookahead_hours:
            deltas.append(step)
    for delta in deltas:
        moment = base_hour + timedelta(hours=delta)
        candidates.append(_hourly_btc_updown_slug(moment))

    # preserve insertion order while removing duplicates
    return list(dict.fromkeys(candidates))


def _hourly_btc_updown_slug(moment_et: datetime) -> str:
    month = moment_et.strftime("%B").lower()
    day = str(moment_et.day)
    hour = moment_et.strftime("%I").lstrip("0") or "12"
    ampm = moment_et.strftime("%p").lower()
    return f"bitcoin-up-or-down-{month}-{day}-{hour}{ampm}-et"


def _is_active_market(market: Mapping[str, Any]) -> bool:
    active = _coerce_bool(market.get("active"), default=True)
    closed = _coerce_bool(market.get("closed"), default=False)
    archived = _coerce_bool(market.get("archived"), default=False)
    accepting_orders = _coerce_bool(market.get("acceptingOrders"), default=True)
    return active and not closed and not archived and accepting_orders


def _looks_like_btc_1h_up_down(text: str, market: Mapping[str, Any]) -> bool:
    if not _BTC_RE.search(text):
        return False

    has_timeframe_text = bool(_ONE_HOUR_RE.search(text))
    duration = _try_extract_duration_seconds(market)
    has_one_hour_duration = duration is not None and 45 * 60 <= duration <= 90 * 60

    outcomes = [_normalize_label(value) for value in _extract_outcomes(market)]
    has_up_down_outcomes = any("up" in outcome for outcome in outcomes) and any(
        "down" in outcome for outcome in outcomes
    )
    has_up_down_text = bool(_UP_RE.search(text) and _DOWN_RE.search(text))

    return (has_timeframe_text or has_one_hour_duration) and (
        has_up_down_outcomes or has_up_down_text
    )


def _extract_up_down_token_ids(market: Mapping[str, Any]) -> OutcomeTokenIds:
    token_map = _extract_token_map_from_token_objects(market)
    if "up" in token_map and "down" in token_map:
        return OutcomeTokenIds(up_token_id=token_map["up"], down_token_id=token_map["down"])

    mapped_direct = _extract_up_down_from_mapping_like(market.get("clobTokenIds"))
    if mapped_direct is not None:
        return mapped_direct

    mapped_direct_alt = _extract_up_down_from_mapping_like(market.get("clob_token_ids"))
    if mapped_direct_alt is not None:
        return mapped_direct_alt

    outcomes = [_normalize_label(value) for value in _extract_outcomes(market)]
    clob_tokens = _extract_clob_token_ids(market)
    if len(outcomes) != len(clob_tokens):
        raise ValueError("outcomes and clobTokenIds length mismatch")

    up_token: str | None = None
    down_token: str | None = None
    for outcome, token_id in zip(outcomes, clob_tokens, strict=False):
        if "up" in outcome:
            up_token = token_id
        elif "down" in outcome:
            down_token = token_id

    if not up_token or not down_token:
        raise ValueError("failed to map Up/Down token IDs")

    return OutcomeTokenIds(up_token_id=up_token, down_token_id=down_token)


def _extract_token_map_from_token_objects(market: Mapping[str, Any]) -> dict[str, str]:
    tokens = market.get("tokens")
    if not isinstance(tokens, list):
        return {}

    mapped: dict[str, str] = {}
    for token in tokens:
        if not isinstance(token, Mapping):
            continue
        outcome = _normalize_label(str(token.get("outcome", "")))
        token_id = _first_non_empty_str(token, "token_id", "tokenId", "clobTokenId", "id")
        if not token_id:
            continue
        if "up" in outcome:
            mapped["up"] = token_id
        elif "down" in outcome:
            mapped["down"] = token_id

    return mapped


def _extract_timing(market: Mapping[str, Any]) -> MarketTiming:
    start = _extract_datetime(
        market,
        "startDate",
        "startTime",
        "start",
        "marketStartDate",
        "startTimestamp",
    )
    end = _extract_datetime(
        market,
        "endDate",
        "endTime",
        "end",
        "marketEndDate",
        "endTimestamp",
    )
    if start is None or end is None:
        raise ValueError("missing market timing")
    if end <= start:
        raise ValueError("invalid market timing")
    return MarketTiming(start=start, end=end)


def _try_extract_duration_seconds(market: Mapping[str, Any]) -> int | None:
    start = _extract_datetime(
        market,
        "startDate",
        "startTime",
        "start",
        "marketStartDate",
        "startTimestamp",
    )
    end = _extract_datetime(
        market,
        "endDate",
        "endTime",
        "end",
        "marketEndDate",
        "endTimestamp",
    )
    if start is None or end is None or end <= start:
        return None
    return int((end - start).total_seconds())


def _extract_datetime(market: Mapping[str, Any], *keys: str) -> datetime | None:
    for key in keys:
        raw = market.get(key)
        parsed = _parse_datetime(raw)
        if parsed is not None:
            return parsed
    return None


def _parse_datetime(raw: Any) -> datetime | None:
    if raw is None:
        return None
    if isinstance(raw, datetime):
        dt = raw
    elif isinstance(raw, (int, float)):
        seconds = raw / 1000.0 if raw > 1_000_000_000_000 else float(raw)
        dt = datetime.fromtimestamp(seconds, tz=timezone.utc)
    elif isinstance(raw, str):
        value = raw.strip()
        if not value:
            return None
        if value.isdigit():
            number = int(value)
            seconds = number / 1000.0 if number > 1_000_000_000_000 else float(number)
            return datetime.fromtimestamp(seconds, tz=timezone.utc)
        if value.endswith("Z"):
            value = f"{value[:-1]}+00:00"
        try:
            dt = datetime.fromisoformat(value)
        except ValueError:
            return None
    else:
        return None

    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _extract_outcomes(market: Mapping[str, Any]) -> list[str]:
    outcomes = market.get("outcomes")
    parsed = _coerce_string_list(outcomes)
    if parsed:
        return parsed

    tokens = market.get("tokens")
    if isinstance(tokens, list):
        extracted = []
        for token in tokens:
            if isinstance(token, Mapping):
                outcome = token.get("outcome")
                if isinstance(outcome, str) and outcome.strip():
                    extracted.append(outcome.strip())
        if extracted:
            return extracted

    return []


def _extract_clob_token_ids(market: Mapping[str, Any]) -> list[str]:
    value = market.get("clobTokenIds")
    parsed = _coerce_string_list(value)
    if parsed:
        return parsed

    # Alternate field naming seen in some payload variants.
    alternate = _coerce_string_list(market.get("clob_token_ids"))
    if alternate:
        return alternate

    tokens = market.get("tokens")
    if isinstance(tokens, list):
        extracted = []
        for token in tokens:
            if isinstance(token, Mapping):
                token_id = _first_non_empty_str(token, "token_id", "tokenId", "clobTokenId", "id")
                if token_id:
                    extracted.append(token_id)
        if extracted:
            return extracted

    return []


def _coerce_string_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if isinstance(value, tuple):
        return [str(item).strip() for item in value if str(item).strip()]
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return []
        if text.startswith("["):
            try:
                decoded = json.loads(text)
            except json.JSONDecodeError:
                return []
            if isinstance(decoded, list):
                return [str(item).strip() for item in decoded if str(item).strip()]
            return []
        if "," in text:
            return [chunk.strip() for chunk in text.split(",") if chunk.strip()]
        return [text]
    return []


def _extract_up_down_from_mapping_like(value: Any) -> OutcomeTokenIds | None:
    mapping = _coerce_mapping_like(value)
    if not mapping:
        return None

    up_token = ""
    down_token = ""
    for key, raw_value in mapping.items():
        if not isinstance(key, str):
            continue
        token_id = str(raw_value).strip()
        if not token_id:
            continue
        normalized_key = _normalize_label(key)
        if "up" in normalized_key:
            up_token = token_id
        elif "down" in normalized_key:
            down_token = token_id

    if up_token and down_token:
        return OutcomeTokenIds(up_token_id=up_token, down_token_id=down_token)
    return None


def _coerce_mapping_like(value: Any) -> Mapping[str, Any] | None:
    if isinstance(value, Mapping):
        return value
    if isinstance(value, str):
        text = value.strip()
        if text.startswith("{"):
            try:
                parsed = json.loads(text)
            except json.JSONDecodeError:
                return None
            if isinstance(parsed, Mapping):
                return parsed
    return None


def _first_non_empty_str(mapping: Mapping[str, Any], *keys: str) -> str:
    for key in keys:
        value = mapping.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def _coerce_bool(value: Any, *, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "1", "yes", "y"}:
            return True
        if normalized in {"false", "0", "no", "n"}:
            return False
    return default


def _market_text_blob(market: Mapping[str, Any]) -> str:
    parts: list[str] = []
    for key in ("question", "title", "slug", "description", "rules", "resolutionSource"):
        value = market.get(key)
        if isinstance(value, str) and value.strip():
            parts.append(value.strip())

    events = market.get("events")
    if isinstance(events, list):
        for event in events:
            if not isinstance(event, Mapping):
                continue
            for key in ("title", "slug", "description"):
                value = event.get(key)
                if isinstance(value, str) and value.strip():
                    parts.append(value.strip())

    return "\n".join(parts)


def _normalize_label(label: str) -> str:
    return re.sub(r"\s+", " ", label).strip().lower()
