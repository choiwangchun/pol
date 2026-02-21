from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Protocol

from ..feeds.http_client import AsyncJsonHttpClient
from ..logger.models import ExecutionLogRecord


def utc_now() -> datetime:
    return datetime.now(tz=timezone.utc)


@dataclass(frozen=True)
class MarketResolution:
    winner: str | None
    resolved: bool


class MarketResolutionResolver(Protocol):
    async def resolve_market(self, market_id: str) -> MarketResolution:
        """Returns resolved winner for one market, when available."""


class _ExecutionReporter(Protocol):
    def append(self, record: ExecutionLogRecord) -> None:
        """Appends one execution-like record."""


@dataclass(frozen=True)
class _PaperFill:
    timestamp: datetime
    market_id: str
    order_id: str
    side: str
    price: float
    size: float
    K: float
    S_t: float
    tau: float
    sigma: float
    q_up: float
    bid_up: float
    ask_up: float
    bid_down: float
    ask_down: float
    edge: float


class PaperSettlementCoordinator:
    """Reconciles unresolved paper fills into settlement records."""

    def __init__(
        self,
        *,
        execution_csv_path: str | Path,
        reporter: _ExecutionReporter,
        resolver: MarketResolutionResolver,
        now_fn=utc_now,
    ) -> None:
        self._execution_csv_path = Path(execution_csv_path)
        self._reporter = reporter
        self._resolver = resolver
        self._now_fn = now_fn

    async def reconcile(self) -> int:
        records = await self.reconcile_records()
        return len(records)

    async def reconcile_records(self) -> list[ExecutionLogRecord]:
        rows = _read_csv_rows(self._execution_csv_path)
        fills = _extract_unsettled_paper_fills(rows)
        if not fills:
            return []

        market_cache: dict[str, MarketResolution] = {}
        appended_records: list[ExecutionLogRecord] = []
        for fill in fills:
            resolution = market_cache.get(fill.market_id)
            if resolution is None:
                resolution = await self._resolver.resolve_market(fill.market_id)
                market_cache[fill.market_id] = resolution

            winner = _normalize_side_label(resolution.winner)
            if not resolution.resolved or winner not in {"up", "down"}:
                continue

            pnl = _paper_binary_pnl(
                entry_price=fill.price,
                size=fill.size,
                position_side=fill.side,
                winner_side=winner,
            )
            outcome = "win" if pnl > 0 else "loss"
            record = ExecutionLogRecord(
                timestamp=self._now_fn(),
                market_id=fill.market_id,
                K=fill.K,
                S_t=fill.S_t,
                tau=fill.tau,
                sigma=fill.sigma,
                q_up=fill.q_up,
                bid_up=fill.bid_up,
                ask_up=fill.ask_up,
                bid_down=fill.bid_down,
                ask_down=fill.ask_down,
                edge=fill.edge,
                order_id=fill.order_id,
                side=fill.side,
                price=fill.price,
                size=fill.size,
                status="paper_settle",
                settlement_outcome=outcome,
                pnl=pnl,
            )
            self._reporter.append(record)
            appended_records.append(record)

        return appended_records


class PolymarketMarketResolutionResolver:
    """Resolves Up/Down winner from Gamma + CLOB market APIs."""

    def __init__(
        self,
        *,
        gamma_base_url: str,
        clob_rest_base_url: str,
        http_client: AsyncJsonHttpClient | None = None,
    ) -> None:
        self._gamma_base_url = gamma_base_url.rstrip("/")
        self._clob_rest_base_url = clob_rest_base_url.rstrip("/")
        self._http = http_client or AsyncJsonHttpClient()

    async def resolve_market(self, market_id: str) -> MarketResolution:
        gamma_payload = await self._http.get_json(
            f"{self._gamma_base_url}/markets",
            params={"id": market_id},
        )
        market_rows = _extract_markets(gamma_payload)
        if not market_rows:
            return MarketResolution(winner=None, resolved=False)

        market = market_rows[0]
        condition_id = str(market.get("conditionId") or market.get("condition_id") or "").strip()
        if not condition_id:
            return MarketResolution(winner=None, resolved=False)

        clob_payload = await self._http.get_json(f"{self._clob_rest_base_url}/markets/{condition_id}")
        winner = _extract_token_winner(clob_payload)
        if winner is not None:
            return MarketResolution(winner=winner, resolved=True)

        if _coerce_bool(clob_payload.get("closed"), default=False):
            fallback_winner = _infer_winner_from_terminal_prices(clob_payload.get("tokens"))
            if fallback_winner is not None:
                return MarketResolution(winner=fallback_winner, resolved=True)

        return MarketResolution(winner=None, resolved=False)


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8") as fp:
        return [dict(row) for row in csv.DictReader(fp)]


def _extract_unsettled_paper_fills(rows: list[dict[str, str]]) -> list[_PaperFill]:
    settled_order_ids: set[str] = set()
    for row in rows:
        status = _normalize_status(row.get("status"))
        if status != "paper_settle":
            continue
        order_id = _normalize_order_id(row.get("order_id"))
        if order_id:
            settled_order_ids.add(order_id)

    fills: list[_PaperFill] = []
    for row in rows:
        status = _normalize_status(row.get("status"))
        if status != "paper_fill":
            continue

        order_id = _normalize_order_id(row.get("order_id"))
        market_id = str(row.get("market_id") or "").strip()
        side = _normalize_side_label(row.get("side"))
        price = _parse_float(row.get("price"))
        size = _parse_float(row.get("size"))
        fill_ts = _parse_datetime(row.get("timestamp"))

        if not order_id or order_id in settled_order_ids:
            continue
        if not market_id or side not in {"up", "down"}:
            continue
        if price is None or price <= 0.0:
            continue
        if size is None or size <= 0.0:
            continue
        if fill_ts is None:
            continue

        fills.append(
            _PaperFill(
                timestamp=fill_ts,
                market_id=market_id,
                order_id=order_id,
                side=side,
                price=price,
                size=size,
                K=_parse_float(row.get("K")) or 0.0,
                S_t=_parse_float(row.get("S_t")) or 0.0,
                tau=_parse_float(row.get("tau")) or 0.0,
                sigma=_parse_float(row.get("sigma")) or 0.0,
                q_up=_parse_float(row.get("q_up")) or 0.5,
                bid_up=_parse_float(row.get("bid_up")) or 0.0,
                ask_up=_parse_float(row.get("ask_up")) or 0.0,
                bid_down=_parse_float(row.get("bid_down")) or 0.0,
                ask_down=_parse_float(row.get("ask_down")) or 0.0,
                edge=_parse_float(row.get("edge")) or 0.0,
            )
        )

    fills.sort(key=lambda value: (value.timestamp, value.order_id))
    return fills


def _paper_binary_pnl(
    *,
    entry_price: float,
    size: float,
    position_side: str,
    winner_side: str,
) -> float:
    stake = entry_price * size
    if position_side == winner_side:
        return size * (1.0 - entry_price)
    return -stake


def _extract_markets(payload: object) -> list[dict[str, object]]:
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]
    if isinstance(payload, dict):
        for key in ("markets", "data", "results"):
            value = payload.get(key)
            if isinstance(value, list):
                return [item for item in value if isinstance(item, dict)]
    return []


def _extract_token_winner(payload: object) -> str | None:
    if not isinstance(payload, dict):
        return None
    tokens = payload.get("tokens")
    if not isinstance(tokens, list):
        return None
    for token in tokens:
        if not isinstance(token, dict):
            continue
        if token.get("winner") is True:
            outcome = _normalize_side_label(token.get("outcome"))
            if outcome in {"up", "down"}:
                return outcome
    return None


def _infer_winner_from_terminal_prices(tokens: object) -> str | None:
    if not isinstance(tokens, list):
        return None
    best_price = -1.0
    best_outcome: str | None = None
    for token in tokens:
        if not isinstance(token, dict):
            continue
        outcome = _normalize_side_label(token.get("outcome"))
        if outcome not in {"up", "down"}:
            continue
        price = _parse_float(token.get("price"))
        if price is None:
            continue
        if price > best_price:
            best_price = price
            best_outcome = outcome
    if best_price >= 0.999:
        return best_outcome
    return None


def _normalize_status(value: object) -> str:
    if value is None:
        return ""
    return str(value).strip().lower()


def _normalize_order_id(value: object) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _normalize_side_label(value: object) -> str:
    if value is None:
        return ""
    raw = str(value).strip().lower()
    if raw in {"up", "buy", "yes", "won", "win"}:
        return "up"
    if raw in {"down", "sell", "no", "lost", "loss"}:
        return "down"
    return raw


def _parse_datetime(value: object) -> datetime | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        normalized = text.replace("Z", "+00:00")
        return datetime.fromisoformat(normalized)
    except ValueError:
        return None


def _parse_float(value: object) -> float | None:
    if value is None:
        return None
    if isinstance(value, (float, int)):
        return float(value)
    text = str(value).strip()
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def _coerce_bool(value: object, *, default: bool) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "1", "yes", "on"}:
            return True
        if lowered in {"false", "0", "no", "off"}:
            return False
    if isinstance(value, (int, float)):
        return value != 0
    return default
