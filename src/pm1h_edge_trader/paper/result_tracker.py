from __future__ import annotations

import csv
import json
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Iterable

from ..logger.models import ExecutionLogRecord


def utc_now() -> datetime:
    return datetime.now(tz=timezone.utc)


class PaperResultTracker:
    """Maintains paper vault state and persists it as JSON."""

    def __init__(
        self,
        *,
        initial_bankroll: float,
        result_json_path: str | Path,
        now_fn=utc_now,
    ) -> None:
        self._initial_bankroll = max(0.0, float(initial_bankroll))
        self._result_json_path = Path(result_json_path)
        self._now_fn = now_fn
        self._reset_state()

    def bootstrap_from_csv(self, csv_path: str | Path) -> None:
        self._reset_state()
        path = Path(csv_path)
        if not path.exists():
            self.write_snapshot()
            return
        with path.open("r", newline="", encoding="utf-8") as fp:
            rows = list(csv.DictReader(fp))
        for record in _records_from_rows(rows):
            self._apply_record(record)
        self.write_snapshot()

    def on_record(self, record: ExecutionLogRecord) -> None:
        changed = self._apply_record(record)
        if changed:
            self.write_snapshot(last_event=record)

    def write_snapshot(self, *, last_event: ExecutionLogRecord | None = None) -> None:
        unsettled_notional = self.unsettled_notional()
        settled = self._settled_trades
        win_rate = (self._wins / settled) if settled > 0 else 0.0

        payload = {
            "updated_at": self._now_fn().isoformat(),
            "initial_bankroll": round(self._initial_bankroll, 8),
            "current_balance": round(self.current_balance(), 8),
            "realized_pnl": round(self._realized_pnl, 8),
            "daily_realized_pnl": round(self.daily_realized_pnl(), 8),
            "settled_trades": self._settled_trades,
            "wins": self._wins,
            "losses": self._losses,
            "win_rate": round(win_rate, 8),
            "unsettled_fills": len(self._unsettled_fills),
            "unsettled_notional": round(unsettled_notional, 8),
            "available_bankroll": round(self.available_bankroll(), 8),
            "last_event": _serialize_event(last_event) if last_event is not None else None,
        }

        self._result_json_path.parent.mkdir(parents=True, exist_ok=True)
        self._result_json_path.write_text(
            json.dumps(payload, ensure_ascii=True, indent=2),
            encoding="utf-8",
        )

    def current_balance(self) -> float:
        return self._initial_bankroll + self._realized_pnl

    def unsettled_notional(self) -> float:
        return sum(self._unsettled_fills.values())

    def available_bankroll(self) -> float:
        return max(0.0, self.current_balance() - self.unsettled_notional())

    def market_unsettled_notional(self, market_id: str) -> float:
        key = str(market_id).strip()
        if not key:
            return 0.0
        return max(0.0, self._unsettled_notional_by_market.get(key, 0.0))

    def market_entry_count(self, market_id: str) -> int:
        key = str(market_id).strip()
        if not key:
            return 0
        return max(0, self._market_entry_counts.get(key, 0))

    def daily_realized_pnl(self, as_of: datetime | date | None = None) -> float:
        day = _normalize_day(as_of) if as_of is not None else _normalize_day(self._now_fn())
        return self._realized_pnl_by_day.get(day.isoformat(), 0.0)

    def _reset_state(self) -> None:
        self._realized_pnl = 0.0
        self._wins = 0
        self._losses = 0
        self._settled_trades = 0
        self._unsettled_fills: dict[str, float] = {}
        self._unsettled_notional_by_market: dict[str, float] = {}
        self._order_market_ids: dict[str, str] = {}
        self._market_entry_counts: dict[str, int] = {}
        self._realized_pnl_by_day: dict[str, float] = {}
        self._settled_order_ids: set[str] = set()

    def _apply_record(self, record: ExecutionLogRecord) -> bool:
        status = _normalize_status(record.status)
        order_id = (record.order_id or "").strip()

        if status == "paper_fill":
            if not order_id or order_id == "-" or order_id in self._settled_order_ids:
                return False
            if order_id in self._unsettled_fills:
                return False
            market_id = str(record.market_id or "").strip()
            if not market_id:
                return False
            notional = max(0.0, float(record.price) * float(record.size))
            self._unsettled_fills[order_id] = notional
            self._order_market_ids[order_id] = market_id
            self._unsettled_notional_by_market[market_id] = (
                self._unsettled_notional_by_market.get(market_id, 0.0) + notional
            )
            self._market_entry_counts[market_id] = self._market_entry_counts.get(market_id, 0) + 1
            return True

        if status == "paper_settle":
            if not order_id or order_id == "-" or order_id in self._settled_order_ids:
                return False
            self._settled_order_ids.add(order_id)
            unsettled_notional = self._unsettled_fills.pop(order_id, None)
            market_id = self._order_market_ids.pop(order_id, "")
            if unsettled_notional is not None and market_id:
                remaining = self._unsettled_notional_by_market.get(market_id, 0.0) - unsettled_notional
                if remaining > 0.0:
                    self._unsettled_notional_by_market[market_id] = remaining
                else:
                    self._unsettled_notional_by_market.pop(market_id, None)

            pnl = float(record.pnl) if record.pnl is not None else 0.0
            self._realized_pnl += pnl
            day_key = _normalize_day(record.timestamp).isoformat()
            self._realized_pnl_by_day[day_key] = self._realized_pnl_by_day.get(day_key, 0.0) + pnl
            self._settled_trades += 1

            outcome = _normalize_outcome(record.settlement_outcome)
            if outcome == "win":
                self._wins += 1
            elif outcome == "loss":
                self._losses += 1
            else:
                if pnl > 0:
                    self._wins += 1
                elif pnl < 0:
                    self._losses += 1
            return True

        return False


def _records_from_rows(rows: Iterable[dict[str, str]]) -> list[ExecutionLogRecord]:
    parsed: list[ExecutionLogRecord] = []
    for row in rows:
        status = _normalize_status(row.get("status"))
        if status not in {"paper_fill", "paper_settle"}:
            continue
        timestamp = _parse_datetime(row.get("timestamp")) or utc_now()
        record = ExecutionLogRecord(
            timestamp=timestamp,
            market_id=str(row.get("market_id") or ""),
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
            order_id=str(row.get("order_id") or "-"),
            side=str(row.get("side") or ""),
            price=_parse_float(row.get("price")) or 0.0,
            size=_parse_float(row.get("size")) or 0.0,
            status=status,
            settlement_outcome=_normalize_optional_text(row.get("settlement_outcome")),
            pnl=_parse_float(row.get("pnl")),
        )
        parsed.append(record)
    return parsed


def _normalize_status(value: object) -> str:
    if value is None:
        return ""
    return str(value).strip().lower()


def _normalize_outcome(value: object) -> str:
    if value is None:
        return ""
    text = str(value).strip().lower()
    if text in {"win", "won", "up", "yes"}:
        return "win"
    if text in {"loss", "lost", "down", "no"}:
        return "loss"
    return ""


def _normalize_optional_text(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _parse_datetime(value: object) -> datetime | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        return datetime.fromisoformat(text.replace("Z", "+00:00"))
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


def _normalize_day(value: datetime | date) -> date:
    if isinstance(value, datetime):
        ts = value
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
        return ts.astimezone(timezone.utc).date()
    return value


def _serialize_event(record: ExecutionLogRecord) -> dict[str, object]:
    return {
        "timestamp": record.timestamp.isoformat(),
        "market_id": record.market_id,
        "order_id": record.order_id,
        "side": record.side,
        "status": record.status,
        "price": record.price,
        "size": record.size,
        "settlement_outcome": record.settlement_outcome,
        "pnl": record.pnl,
    }
