from __future__ import annotations

import csv
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class TradeSummary:
    trade_count: int
    win_count: int
    loss_count: int
    hit_rate: float
    total_pnl: float


def summarize_csv(path: str | Path) -> TradeSummary:
    csv_path = Path(path)
    if not csv_path.exists():
        return TradeSummary(trade_count=0, win_count=0, loss_count=0, hit_rate=0.0, total_pnl=0.0)

    with csv_path.open("r", newline="", encoding="utf-8") as fp:
        rows = list(csv.DictReader(fp))
    return _summarize_rows(rows)


def summarize_sqlite(db_path: str | Path, *, table_name: str = "execution_logs") -> TradeSummary:
    db = Path(db_path)
    if not db.exists():
        return TradeSummary(trade_count=0, win_count=0, loss_count=0, hit_rate=0.0, total_pnl=0.0)

    table = _validate_sql_identifier(table_name)
    conn = sqlite3.connect(db)
    try:
        cursor = conn.execute(
            f"""
            SELECT settlement_outcome, pnl
            FROM {table}
            """
        )
        rows = [
            {
                "settlement_outcome": settlement_outcome,
                "pnl": pnl,
            }
            for settlement_outcome, pnl in cursor.fetchall()
        ]
    finally:
        conn.close()

    return _summarize_rows(rows)


def _summarize_rows(rows: Iterable[dict[str, object]]) -> TradeSummary:
    trade_count = 0
    win_count = 0
    loss_count = 0
    total_pnl = 0.0

    for row in rows:
        pnl = _parse_float(row.get("pnl"))
        outcome = _normalize_outcome(row.get("settlement_outcome"))

        has_settlement = pnl is not None or bool(outcome)
        if not has_settlement:
            continue

        trade_count += 1

        if pnl is not None:
            total_pnl += pnl
            if pnl > 0:
                win_count += 1
            elif pnl < 0:
                loss_count += 1
        else:
            if outcome in {"win", "won", "yes", "up"}:
                win_count += 1
            elif outcome in {"loss", "lost", "no", "down"}:
                loss_count += 1

    hit_rate = (win_count / trade_count) if trade_count else 0.0
    return TradeSummary(
        trade_count=trade_count,
        win_count=win_count,
        loss_count=loss_count,
        hit_rate=hit_rate,
        total_pnl=total_pnl,
    )


def _parse_float(value: object) -> float | None:
    if value is None:
        return None
    if isinstance(value, float):
        return value
    if isinstance(value, int):
        return float(value)

    text = str(value).strip()
    if not text:
        return None

    try:
        return float(text)
    except ValueError:
        return None


def _normalize_outcome(value: object) -> str:
    if value is None:
        return ""
    return str(value).strip().lower()


def _validate_sql_identifier(identifier: str) -> str:
    if not identifier.replace("_", "").isalnum() or identifier[0].isdigit():
        raise ValueError(f"invalid SQL identifier: {identifier}")
    return identifier
