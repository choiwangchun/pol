from __future__ import annotations

import csv
import sqlite3
from pathlib import Path
from typing import Protocol

from .models import ExecutionLogRecord


class ExecutionReporter(Protocol):
    def append(self, record: ExecutionLogRecord) -> None:
        """Persist one execution record."""


class CSVExecutionReporter:
    """Append-only CSV execution logger."""

    def __init__(self, path: str | Path) -> None:
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        if not self._path.exists() or self._path.stat().st_size == 0:
            self._write_header()

    @property
    def path(self) -> Path:
        return self._path

    def append(self, record: ExecutionLogRecord) -> None:
        with self._path.open("a", newline="", encoding="utf-8") as fp:
            writer = csv.DictWriter(fp, fieldnames=ExecutionLogRecord.csv_fields())
            writer.writerow(record.as_dict())

    def _write_header(self) -> None:
        with self._path.open("w", newline="", encoding="utf-8") as fp:
            writer = csv.DictWriter(fp, fieldnames=ExecutionLogRecord.csv_fields())
            writer.writeheader()


class SQLiteExecutionReporter:
    """SQLite execution logger with one normalized events table."""

    def __init__(self, db_path: str | Path, *, table_name: str = "execution_logs") -> None:
        self._db_path = Path(db_path)
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._table_name = _validate_sql_identifier(table_name)
        self._conn = sqlite3.connect(self._db_path)
        self._create_table()

    @property
    def db_path(self) -> Path:
        return self._db_path

    @property
    def table_name(self) -> str:
        return self._table_name

    def append(self, record: ExecutionLogRecord) -> None:
        self._conn.execute(
            f"""
            INSERT INTO {self._table_name} (
                timestamp,
                market_id,
                K,
                S_t,
                tau,
                sigma,
                q_up,
                bid_up,
                ask_up,
                bid_down,
                ask_down,
                edge,
                order_id,
                side,
                price,
                size,
                status,
                settlement_outcome,
                pnl
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                record.timestamp.isoformat(),
                record.market_id,
                record.K,
                record.S_t,
                record.tau,
                record.sigma,
                record.q_up,
                record.bid_up,
                record.ask_up,
                record.bid_down,
                record.ask_down,
                record.edge,
                record.order_id,
                record.side,
                record.price,
                record.size,
                record.status,
                record.settlement_outcome,
                record.pnl,
            ),
        )
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()

    def _create_table(self) -> None:
        self._conn.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {self._table_name} (
                timestamp TEXT NOT NULL,
                market_id TEXT NOT NULL,
                K REAL NOT NULL,
                S_t REAL NOT NULL,
                tau REAL NOT NULL,
                sigma REAL NOT NULL,
                q_up REAL NOT NULL,
                bid_up REAL NOT NULL,
                ask_up REAL NOT NULL,
                bid_down REAL NOT NULL,
                ask_down REAL NOT NULL,
                edge REAL NOT NULL,
                order_id TEXT NOT NULL,
                side TEXT NOT NULL,
                price REAL NOT NULL,
                size REAL NOT NULL,
                status TEXT NOT NULL,
                settlement_outcome TEXT,
                pnl REAL
            )
            """
        )
        self._conn.commit()


def _validate_sql_identifier(identifier: str) -> str:
    if not identifier.replace("_", "").isalnum() or identifier[0].isdigit():
        raise ValueError(f"invalid SQL identifier: {identifier}")
    return identifier
