from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any


@dataclass(frozen=True)
class ExecutionLogRecord:
    timestamp: datetime
    market_id: str
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
    order_id: str
    side: str
    price: float
    size: float
    status: str
    settlement_outcome: str | None = None
    pnl: float | None = None

    @classmethod
    def csv_fields(cls) -> list[str]:
        return [
            "timestamp",
            "market_id",
            "K",
            "S_t",
            "tau",
            "sigma",
            "q_up",
            "bid_up",
            "ask_up",
            "bid_down",
            "ask_down",
            "edge",
            "order_id",
            "side",
            "price",
            "size",
            "status",
            "settlement_outcome",
            "pnl",
        ]

    def as_dict(self) -> dict[str, Any]:
        return {
            "timestamp": self.timestamp.isoformat(),
            "market_id": self.market_id,
            "K": self.K,
            "S_t": self.S_t,
            "tau": self.tau,
            "sigma": self.sigma,
            "q_up": self.q_up,
            "bid_up": self.bid_up,
            "ask_up": self.ask_up,
            "bid_down": self.bid_down,
            "ask_down": self.ask_down,
            "edge": self.edge,
            "order_id": self.order_id,
            "side": self.side,
            "price": self.price,
            "size": self.size,
            "status": self.status,
            "settlement_outcome": self.settlement_outcome,
            "pnl": self.pnl,
        }
