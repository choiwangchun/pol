from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Optional, Sequence


def utc_now() -> datetime:
    return datetime.now(tz=timezone.utc)


class Side(str, Enum):
    BUY = "buy"
    SELL = "sell"


class VenueOrderSide(str, Enum):
    BUY = "BUY"
    SELL = "SELL"


class OrderStatus(str, Enum):
    OPEN = "open"
    CANCELED = "canceled"
    FILLED = "filled"
    REPLACED = "replaced"
    REJECTED = "rejected"


class ExecutionActionType(str, Enum):
    PLACE = "place"
    CANCEL = "cancel"
    REQUOTE = "requote"
    SKIP = "skip"
    KILL_SWITCH_TRIGGERED = "kill_switch_triggered"


class KillSwitchReason(str, Enum):
    FEE_NON_ZERO = "fee_non_zero"
    DATA_STALE = "data_stale"
    CLOCK_DRIFT = "clock_drift"
    BOOK_MISMATCH = "book_mismatch"
    HEARTBEAT_FAILED = "heartbeat_failed"
    ORDER_RECONCILIATION_FAILED = "order_reconciliation_failed"


@dataclass(frozen=True)
class SafetySurface:
    fee_rate: float
    data_age_s: float
    clock_drift_s: float
    book_is_consistent: bool


@dataclass(frozen=True)
class IntentSignal:
    market_id: str
    token_id: str
    side: Side
    edge: float
    min_edge: float
    desired_price: Optional[float]
    size: float
    seconds_to_expiry: float
    signal_ts: datetime
    allow_entry: bool = True
    order_side: VenueOrderSide = VenueOrderSide.BUY


@dataclass(frozen=True)
class OrderRequest:
    market_id: str
    token_id: str
    side: Side
    price: float
    size: float
    submitted_at: datetime
    order_side: VenueOrderSide = VenueOrderSide.BUY


@dataclass(frozen=True)
class OrderHandle:
    order_id: str
    status: OrderStatus
    acknowledged_at: datetime


@dataclass(frozen=True)
class CancelResult:
    order_id: str
    canceled: bool
    acknowledged_at: datetime


@dataclass
class ActiveIntent:
    order_id: str
    market_id: str
    token_id: str
    side: Side
    price: float
    size: float
    edge: float
    created_at: datetime
    last_quoted_at: datetime
    status: OrderStatus = OrderStatus.OPEN


@dataclass(frozen=True)
class ExecutionAction:
    action_type: ExecutionActionType
    market_id: str
    side: Optional[Side] = None
    order_id: Optional[str] = None
    reason: Optional[str] = None


@dataclass
class KillSwitchState:
    active: bool = False
    reasons: Sequence[KillSwitchReason] = field(default_factory=tuple)
    activated_at: Optional[datetime] = None


@dataclass
class ExecutionConfig:
    requote_interval_s: float = 5.0
    max_intent_age_s: float = 20.0
    entry_block_window_s: float = 60.0
    max_data_age_s: float = 3.0
    max_clock_drift_s: float = 1.0
    require_book_match: bool = True
    fee_must_be_zero: bool = True
    max_entries_per_market: int = 1
    price_epsilon: float = 1e-9
    latch_kill_switch: bool = True


@dataclass
class ExecutionResult:
    actions: list[ExecutionAction]
    kill_switch: KillSwitchState

    @property
    def kill_switch_active(self) -> bool:
        return self.kill_switch.active
