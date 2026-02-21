from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Callable, Protocol

from .types import CancelResult, OrderHandle, OrderRequest, OrderStatus, utc_now


class ExecutionAdapter(Protocol):
    def place_limit_order(self, request: OrderRequest) -> OrderHandle:
        """Submit a limit order through a venue adapter."""

    def cancel_order(self, order_id: str) -> CancelResult:
        """Cancel a previously submitted order."""


@dataclass
class _DryRunOrder:
    request: OrderRequest
    status: OrderStatus


class DryRunExecutionAdapter(ExecutionAdapter):
    """In-memory adapter for simulation and integration tests."""

    def __init__(self, *, id_prefix: str = "dry", now_fn: Callable[[], datetime] = utc_now) -> None:
        self._id_prefix = id_prefix
        self._now_fn = now_fn
        self._counter = 0
        self._orders: dict[str, _DryRunOrder] = {}

    def place_limit_order(self, request: OrderRequest) -> OrderHandle:
        self._counter += 1
        order_id = f"{self._id_prefix}-{self._counter}"
        self._orders[order_id] = _DryRunOrder(request=request, status=OrderStatus.OPEN)
        return OrderHandle(order_id=order_id, status=OrderStatus.OPEN, acknowledged_at=self._now_fn())

    def cancel_order(self, order_id: str) -> CancelResult:
        order = self._orders.get(order_id)
        if order is None or order.status != OrderStatus.OPEN:
            return CancelResult(order_id=order_id, canceled=False, acknowledged_at=self._now_fn())
        order.status = OrderStatus.CANCELED
        return CancelResult(order_id=order_id, canceled=True, acknowledged_at=self._now_fn())

    def mark_filled(self, order_id: str) -> bool:
        order = self._orders.get(order_id)
        if order is None or order.status != OrderStatus.OPEN:
            return False
        order.status = OrderStatus.FILLED
        return True

    def order_status(self, order_id: str) -> OrderStatus | None:
        order = self._orders.get(order_id)
        if order is None:
            return None
        return order.status


PlaceLimitOrderFn = Callable[[OrderRequest], OrderHandle]
CancelOrderFn = Callable[[str], CancelResult]


class LiveExecutionAdapter(ExecutionAdapter):
    """Thin wrapper over real broker/venue callables."""

    def __init__(
        self,
        *,
        place_limit_order_fn: PlaceLimitOrderFn,
        cancel_order_fn: CancelOrderFn,
    ) -> None:
        self._place_limit_order_fn = place_limit_order_fn
        self._cancel_order_fn = cancel_order_fn

    def place_limit_order(self, request: OrderRequest) -> OrderHandle:
        return self._place_limit_order_fn(request)

    def cancel_order(self, order_id: str) -> CancelResult:
        return self._cancel_order_fn(order_id)
