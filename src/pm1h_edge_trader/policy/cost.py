from __future__ import annotations

from datetime import datetime


class OnlineCostEstimator:
    """EMA-based online execution-cost estimator."""

    def __init__(
        self,
        *,
        alpha: float = 0.05,
        decay: float = 0.999,
        min_cost: float = 0.0,
        max_cost: float = 0.25,
        ema_cost: float = 0.0,
        ema_slippage: float = 0.0,
        last_update_ts: datetime | None = None,
    ) -> None:
        self._alpha = min(1.0, max(1e-6, float(alpha)))
        self._decay = min(1.0, max(0.0, float(decay)))
        self._min_cost = max(0.0, float(min_cost))
        self._max_cost = max(self._min_cost, float(max_cost))
        self._ema_cost = self._clamp(float(ema_cost))
        self._ema_slippage = self._clamp(float(ema_slippage))
        self._last_update_ts = last_update_ts

    def get(self) -> float:
        return self._ema_cost

    def update(
        self,
        *,
        decision_price: float,
        fill_price: float,
        spread: float,
        matched_size: float,
        ts: datetime | None,
        freeze: bool = False,
    ) -> float:
        if matched_size <= 0.0:
            return self._ema_cost
        if freeze:
            return self._ema_cost

        slippage = abs(float(fill_price) - float(decision_price))
        half_spread = max(0.0, float(spread)) * 0.5
        observed = self._clamp(slippage + half_spread)

        self._ema_slippage = self._ema(self._ema_slippage, self._clamp(slippage))
        self._ema_cost = self._ema(self._ema_cost, observed)
        self._last_update_ts = ts
        return self._ema_cost

    def decay_without_fill(self) -> float:
        self._ema_cost = self._clamp(self._ema_cost * self._decay)
        self._ema_slippage = self._clamp(self._ema_slippage * self._decay)
        return self._ema_cost

    def export_state(self) -> dict[str, float | str | None]:
        return {
            "ema_cost": self._ema_cost,
            "ema_slippage": self._ema_slippage,
            "last_update_ts": self._last_update_ts.isoformat() if self._last_update_ts is not None else None,
            "alpha": self._alpha,
            "decay": self._decay,
            "min_cost": self._min_cost,
            "max_cost": self._max_cost,
        }

    def import_state(self, payload: dict[str, object]) -> None:
        ema_cost = payload.get("ema_cost")
        if isinstance(ema_cost, (int, float)):
            self._ema_cost = self._clamp(float(ema_cost))
        ema_slippage = payload.get("ema_slippage")
        if isinstance(ema_slippage, (int, float)):
            self._ema_slippage = self._clamp(float(ema_slippage))
        raw_ts = payload.get("last_update_ts")
        if isinstance(raw_ts, str) and raw_ts.strip():
            try:
                self._last_update_ts = datetime.fromisoformat(raw_ts.replace("Z", "+00:00"))
            except ValueError:
                self._last_update_ts = None

    def _ema(self, current: float, observed: float) -> float:
        return ((1.0 - self._alpha) * current) + (self._alpha * observed)

    def _clamp(self, value: float) -> float:
        return min(self._max_cost, max(self._min_cost, value))

