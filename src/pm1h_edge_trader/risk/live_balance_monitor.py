from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import Callable

from ..config import PolymarketLiveAuthConfig
from ..live_balance import LiveBalanceSnapshot, fetch_live_balance_snapshot


@dataclass(frozen=True)
class LiveBalanceBaseline:
    bankroll: float
    started_at: datetime
    day: date


@dataclass(frozen=True)
class LiveBalanceCheck:
    snapshot: LiveBalanceSnapshot
    baseline: LiveBalanceBaseline
    drawdown: float
    triggered: bool
    checked_at: datetime


class LiveBalanceMonitor:
    def __init__(
        self,
        *,
        auth: PolymarketLiveAuthConfig,
        host: str,
        refresh_seconds: float,
        max_drawdown: float,
        fetch_fn: Callable[..., LiveBalanceSnapshot] = fetch_live_balance_snapshot,
        adapter_factory: Callable[..., object] | None = None,
    ) -> None:
        self._auth = auth
        self._host = host
        self._refresh_seconds = max(1.0, float(refresh_seconds))
        self._max_drawdown = max(0.0, float(max_drawdown))
        self._fetch_fn = fetch_fn
        self._adapter_factory = adapter_factory
        self._baseline: LiveBalanceBaseline | None = None
        self._next_refresh_at: datetime | None = None

    def maybe_refresh(self, *, now: datetime) -> LiveBalanceCheck | None:
        if self._next_refresh_at is not None and now < self._next_refresh_at:
            return None
        kwargs = {
            "auth": self._auth,
            "host": self._host,
        }
        if self._adapter_factory is not None:
            kwargs["adapter_factory"] = self._adapter_factory
        snapshot = self._fetch_fn(**kwargs)
        baseline = self._baseline
        if baseline is None or baseline.day != now.date():
            baseline = LiveBalanceBaseline(
                bankroll=max(0.0, float(snapshot.bankroll)),
                started_at=now,
                day=now.date(),
            )
            self._baseline = baseline
        drawdown = self.drawdown(current_bankroll=snapshot.bankroll)
        triggered = self.should_trigger(drawdown=drawdown, max_drawdown=self._max_drawdown)
        self._next_refresh_at = now + timedelta(seconds=self._refresh_seconds)
        return LiveBalanceCheck(
            snapshot=snapshot,
            baseline=baseline,
            drawdown=drawdown,
            triggered=triggered,
            checked_at=now,
        )

    def drawdown(self, *, current_bankroll: float) -> float:
        baseline = self._baseline
        if baseline is None:
            return 0.0
        current = max(0.0, float(current_bankroll))
        return max(0.0, baseline.bankroll - current)

    @staticmethod
    def should_trigger(*, drawdown: float, max_drawdown: float) -> bool:
        return max_drawdown > 0.0 and drawdown > max_drawdown
