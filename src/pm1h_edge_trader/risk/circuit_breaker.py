from __future__ import annotations

from dataclasses import dataclass

from ..execution.types import KillSwitchReason


@dataclass(frozen=True)
class CircuitBreakerSnapshot:
    daily_realized_pnl: float
    max_daily_loss: float
    decision_bankroll: float
    market_notional: float
    max_market_notional: float
    worst_case_loss: float = 0.0
    max_worst_case_loss: float = 0.0
    epsilon: float = 1e-9


@dataclass(frozen=True)
class CircuitBreakerResult:
    triggered: bool
    reasons: tuple[KillSwitchReason, ...]
    details: dict[str, float]


def evaluate_circuit_breaker(snapshot: CircuitBreakerSnapshot) -> CircuitBreakerResult:
    reasons: list[KillSwitchReason] = []
    details = {
        "daily_realized_pnl": float(snapshot.daily_realized_pnl),
        "max_daily_loss": float(snapshot.max_daily_loss),
        "decision_bankroll": float(snapshot.decision_bankroll),
        "market_notional": float(snapshot.market_notional),
        "max_market_notional": float(snapshot.max_market_notional),
        "worst_case_loss": float(snapshot.worst_case_loss),
        "max_worst_case_loss": float(snapshot.max_worst_case_loss),
    }
    epsilon = max(0.0, float(snapshot.epsilon))

    if snapshot.max_daily_loss > 0.0 and snapshot.daily_realized_pnl <= (-snapshot.max_daily_loss - epsilon):
        reasons.append(KillSwitchReason.DAILY_LOSS_LIMIT)

    if snapshot.decision_bankroll <= epsilon:
        reasons.append(KillSwitchReason.BANKROLL_DEPLETED)

    if snapshot.max_market_notional > 0.0 and snapshot.market_notional > (snapshot.max_market_notional + epsilon):
        reasons.append(KillSwitchReason.MARKET_NOTIONAL_LIMIT_BREACH)

    if snapshot.max_worst_case_loss > 0.0 and snapshot.worst_case_loss > (snapshot.max_worst_case_loss + epsilon):
        reasons.append(KillSwitchReason.WORST_CASE_LOSS_LIMIT)

    return CircuitBreakerResult(
        triggered=bool(reasons),
        reasons=tuple(reasons),
        details=details,
    )
