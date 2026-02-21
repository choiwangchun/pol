from __future__ import annotations

from dataclasses import dataclass

from .types import ExecutionConfig, KillSwitchReason, SafetySurface


@dataclass(frozen=True)
class SafetyCheckResult:
    reasons: tuple[KillSwitchReason, ...]

    @property
    def triggered(self) -> bool:
        return bool(self.reasons)


class SafetyGuard:
    """Evaluates pre-trade runtime safety conditions."""

    def __init__(self, config: ExecutionConfig) -> None:
        self._config = config

    def evaluate(self, safety: SafetySurface) -> SafetyCheckResult:
        reasons: list[KillSwitchReason] = []

        if self._config.fee_must_be_zero and safety.fee_rate != 0.0:
            reasons.append(KillSwitchReason.FEE_NON_ZERO)

        if safety.data_age_s > self._config.max_data_age_s:
            reasons.append(KillSwitchReason.DATA_STALE)

        if abs(safety.clock_drift_s) > self._config.max_clock_drift_s:
            reasons.append(KillSwitchReason.CLOCK_DRIFT)

        if self._config.require_book_match and not safety.book_is_consistent:
            reasons.append(KillSwitchReason.BOOK_MISMATCH)

        return SafetyCheckResult(reasons=tuple(reasons))
