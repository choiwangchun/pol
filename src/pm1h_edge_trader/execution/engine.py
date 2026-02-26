from __future__ import annotations

from datetime import datetime
from typing import Callable

from .adapters import ExecutionAdapter
from .safety import SafetyGuard
from .types import (
    ActiveIntent,
    ExecutionAction,
    ExecutionActionType,
    ExecutionConfig,
    ExecutionResult,
    IntentSignal,
    KillSwitchReason,
    KillSwitchState,
    OrderRequest,
    OrderStatus,
    SafetySurface,
    utc_now,
)


class LimitOrderExecutionEngine:
    """Stateful lifecycle manager for limit-order intents."""

    def __init__(
        self,
        *,
        adapter: ExecutionAdapter,
        config: ExecutionConfig | None = None,
        now_fn: Callable[[], datetime] = utc_now,
    ) -> None:
        self._adapter = adapter
        self._config = config or ExecutionConfig()
        self._now_fn = now_fn
        self._safety_guard = SafetyGuard(self._config)
        self._kill_switch = KillSwitchState()
        self._active_intents: dict[tuple[str, str], ActiveIntent] = {}
        self._filled_entries_by_market: dict[str, int] = {}
        self._counted_filled_orders: set[str] = set()

    @property
    def kill_switch(self) -> KillSwitchState:
        return self._kill_switch

    def active_intents(self) -> dict[tuple[str, str], ActiveIntent]:
        return dict(self._active_intents)

    def reset_kill_switch(self) -> None:
        self._kill_switch = KillSwitchState()

    def set_price_epsilon(self, epsilon: float) -> None:
        if epsilon <= 0.0:
            return
        self._config.price_epsilon = epsilon

    def drop_intent(self, order_id: str) -> bool:
        """Drops an active intent without counting it as a fill."""
        normalized = order_id.strip()
        if not normalized:
            return False

        for key, active in list(self._active_intents.items()):
            if active.order_id != normalized:
                continue
            self._active_intents.pop(key, None)
            return True
        return False

    def cancel_all_intents(self, *, reason: str) -> list[ExecutionAction]:
        actions: list[ExecutionAction] = []
        for key, active in list(self._active_intents.items()):
            actions.extend(self._cancel_intent(key=key, active=active, reason=reason))
        return actions

    def latch_kill_switch(
        self,
        *,
        reason: KillSwitchReason,
        now: datetime | None = None,
    ) -> ExecutionResult:
        actions = self._activate_kill_switch((reason,), now=now or self._now_fn())
        return ExecutionResult(actions=actions, kill_switch=self._kill_switch)

    def mark_order_filled(self, order_id: str) -> bool:
        """Marks an order as filled and clears its active intent."""
        normalized = order_id.strip()
        if not normalized:
            return False

        marked = False
        mark_filled = getattr(self._adapter, "mark_filled", None)
        if callable(mark_filled):
            marked = bool(mark_filled(normalized))

        for key, active in list(self._active_intents.items()):
            if active.order_id != normalized:
                continue
            active.status = OrderStatus.FILLED
            self._active_intents.pop(key, None)
            if normalized not in self._counted_filled_orders:
                self._filled_entries_by_market[active.market_id] = (
                    self._filled_entries_by_market.get(active.market_id, 0) + 1
                )
                self._counted_filled_orders.add(normalized)
            marked = True

        return marked

    def process_signal(
        self,
        signal: IntentSignal,
        safety: SafetySurface,
        *,
        now: datetime | None = None,
    ) -> ExecutionResult:
        current_time = now or self._now_fn()
        actions: list[ExecutionAction] = []

        safety_result = self._safety_guard.evaluate(safety)
        if safety_result.triggered:
            actions.extend(self._activate_kill_switch(safety_result.reasons, now=current_time))
            return ExecutionResult(actions=actions, kill_switch=self._kill_switch)

        if self._kill_switch.active and self._config.latch_kill_switch:
            actions.append(
                ExecutionAction(
                    action_type=ExecutionActionType.SKIP,
                    market_id=signal.market_id,
                    token_id=signal.token_id,
                    side=signal.side,
                    reason="kill_switch_latched",
                )
            )
            return ExecutionResult(actions=actions, kill_switch=self._kill_switch)

        key = self._intent_key(signal.market_id, signal.token_id)
        active = self._active_intents.get(key)

        if active is not None:
            actions.extend(self._maybe_cancel_or_requote(signal=signal, active=active, now=current_time))
            active = self._active_intents.get(key)

        if active is None:
            actions.extend(self._maybe_place(signal=signal, now=current_time))
        else:
            actions.append(
                ExecutionAction(
                    action_type=ExecutionActionType.SKIP,
                    market_id=signal.market_id,
                    token_id=signal.token_id,
                    side=signal.side,
                    order_id=active.order_id,
                    reason="intent_unchanged",
                )
            )

        return ExecutionResult(actions=actions, kill_switch=self._kill_switch)

    def sweep_stale_intents(self, *, now: datetime | None = None) -> ExecutionResult:
        current_time = now or self._now_fn()
        actions: list[ExecutionAction] = []
        for key, active in list(self._active_intents.items()):
            age_s = (current_time - active.created_at).total_seconds()
            if age_s > self._config.max_intent_age_s:
                actions.extend(self._cancel_intent(key=key, active=active, reason="intent_stale"))
        return ExecutionResult(actions=actions, kill_switch=self._kill_switch)

    def _maybe_cancel_or_requote(
        self,
        *,
        signal: IntentSignal,
        active: ActiveIntent,
        now: datetime,
    ) -> list[ExecutionAction]:
        actions: list[ExecutionAction] = []
        key = self._intent_key(signal.market_id, signal.token_id)

        if (
            signal.edge < signal.min_edge
            or not signal.allow_entry
            or signal.desired_price is None
            or signal.size <= 0.0
        ):
            actions.extend(self._cancel_intent(key=key, active=active, reason="edge_disappeared"))
            return actions

        age_s = (now - active.created_at).total_seconds()
        if age_s > self._config.max_intent_age_s:
            actions.extend(self._cancel_intent(key=key, active=active, reason="intent_stale"))
            return actions

        time_since_quote_s = (now - active.last_quoted_at).total_seconds()
        should_requote = time_since_quote_s >= self._config.requote_interval_s
        price_changed = abs(signal.desired_price - active.price) > self._config.price_epsilon
        size_changed = abs(signal.size - active.size) > self._config.price_epsilon

        if should_requote and (price_changed or size_changed):
            actions.append(
                ExecutionAction(
                    action_type=ExecutionActionType.REQUOTE,
                    market_id=signal.market_id,
                    token_id=signal.token_id,
                    side=signal.side,
                    order_id=active.order_id,
                    reason="requote_interval_elapsed",
                )
            )
            actions.extend(self._cancel_intent(key=key, active=active, reason="requote"))

        return actions

    def _maybe_place(self, *, signal: IntentSignal, now: datetime) -> list[ExecutionAction]:
        entry_limit = self._config.max_entries_per_market
        filled_entries = self._filled_entries_by_market.get(signal.market_id, 0)
        if entry_limit > 0 and filled_entries >= entry_limit:
            return [
                ExecutionAction(
                    action_type=ExecutionActionType.SKIP,
                    market_id=signal.market_id,
                    token_id=signal.token_id,
                    side=signal.side,
                    reason="market_entry_limit",
                )
            ]

        if signal.size <= 0.0:
            return [
                ExecutionAction(
                    action_type=ExecutionActionType.SKIP,
                    market_id=signal.market_id,
                    token_id=signal.token_id,
                    side=signal.side,
                    reason="non_positive_size",
                )
            ]

        if signal.edge < signal.min_edge:
            return [
                ExecutionAction(
                    action_type=ExecutionActionType.SKIP,
                    market_id=signal.market_id,
                    token_id=signal.token_id,
                    side=signal.side,
                    reason="edge_below_threshold",
                )
            ]

        if not signal.allow_entry:
            return [
                ExecutionAction(
                    action_type=ExecutionActionType.SKIP,
                    market_id=signal.market_id,
                    token_id=signal.token_id,
                    side=signal.side,
                    reason="entry_not_allowed",
                )
            ]

        if signal.seconds_to_expiry <= self._config.entry_block_window_s:
            return [
                ExecutionAction(
                    action_type=ExecutionActionType.SKIP,
                    market_id=signal.market_id,
                    token_id=signal.token_id,
                    side=signal.side,
                    reason="near_expiry_entry_block",
                )
            ]

        if signal.desired_price is None:
            return [
                ExecutionAction(
                    action_type=ExecutionActionType.SKIP,
                    market_id=signal.market_id,
                    token_id=signal.token_id,
                    side=signal.side,
                    reason="missing_desired_price",
                )
            ]
        if signal.desired_price <= 0.0 or signal.desired_price >= 1.0:
            return [
                ExecutionAction(
                    action_type=ExecutionActionType.SKIP,
                    market_id=signal.market_id,
                    token_id=signal.token_id,
                    side=signal.side,
                    reason="invalid_desired_price",
                )
            ]

        request = OrderRequest(
            market_id=signal.market_id,
            token_id=signal.token_id,
            side=signal.side,
            price=signal.desired_price,
            size=signal.size,
            submitted_at=now,
            order_side=signal.order_side,
        )
        handle = self._adapter.place_limit_order(request)
        self._active_intents[self._intent_key(signal.market_id, signal.token_id)] = ActiveIntent(
            order_id=handle.order_id,
            market_id=signal.market_id,
            token_id=signal.token_id,
            side=signal.side,
            price=signal.desired_price,
            size=signal.size,
            edge=signal.edge,
            created_at=now,
            last_quoted_at=now,
            order_side=signal.order_side,
            status=OrderStatus.OPEN,
        )
        return [
            ExecutionAction(
                action_type=ExecutionActionType.PLACE,
                market_id=signal.market_id,
                token_id=signal.token_id,
                side=signal.side,
                order_id=handle.order_id,
                reason="new_intent",
            )
        ]

    def _cancel_intent(
        self,
        *,
        key: tuple[str, str],
        active: ActiveIntent,
        reason: str,
    ) -> list[ExecutionAction]:
        cancel_result = self._adapter.cancel_order(active.order_id)
        self._active_intents.pop(key, None)
        cancel_reason = reason if cancel_result.canceled else f"{reason}:cancel_not_confirmed"
        return [
            ExecutionAction(
                action_type=ExecutionActionType.CANCEL,
                market_id=active.market_id,
                token_id=active.token_id,
                side=active.side,
                order_id=active.order_id,
                reason=cancel_reason,
            )
        ]

    def _activate_kill_switch(
        self,
        reasons: tuple[KillSwitchReason, ...],
        *,
        now: datetime,
    ) -> list[ExecutionAction]:
        if self._kill_switch.active and self._config.latch_kill_switch:
            return [
                ExecutionAction(
                    action_type=ExecutionActionType.KILL_SWITCH_TRIGGERED,
                    market_id="*",
                    token_id=None,
                    reason="kill_switch_already_latched",
                )
            ]

        self._kill_switch = KillSwitchState(active=True, reasons=reasons, activated_at=now)
        actions = [
            ExecutionAction(
                action_type=ExecutionActionType.KILL_SWITCH_TRIGGERED,
                market_id="*",
                token_id=None,
                reason=",".join(reason.value for reason in reasons),
            )
        ]

        for key, active in list(self._active_intents.items()):
            actions.extend(self._cancel_intent(key=key, active=active, reason="kill_switch"))

        return actions

    @staticmethod
    def _intent_key(market_id: str, token_id: str) -> tuple[str, str]:
        return (str(market_id).strip(), str(token_id).strip())
