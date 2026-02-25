from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CompleteSetArbDecision:
    should_trade: bool
    reason: str
    ask_sum: float
    pair_cost: float
    profit_per_pair: float
    pair_tokens: float
    notional_up: float
    notional_down: float
    total_notional: float


def decide_complete_set_arb(
    *,
    ask_up: float,
    ask_down: float,
    bankroll: float,
    min_profit: float,
    max_notional: float,
    min_order_notional: float,
) -> CompleteSetArbDecision:
    up = float(ask_up)
    down = float(ask_down)
    ask_sum = up + down
    if up <= 0.0 or down <= 0.0:
        return _no_trade("invalid_ask", ask_sum)
    if ask_sum >= 1.0:
        return _no_trade("ask_sum_not_profitable", ask_sum)

    profit_per_pair = 1.0 - ask_sum
    if profit_per_pair < max(0.0, float(min_profit)):
        return _no_trade("min_profit_not_met", ask_sum, profit_per_pair=profit_per_pair)

    budget = min(max(0.0, float(bankroll)), max(0.0, float(max_notional)))
    if budget <= 0.0:
        return _no_trade("budget_non_positive", ask_sum, profit_per_pair=profit_per_pair)

    pair_tokens = budget / ask_sum
    notional_up = pair_tokens * up
    notional_down = pair_tokens * down
    total_notional = notional_up + notional_down
    min_notional = max(0.0, float(min_order_notional))
    if notional_up < min_notional or notional_down < min_notional:
        return _no_trade(
            "min_order_notional_not_met",
            ask_sum,
            profit_per_pair=profit_per_pair,
        )

    return CompleteSetArbDecision(
        should_trade=True,
        reason="arb_opportunity",
        ask_sum=ask_sum,
        pair_cost=ask_sum,
        profit_per_pair=profit_per_pair,
        pair_tokens=pair_tokens,
        notional_up=notional_up,
        notional_down=notional_down,
        total_notional=total_notional,
    )


def _no_trade(reason: str, ask_sum: float, *, profit_per_pair: float = 0.0) -> CompleteSetArbDecision:
    return CompleteSetArbDecision(
        should_trade=False,
        reason=reason,
        ask_sum=ask_sum,
        pair_cost=ask_sum,
        profit_per_pair=profit_per_pair,
        pair_tokens=0.0,
        notional_up=0.0,
        notional_down=0.0,
        total_notional=0.0,
    )
