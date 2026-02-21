"""Core probability, edge, and sizing math for 1h prediction trading."""

from __future__ import annotations

from dataclasses import dataclass
from math import erf, log, sqrt
from typing import Literal

EPSILON = 1e-12


@dataclass(frozen=True)
class EdgeResult:
    edge_up: float
    edge_down: float
    edge_up_after_buffer: float
    edge_down_after_buffer: float


@dataclass(frozen=True)
class TradeIntent:
    side: Literal["up", "down"]
    should_trade: bool
    probability: float
    ask: float
    edge: float
    edge_after_buffer: float
    kelly_fraction: float
    order_size: float
    reason: str


@dataclass(frozen=True)
class DecisionOutcome:
    sigma: float
    q_up: float
    q_down: float
    up: TradeIntent
    down: TradeIntent


@dataclass(frozen=True)
class DecisionConfig:
    sigma_weight: float = 0.5
    edge_min: float = 0.0
    edge_buffer_up: float = 0.0
    edge_buffer_down: float = 0.0
    kelly_fraction: float = 0.5
    f_cap: float = 0.1
    min_order_size: float = 0.0
    epsilon: float = EPSILON


def normal_cdf(x: float) -> float:
    """Standard normal CDF."""
    return 0.5 * (1.0 + erf(x / sqrt(2.0)))


def blended_sigma(rv: float, iv: float | None, w: float) -> float:
    """
    Blend realized and implied volatility:
      sigma^2 = w * rv^2 + (1 - w) * iv^2
    If iv is missing, fall back to rv.
    """
    _validate_non_negative("rv", rv)
    _validate_probability("w", w)
    iv_value = rv if iv is None else iv
    _validate_non_negative("iv", iv_value)

    variance = w * (rv**2) + (1.0 - w) * (iv_value**2)
    # Defensive clamp for potential negative round-off.
    variance = max(0.0, variance)
    return sqrt(variance)


def compute_probabilities(
    spot: float,
    strike: float,
    sigma: float,
    tau: float,
    *,
    epsilon: float = EPSILON,
) -> tuple[float, float]:
    """
    Compute q_up and q_down where:
      z = (ln(S_t/K) - 0.5 * sigma^2 * tau) / (sigma * sqrt(tau))
      q_up = Phi(z)
      q_down = 1 - q_up
    """
    _validate_positive("spot", spot)
    _validate_positive("strike", strike)
    _validate_non_negative("sigma", sigma)
    _validate_non_negative("tau", tau)
    _validate_positive("epsilon", epsilon)

    if tau <= epsilon or sigma <= epsilon:
        q_up = 1.0 if spot >= strike else 0.0
        return q_up, 1.0 - q_up

    denom = sigma * sqrt(tau)
    if denom <= epsilon:
        q_up = 1.0 if spot >= strike else 0.0
        return q_up, 1.0 - q_up

    z = (log(spot / strike) - 0.5 * (sigma**2) * tau) / denom
    q_up = _clamp_unit_interval(normal_cdf(z))
    return q_up, 1.0 - q_up


def evaluate_edges(
    q_up: float,
    q_down: float,
    ask_up: float,
    ask_down: float,
    *,
    buffer_up: float = 0.0,
    buffer_down: float = 0.0,
) -> EdgeResult:
    """Evaluate raw and buffer-adjusted edges."""
    _validate_probability("q_up", q_up)
    _validate_probability("q_down", q_down)
    _validate_probability("ask_up", ask_up)
    _validate_probability("ask_down", ask_down)
    _validate_non_negative("buffer_up", buffer_up)
    _validate_non_negative("buffer_down", buffer_down)

    edge_up = q_up - ask_up
    edge_down = q_down - ask_down
    return EdgeResult(
        edge_up=edge_up,
        edge_down=edge_down,
        edge_up_after_buffer=edge_up - buffer_up,
        edge_down_after_buffer=edge_down - buffer_down,
    )


def fractional_kelly(
    q: float,
    c: float,
    *,
    kelly_fraction: float = 1.0,
    f_cap: float = 1.0,
) -> float:
    """
    Fractional Kelly position fraction:
      f* = (q - c) / (1 - c), if q > c else 0
      f = kelly_fraction * f*
    """
    _validate_probability("q", q)
    _validate_probability("c", c)
    _validate_non_negative("kelly_fraction", kelly_fraction)
    _validate_non_negative("f_cap", f_cap)

    if q <= c or c >= 1.0:
        return 0.0

    f_star = (q - c) / (1.0 - c)
    f = kelly_fraction * f_star
    return min(max(0.0, f), f_cap)


def kelly_order_size(
    bankroll: float,
    q: float,
    c: float,
    *,
    kelly_fraction: float = 1.0,
    f_cap: float = 1.0,
    min_order_size: float = 0.0,
) -> float:
    """Convert Kelly fraction to notional order size with minimum-size floor."""
    _validate_non_negative("bankroll", bankroll)
    _validate_non_negative("min_order_size", min_order_size)

    f = fractional_kelly(q, c, kelly_fraction=kelly_fraction, f_cap=f_cap)
    size = bankroll * f
    if size < min_order_size:
        return 0.0
    return size


class DecisionEngine:
    """Small deterministic engine that returns trade/no-trade intents."""

    def __init__(self, config: DecisionConfig):
        self.config = config

    def decide(
        self,
        *,
        spot: float,
        strike: float,
        tau: float,
        rv: float,
        ask_up: float,
        ask_down: float,
        bankroll: float,
        iv: float | None = None,
    ) -> DecisionOutcome:
        sigma = blended_sigma(rv=rv, iv=iv, w=self.config.sigma_weight)
        q_up, q_down = compute_probabilities(
            spot=spot,
            strike=strike,
            sigma=sigma,
            tau=tau,
            epsilon=self.config.epsilon,
        )
        edge = evaluate_edges(
            q_up=q_up,
            q_down=q_down,
            ask_up=ask_up,
            ask_down=ask_down,
            buffer_up=self.config.edge_buffer_up,
            buffer_down=self.config.edge_buffer_down,
        )
        up_intent = self._build_intent(
            side="up",
            probability=q_up,
            ask=ask_up,
            edge=edge.edge_up,
            edge_after_buffer=edge.edge_up_after_buffer,
            bankroll=bankroll,
        )
        down_intent = self._build_intent(
            side="down",
            probability=q_down,
            ask=ask_down,
            edge=edge.edge_down,
            edge_after_buffer=edge.edge_down_after_buffer,
            bankroll=bankroll,
        )
        return DecisionOutcome(
            sigma=sigma,
            q_up=q_up,
            q_down=q_down,
            up=up_intent,
            down=down_intent,
        )

    def _build_intent(
        self,
        *,
        side: Literal["up", "down"],
        probability: float,
        ask: float,
        edge: float,
        edge_after_buffer: float,
        bankroll: float,
    ) -> TradeIntent:
        kelly_f = fractional_kelly(
            q=probability,
            c=ask,
            kelly_fraction=self.config.kelly_fraction,
            f_cap=self.config.f_cap,
        )
        order_size = bankroll * kelly_f

        should_trade = (
            bankroll > 0.0
            and edge_after_buffer >= self.config.edge_min
            and kelly_f > 0.0
            and order_size >= self.config.min_order_size
        )

        if should_trade:
            reason = "trade"
        elif bankroll <= 0.0:
            reason = "bankroll_non_positive"
            order_size = 0.0
        elif edge_after_buffer < self.config.edge_min:
            reason = "edge_below_min"
            order_size = 0.0
        elif kelly_f <= 0.0:
            reason = "kelly_zero"
            order_size = 0.0
        else:
            reason = "below_min_order_size"
            order_size = 0.0

        return TradeIntent(
            side=side,
            should_trade=should_trade,
            probability=probability,
            ask=ask,
            edge=edge,
            edge_after_buffer=edge_after_buffer,
            kelly_fraction=kelly_f,
            order_size=order_size,
            reason=reason,
        )


def _validate_positive(name: str, value: float) -> None:
    if value <= 0.0:
        raise ValueError(f"{name} must be > 0, got {value}")


def _validate_non_negative(name: str, value: float) -> None:
    if value < 0.0:
        raise ValueError(f"{name} must be >= 0, got {value}")


def _validate_probability(name: str, value: float) -> None:
    if value < 0.0 or value > 1.0:
        raise ValueError(f"{name} must be in [0, 1], got {value}")


def _clamp_unit_interval(value: float) -> float:
    return min(1.0, max(0.0, value))
