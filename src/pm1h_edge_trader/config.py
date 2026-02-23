"""Typed runtime configuration for PM-1H Edge Trader MVP."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Mapping


try:  # Python 3.11+
    from enum import StrEnum as _BaseStrEnum
except ImportError:  # pragma: no cover - Python <3.11 compatibility
    class _BaseStrEnum(str, Enum):
        pass


class RuntimeMode(_BaseStrEnum):
    """Trading runtime mode."""

    DRY_RUN = "dry-run"
    LIVE = "live"


@dataclass(slots=True, frozen=True)
class PolymarketConfig:
    gamma_base_url: str = "https://gamma-api.polymarket.com"
    clob_rest_base_url: str = "https://clob.polymarket.com"
    clob_ws_url: str = "wss://ws-subscriptions-clob.polymarket.com/ws/market"
    require_rule_match: bool = True
    market_slug: str | None = None


@dataclass(slots=True, frozen=True)
class PolymarketLiveAuthConfig:
    private_key: str | None = None
    funder: str | None = None
    signature_type: int = 1
    chain_id: int = 137
    api_key: str | None = None
    api_secret: str | None = None
    api_passphrase: str | None = None


@dataclass(slots=True, frozen=True)
class BinanceConfig:
    rest_base_url: str = "https://api.binance.com"
    ws_url: str | None = None
    symbol: str = "BTCUSDT"
    interval: str = "1h"


@dataclass(slots=True, frozen=True)
class VolatilityConfig:
    rv_interval: str = "1h"
    rv_lookback_hours: int = 48
    rv_ewma_half_life: float = 0.0
    rv_short_interval: str = "5m"
    rv_short_lookback_hours: int = 12
    rv_short_ewma_half_life: float = 18.0
    rv_tau_switch_seconds: float = 1800.0
    rv_refresh_seconds: float = 60.0
    rv_fallback: float = 0.55
    rv_floor: float = 0.10
    enable_deribit_iv: bool = False
    deribit_iv_currency: str = "BTC"
    deribit_iv_resolution_minutes: int = 60
    deribit_iv_lookback_hours: int = 48
    deribit_iv_refresh_seconds: float = 60.0
    deribit_iv_floor: float = 0.05
    deribit_iv_cap: float = 3.00
    sigma_weight: float = 1.0
    iv_override: float | None = None


@dataclass(slots=True, frozen=True)
class DecisionConfig:
    edge_min: float = 0.015
    edge_buffer_up: float = 0.010
    edge_buffer_down: float = 0.010
    cost_rate_up: float = 0.0
    cost_rate_down: float = 0.0


@dataclass(slots=True, frozen=True)
class RiskConfig:
    bankroll: float = 10_000.0
    kelly_fraction: float = 0.25
    f_cap: float = 0.05
    min_order_notional: float = 5.0
    max_market_notional: float = 2_500.0
    max_daily_loss: float = 2_000.0
    max_entries_per_market: int = 1


@dataclass(slots=True, frozen=True)
class LoopConfig:
    tick_seconds: float = 1.0
    max_ticks: int = 0
    feed_ready_timeout_seconds: float = 12.0
    enable_websocket: bool = True


@dataclass(slots=True, frozen=True)
class SafetyConfig:
    fee_check_interval_seconds: float = 20.0
    max_data_age_seconds: float = 4.0
    max_clock_drift_seconds: float = 3.0
    require_book_match: bool = True
    fee_must_be_zero: bool = True
    entry_block_window_seconds: float = 60.0
    heartbeat_interval_seconds: float = 5.0
    open_order_reconcile_interval_seconds: float = 10.0
    requote_interval_seconds: float = 15.0
    max_intent_age_seconds: float = 45.0
    kill_switch_latch: bool = True


@dataclass(slots=True, frozen=True)
class LoggingConfig:
    root_dir: Path = Path("logs")
    app_log_name: str = "app.log"
    execution_csv_name: str = "executions.csv"

    @property
    def app_log_path(self) -> Path:
        return self.root_dir / self.app_log_name

    @property
    def execution_csv_path(self) -> Path:
        return self.root_dir / self.execution_csv_name


@dataclass(slots=True, frozen=True)
class AppConfig:
    mode: RuntimeMode = RuntimeMode.DRY_RUN
    fresh_start: bool = False
    polymarket: PolymarketConfig = field(default_factory=PolymarketConfig)
    polymarket_live_auth: PolymarketLiveAuthConfig = field(default_factory=PolymarketLiveAuthConfig)
    binance: BinanceConfig = field(default_factory=BinanceConfig)
    volatility: VolatilityConfig = field(default_factory=VolatilityConfig)
    decision: DecisionConfig = field(default_factory=DecisionConfig)
    risk: RiskConfig = field(default_factory=RiskConfig)
    loop: LoopConfig = field(default_factory=LoopConfig)
    safety: SafetyConfig = field(default_factory=SafetyConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)

    @property
    def is_dry_run(self) -> bool:
        return self.mode == RuntimeMode.DRY_RUN


def build_config(
    *,
    mode: RuntimeMode,
    binance_symbol: str,
    market_slug: str | None,
    tick_seconds: float,
    max_ticks: int,
    bankroll: float,
    edge_min: float,
    edge_buffer: float,
    kelly_fraction: float,
    f_cap: float,
    min_order_notional: float,
    rv_fallback: float,
    sigma_weight: float,
    iv_override: float | None,
    log_dir: Path,
    enable_websocket: bool,
    cost_rate: float = 0.005,
    max_market_notional: float | None = None,
    max_daily_loss: float | None = None,
    max_entries_per_market: int = 1,
    rv_interval: str | None = None,
    rv_ewma_half_life: float | None = None,
    rv_short_interval: str | None = None,
    rv_short_lookback_hours: int | None = None,
    rv_short_ewma_half_life: float | None = None,
    rv_tau_switch_seconds: float | None = None,
    enable_deribit_iv: bool = False,
    deribit_iv_resolution_minutes: int | None = None,
    deribit_iv_lookback_hours: int | None = None,
    deribit_iv_refresh_seconds: float | None = None,
    fresh_start: bool = False,
    polymarket_live_auth: PolymarketLiveAuthConfig | None = None,
) -> AppConfig:
    """Constructs an app config from CLI inputs."""
    market_notional_cap = max_market_notional
    if market_notional_cap is None:
        market_notional_cap = max(0.0, bankroll * 0.25)
    daily_loss_cap = max_daily_loss
    if daily_loss_cap is None:
        daily_loss_cap = max(0.0, bankroll * 0.20)

    return AppConfig(
        mode=mode,
        fresh_start=fresh_start,
        polymarket=PolymarketConfig(market_slug=market_slug.strip() if market_slug else None),
        polymarket_live_auth=polymarket_live_auth or PolymarketLiveAuthConfig(),
        binance=BinanceConfig(symbol=binance_symbol.upper()),
        volatility=VolatilityConfig(
            rv_interval=(rv_interval or "1h").strip() or "1h",
            rv_fallback=rv_fallback,
            rv_ewma_half_life=max(0.0, rv_ewma_half_life or 0.0),
            rv_short_interval=(rv_short_interval or "5m").strip() or "5m",
            rv_short_lookback_hours=max(1, rv_short_lookback_hours or 12),
            rv_short_ewma_half_life=max(0.0, rv_short_ewma_half_life or 18.0),
            rv_tau_switch_seconds=max(1.0, rv_tau_switch_seconds or 1800.0),
            sigma_weight=sigma_weight,
            iv_override=iv_override,
            enable_deribit_iv=enable_deribit_iv,
            deribit_iv_resolution_minutes=max(1, deribit_iv_resolution_minutes or 60),
            deribit_iv_lookback_hours=max(1, deribit_iv_lookback_hours or 48),
            deribit_iv_refresh_seconds=max(5.0, deribit_iv_refresh_seconds or 60.0),
        ),
        decision=DecisionConfig(
            edge_min=edge_min,
            edge_buffer_up=edge_buffer,
            edge_buffer_down=edge_buffer,
            cost_rate_up=max(0.0, cost_rate),
            cost_rate_down=max(0.0, cost_rate),
        ),
        risk=RiskConfig(
            bankroll=bankroll,
            kelly_fraction=kelly_fraction,
            f_cap=f_cap,
            min_order_notional=min_order_notional,
            max_market_notional=max(0.0, market_notional_cap),
            max_daily_loss=max(0.0, daily_loss_cap),
            max_entries_per_market=max(1, max_entries_per_market),
        ),
        loop=LoopConfig(
            tick_seconds=tick_seconds,
            max_ticks=max_ticks,
            enable_websocket=enable_websocket,
        ),
        logging=LoggingConfig(root_dir=log_dir),
    )


def load_polymarket_live_auth_from_env(
    env: Mapping[str, str] | None = None,
) -> PolymarketLiveAuthConfig:
    source = env if env is not None else os.environ
    return PolymarketLiveAuthConfig(
        private_key=_normalize_optional_text(source.get("POLYMARKET_PRIVATE_KEY")),
        funder=_normalize_optional_text(source.get("POLYMARKET_FUNDER")),
        signature_type=_parse_env_int(source, "POLYMARKET_SIGNATURE_TYPE", default=1),
        chain_id=_parse_env_int(source, "POLYMARKET_CHAIN_ID", default=137),
        api_key=_normalize_optional_text(source.get("POLYMARKET_API_KEY")),
        api_secret=_normalize_optional_text(source.get("POLYMARKET_API_SECRET")),
        api_passphrase=_normalize_optional_text(source.get("POLYMARKET_API_PASSPHRASE")),
    )


def validate_live_mode_credentials(config: AppConfig) -> None:
    if config.mode != RuntimeMode.LIVE:
        return
    missing: list[str] = []
    if not _has_text(config.polymarket_live_auth.private_key):
        missing.append("POLYMARKET_PRIVATE_KEY")
    if not _has_text(config.polymarket_live_auth.funder):
        missing.append("POLYMARKET_FUNDER")
    if missing:
        missing_csv = ", ".join(missing)
        raise RuntimeError(
            f"--mode live requires environment variables: {missing_csv}"
        )

    if config.polymarket_live_auth.signature_type < 0:
        raise RuntimeError("POLYMARKET_SIGNATURE_TYPE must be a non-negative integer.")
    if config.polymarket_live_auth.chain_id <= 0:
        raise RuntimeError("POLYMARKET_CHAIN_ID must be a positive integer.")


def _normalize_optional_text(value: str | None) -> str | None:
    if value is None:
        return None
    normalized = value.strip()
    return normalized or None


def _has_text(value: str | None) -> bool:
    return value is not None and bool(value.strip())


def _parse_env_int(source: Mapping[str, str], key: str, *, default: int) -> int:
    raw = source.get(key)
    if raw is None:
        return default
    normalized = raw.strip()
    if not normalized:
        return default
    try:
        return int(normalized)
    except ValueError as exc:
        raise RuntimeError(f"{key} must be an integer.") from exc
