"""CLI entrypoint for PM-1H Edge Trader MVP."""

from __future__ import annotations

import argparse
import asyncio
import csv
import logging
import os
import signal
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from math import ceil, log, sqrt
from pathlib import Path
from typing import Any, Mapping, Sequence

from .config import (
    AppConfig,
    RuntimeMode,
    build_config,
    load_polymarket_live_auth_from_env,
    validate_live_mode_credentials,
)
from .engine import DecisionConfig as ProbabilityDecisionConfig
from .engine import DecisionEngine as ProbabilityDecisionEngine
from .engine import DecisionOutcome
from .execution import (
    ActiveIntent,
    DryRunExecutionAdapter,
    ExecutionAction,
    ExecutionActionType,
    ExecutionConfig as ExecutionEngineConfig,
    IntentSignal,
    KillSwitchReason,
    LimitOrderExecutionEngine,
    OutcomeSide,
    PolymarketLiveExecutionAdapter,
    SafetySurface,
    VenueOrderSide,
)
from .feeds import BinanceUnderlyingAdapter, ClobOrderbookAdapter
from .feeds.http_client import AsyncJsonHttpClient
from .feeds.models import BestQuote, UnderlyingSnapshot
from .logger import CSVExecutionReporter, ExecutionLogRecord
from .live_balance import LiveBalanceSnapshot, fetch_live_balance_snapshot
from .live_claim import LiveAutoClaimWorker, build_live_auto_claim_worker
from .markets import FeeRateGuardClient, GammaMarketDiscoveryClient, MarketCandidate
from .paper import PaperResultTracker, PaperSettlementCoordinator, PolymarketMarketResolutionResolver
from .policy import PolicyBanditConfig, PolicyBanditController, PolicySelection
from .positions import DataApiPositionsClient, evaluate_position_mismatch, extract_position_snapshot
from .risk import CircuitBreakerSnapshot, LiveBalanceMonitor, evaluate_circuit_breaker
from .strategy import CompleteSetArbDecision, decide_complete_set_arb

LOGGER = logging.getLogger("pm1h_edge_trader")
SECONDS_PER_YEAR = 365.0 * 24.0 * 60.0 * 60.0
LIVE_RECON_MAX_RETRIES = 3


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="PM-1H Edge Trader (MVP)")
    parser.add_argument(
        "--mode",
        choices=[mode.value for mode in RuntimeMode],
        default=RuntimeMode.DRY_RUN.value,
        help="Execution mode: dry-run or live.",
    )
    parser.add_argument(
        "--market-slug",
        default=None,
        help="Optional exact Polymarket market slug override.",
    )
    parser.add_argument("--binance-symbol", default="BTCUSDT", help="Underlying symbol.")
    parser.add_argument(
        "--tick-seconds",
        type=float,
        default=1.0,
        help="Loop interval in seconds.",
    )
    parser.add_argument(
        "--max-ticks",
        type=int,
        default=0,
        help="Number of ticks to run. 0 means run until interrupted.",
    )
    parser.add_argument("--bankroll", type=float, default=10_000.0, help="Risk capital.")
    parser.add_argument(
        "--edge-min",
        type=float,
        default=0.015,
        help="Minimum raw edge required (probability points).",
    )
    parser.add_argument(
        "--edge-buffer",
        type=float,
        default=0.010,
        help="Execution buffer subtracted from fair probability.",
    )
    parser.add_argument(
        "--cost-rate",
        type=float,
        default=0.005,
        help="Per-side cost haircut applied to q-ask edge (probability points).",
    )
    parser.add_argument(
        "--kelly-fraction",
        type=float,
        default=0.25,
        help="Fractional Kelly multiplier.",
    )
    parser.add_argument("--f-cap", type=float, default=0.05, help="Max bankroll fraction per side.")
    parser.add_argument(
        "--min-order-notional",
        type=float,
        default=0.3,
        help="Minimum order notional in quote currency.",
    )
    parser.add_argument(
        "--max-market-notional",
        type=float,
        default=None,
        help="Hard cap for unsettled notional in a single market. Default: 25%% of bankroll.",
    )
    parser.add_argument(
        "--max-daily-loss",
        type=float,
        default=None,
        help="Hard cap for realized daily loss. Default: 20%% of bankroll.",
    )
    parser.add_argument(
        "--max-worst-case-loss",
        type=float,
        default=None,
        help="Hard cap for worst-case loss proxy (unsettled + open-order notional). Default: 35%% of bankroll.",
    )
    parser.add_argument(
        "--max-live-drawdown",
        type=float,
        default=0.0,
        help="Hard cap for live bankroll drawdown from session baseline. 0 disables.",
    )
    parser.add_argument(
        "--live-balance-refresh-seconds",
        type=float,
        default=30.0,
        help="Refresh interval for live balance polling used by drawdown guard.",
    )
    parser.add_argument(
        "--max-entries-per-market",
        type=int,
        default=1,
        help="Maximum number of fills allowed per market.",
    )
    parser.add_argument(
        "--rv-fallback",
        type=float,
        default=0.55,
        help="Fallback annualized RV when estimator fails.",
    )
    parser.add_argument(
        "--rv-interval",
        default="1h",
        help="Binance kline interval used for long-horizon RV (e.g. 1h, 15m, 5m).",
    )
    parser.add_argument(
        "--rv-ewma-half-life",
        type=float,
        default=0.0,
        help="EWMA half-life in periods for long-horizon RV. 0 disables EWMA.",
    )
    parser.add_argument(
        "--rv-short-interval",
        default="5m",
        help="Binance kline interval used for short-horizon RV.",
    )
    parser.add_argument(
        "--rv-short-lookback-hours",
        type=int,
        default=12,
        help="Lookback window in hours for short-horizon RV.",
    )
    parser.add_argument(
        "--rv-short-ewma-half-life",
        type=float,
        default=18.0,
        help="EWMA half-life in periods for short-horizon RV.",
    )
    parser.add_argument(
        "--rv-tau-switch-seconds",
        type=float,
        default=1800.0,
        help="Time-to-expiry threshold where RV blend shifts toward short horizon.",
    )
    parser.add_argument(
        "--sigma-weight",
        type=float,
        default=1.0,
        help="RV/IV blend weight (1.0 = RV only).",
    )
    parser.add_argument(
        "--iv",
        type=float,
        default=None,
        help="Optional implied volatility override (annualized, decimal).",
    )
    parser.add_argument(
        "--enable-deribit-iv",
        action="store_true",
        help="Enable Deribit DVOL feed as dynamic IV source when --iv is not provided.",
    )
    parser.add_argument(
        "--deribit-iv-resolution-minutes",
        type=int,
        default=60,
        help="Deribit DVOL candle resolution in minutes.",
    )
    parser.add_argument(
        "--deribit-iv-lookback-hours",
        type=int,
        default=48,
        help="Lookback window for Deribit DVOL sampling.",
    )
    parser.add_argument(
        "--deribit-iv-refresh-seconds",
        type=float,
        default=60.0,
        help="Refresh cadence for Deribit DVOL polling.",
    )
    parser.add_argument(
        "--enable-policy-bandit",
        action="store_true",
        help="Enable contextual bandit policy profile selection.",
    )
    parser.add_argument(
        "--policy-shadow-mode",
        action="store_true",
        help="Select policy profiles for logging only and execute balanced profile.",
    )
    parser.add_argument(
        "--policy-exploration-epsilon",
        type=float,
        default=0.05,
        help="Exploration epsilon for policy bandit in dry-run mode.",
    )
    parser.add_argument(
        "--policy-ucb-c",
        type=float,
        default=1.0,
        help="UCB exploration coefficient for policy profile scoring.",
    )
    parser.add_argument(
        "--policy-reward-turnover-lambda",
        type=float,
        default=0.0,
        help="Turnover penalty multiplier in policy reward calculation.",
    )
    parser.add_argument(
        "--policy-reward-risk-penalty",
        type=float,
        default=0.0,
        help="Additional reward penalty applied on risk-limit breach.",
    )
    parser.add_argument(
        "--policy-vol-ratio-threshold",
        type=float,
        default=1.1,
        help="rv_short/rv_long threshold for high-vol context bucket.",
    )
    parser.add_argument(
        "--policy-spread-tight-threshold",
        type=float,
        default=0.03,
        help="Max spread considered tight for context bucketing.",
    )
    parser.add_argument(
        "--disable-auto-claim",
        action="store_true",
        help="Disable automatic on-chain redeem (claim) of resolved winning positions in live mode.",
    )
    parser.add_argument(
        "--auto-claim-interval-seconds",
        type=float,
        default=60.0,
        help="Polling interval for live auto-claim loop.",
    )
    parser.add_argument(
        "--auto-claim-size-threshold",
        type=float,
        default=0.0001,
        help="Minimum position size to include in auto-claim lookup.",
    )
    parser.add_argument(
        "--auto-claim-cooldown-seconds",
        type=float,
        default=600.0,
        help="Cooldown per condition between auto-claim attempts.",
    )
    parser.add_argument(
        "--auto-claim-tx-timeout-seconds",
        type=float,
        default=120.0,
        help="Timeout for on-chain claim transaction receipt.",
    )
    parser.add_argument(
        "--polygon-rpc-url",
        default=None,
        help="Polygon RPC URL for auto-claim transaction submission.",
    )
    parser.add_argument(
        "--data-api-base-url",
        default="https://data-api.polymarket.com",
        help="Polymarket data API base URL used for redeemable position lookup.",
    )
    parser.add_argument(
        "--disable-websocket",
        action="store_true",
        help="Disable websocket and use REST polling only.",
    )
    parser.add_argument(
        "--cancel-orphan-orders",
        action="store_true",
        help="Allow automatic cancellation of venue open orders unknown to the local runtime.",
    )
    parser.add_argument(
        "--hard-kill-on-daily-loss",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Latch kill switch and cancel orders when daily loss cap is reached (default: enabled).",
    )
    parser.add_argument(
        "--position-reconcile-interval-seconds",
        type=float,
        default=10.0,
        help="Polling interval for live wallet-position reconciliation.",
    )
    parser.add_argument(
        "--position-mismatch-policy",
        choices=["kill", "block", "warn"],
        default="kill",
        help="Action when live wallet positions differ from local bot expectations.",
    )
    parser.add_argument(
        "--position-size-threshold",
        type=float,
        default=0.0001,
        help="Minimum wallet position size used for mismatch checks.",
    )
    parser.add_argument(
        "--position-size-relative-tolerance",
        type=float,
        default=0.05,
        help="Relative tolerance for wallet-vs-local position size divergence checks.",
    )
    parser.add_argument(
        "--adopt-existing-positions",
        action="store_true",
        help="On restart, block new entries instead of immediate kill when wallet has unexpected positions.",
    )
    parser.add_argument(
        "--adopt-existing-positions-policy",
        choices=["kill", "block", "warn"],
        default="block",
        help="Policy for adopt mode when wallet-only positions are detected.",
    )
    parser.add_argument(
        "--enable-complete-set-arb",
        action="store_true",
        help="Enable complete-set arbitrage mode (buy UP+DOWN when ask_sum<1).",
    )
    parser.add_argument(
        "--arb-min-profit",
        type=float,
        default=0.0,
        help="Minimum required complete-set profit per pair (1-(ask_up+ask_down)).",
    )
    parser.add_argument(
        "--arb-max-notional",
        type=float,
        default=25.0,
        help="Maximum total notional allocated to one complete-set attempt.",
    )
    parser.add_argument(
        "--arb-fill-timeout-seconds",
        type=float,
        default=15.0,
        help="Timeout for one-leg-only complete-set state before defensive stop.",
    )
    parser.add_argument(
        "--log-dir",
        type=Path,
        default=Path("logs"),
        help="Directory for app/execution logs.",
    )
    parser.add_argument(
        "--fresh-start",
        action="store_true",
        help="Ignore prior executions.csv state and start result tracking from a clean vault snapshot.",
    )
    return parser


def configure_logging(config: AppConfig) -> None:
    config.logging.root_dir.mkdir(parents=True, exist_ok=True)
    handlers: list[logging.Handler] = [
        logging.StreamHandler(),
        logging.FileHandler(config.logging.app_log_path, encoding="utf-8"),
    ]
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        handlers=handlers,
    )


@dataclass(slots=True)
class FeeState:
    max_fee_rate: float
    checked_at: datetime


@dataclass(slots=True, frozen=True)
class AdaptiveSigmaSnapshot:
    long_sigma: float
    short_sigma: float
    blended_sigma: float
    short_weight: float


@dataclass(slots=True, frozen=True)
class LiveFillSnapshot:
    matched_size: float
    price: float | None


class RealizedVolEstimator:
    """Computes annualized realized volatility from Binance kline closes."""

    def __init__(
        self,
        *,
        rest_base_url: str,
        symbol: str,
        interval: str,
        lookback_hours: int,
        refresh_seconds: float,
        fallback_sigma: float,
        floor_sigma: float,
        ewma_half_life: float = 0.0,
        http_client: AsyncJsonHttpClient | None = None,
    ) -> None:
        self._rest_base_url = rest_base_url.rstrip("/")
        self._symbol = symbol.upper()
        self._interval = interval
        self._interval_seconds = _interval_to_seconds(interval)
        self._lookback_hours = max(1, lookback_hours)
        self._refresh_seconds = max(5.0, refresh_seconds)
        self._fallback_sigma = max(fallback_sigma, floor_sigma)
        self._floor_sigma = max(0.0, floor_sigma)
        self._ewma_half_life = max(0.0, ewma_half_life)
        self._http = http_client or AsyncJsonHttpClient()
        self._kline_limit = _compute_kline_limit(
            lookback_hours=self._lookback_hours,
            interval_seconds=self._interval_seconds,
        )
        self._cached_sigma: float = self._fallback_sigma
        self._cached_at: datetime | None = None

    async def get_sigma(self, *, now: datetime) -> float:
        if self._cached_at is not None:
            age_s = (now - self._cached_at).total_seconds()
            if age_s <= self._refresh_seconds:
                return self._cached_sigma

        sigma = await self._fetch_sigma_with_fallback()
        self._cached_sigma = max(self._floor_sigma, sigma)
        self._cached_at = now
        return self._cached_sigma

    async def _fetch_sigma_with_fallback(self) -> float:
        try:
            rows = await self._http.get_json(
                f"{self._rest_base_url}/api/v3/klines",
                params={
                    "symbol": self._symbol,
                    "interval": self._interval,
                    "limit": self._kline_limit,
                },
            )
            closes = _extract_closes(rows)
            returns = _log_returns(closes)
            if len(returns) < 2:
                return self._fallback_sigma
            period_vol = (
                _ewma_std(returns, half_life=self._ewma_half_life)
                if self._ewma_half_life > 0.0
                else _sample_std(returns)
            )
            annualized = period_vol * sqrt(SECONDS_PER_YEAR / float(self._interval_seconds))
            if annualized <= 0.0:
                return self._fallback_sigma
            return annualized
        except Exception:
            return self._fallback_sigma


class AdaptiveRealizedVolEstimator:
    """Blends long/short RV horizons based on time-to-expiry."""

    def __init__(
        self,
        *,
        long_horizon: object,
        short_horizon: object,
        tau_switch_seconds: float,
        floor_sigma: float,
        fallback_sigma: float,
    ) -> None:
        self._long_horizon = long_horizon
        self._short_horizon = short_horizon
        self._tau_switch_seconds = max(1.0, tau_switch_seconds)
        self._floor_sigma = max(0.0, floor_sigma)
        self._fallback_sigma = max(fallback_sigma, self._floor_sigma)

    async def get_snapshot(self, *, now: datetime, seconds_to_expiry: float) -> AdaptiveSigmaSnapshot:
        try:
            long_sigma = float(await self._long_horizon.get_sigma(now=now))
            short_sigma = float(await self._short_horizon.get_sigma(now=now))
            weight_short = _tau_short_weight(seconds_to_expiry, self._tau_switch_seconds)
            variance = ((1.0 - weight_short) * (long_sigma**2)) + (weight_short * (short_sigma**2))
            blended = sqrt(max(variance, 0.0))
            if blended <= 0.0:
                blended = self._fallback_sigma
            return AdaptiveSigmaSnapshot(
                long_sigma=max(self._floor_sigma, long_sigma),
                short_sigma=max(self._floor_sigma, short_sigma),
                blended_sigma=max(self._floor_sigma, blended),
                short_weight=weight_short,
            )
        except Exception:
            return AdaptiveSigmaSnapshot(
                long_sigma=self._fallback_sigma,
                short_sigma=self._fallback_sigma,
                blended_sigma=self._fallback_sigma,
                short_weight=0.0,
            )

    async def get_sigma(self, *, now: datetime, seconds_to_expiry: float) -> float:
        snapshot = await self.get_snapshot(now=now, seconds_to_expiry=seconds_to_expiry)
        return snapshot.blended_sigma


class DeribitVolatilityIndexEstimator:
    """Polls Deribit volatility-index candles and returns latest annualized IV."""

    def __init__(
        self,
        *,
        currency: str,
        resolution_minutes: int,
        lookback_hours: int,
        refresh_seconds: float,
        floor: float,
        cap: float,
        http_client: AsyncJsonHttpClient | None = None,
    ) -> None:
        self._currency = currency.upper()
        self._resolution_minutes = max(1, resolution_minutes)
        self._lookback_hours = max(1, lookback_hours)
        self._refresh_seconds = max(5.0, refresh_seconds)
        self._floor = max(0.0, floor)
        self._cap = max(self._floor, cap)
        self._http = http_client or AsyncJsonHttpClient()
        self._cached_iv: float | None = None
        self._cached_at: datetime | None = None

    async def get_iv(self, *, now: datetime) -> float | None:
        if self._cached_at is not None:
            age_s = (now - self._cached_at).total_seconds()
            if age_s <= self._refresh_seconds:
                return self._cached_iv

        fetched = await self._fetch_with_fallback(now=now)
        self._cached_iv = fetched
        self._cached_at = now
        return self._cached_iv

    async def _fetch_with_fallback(self, *, now: datetime) -> float | None:
        try:
            end_ms = int(now.timestamp() * 1000)
            start_ms = int((now - timedelta(hours=self._lookback_hours)).timestamp() * 1000)
            payload = await self._http.get_json(
                "https://www.deribit.com/api/v2/public/get_volatility_index_data",
                params={
                    "currency": self._currency,
                    "resolution": str(self._resolution_minutes),
                    "start_timestamp": start_ms,
                    "end_timestamp": end_ms,
                },
            )
            close = _extract_deribit_iv_close(payload)
            if close is None:
                return None
            as_decimal = close / 100.0 if close > 3.0 else close
            clamped = min(self._cap, max(self._floor, as_decimal))
            if clamped <= 0.0:
                return None
            return clamped
        except Exception:
            return self._cached_iv


class PM1HEdgeTraderApp:
    """Orchestrates market discovery, feeds, decisioning, execution, and logging."""

    def __init__(self, config: AppConfig) -> None:
        self._config = config

        self._discovery = GammaMarketDiscoveryClient(base_url=config.polymarket.gamma_base_url)
        self._fee_guard = FeeRateGuardClient(base_url=config.polymarket.clob_rest_base_url)
        self._reporter = CSVExecutionReporter(config.logging.execution_csv_path)

        self._base_decision_config = ProbabilityDecisionConfig(
            sigma_weight=config.volatility.sigma_weight,
            edge_min=config.decision.edge_min,
            edge_buffer_up=config.decision.edge_buffer_up,
            edge_buffer_down=config.decision.edge_buffer_down,
            cost_rate_up=config.decision.cost_rate_up,
            cost_rate_down=config.decision.cost_rate_down,
            kelly_fraction=config.risk.kelly_fraction,
            f_cap=config.risk.f_cap,
            min_order_size=config.risk.min_order_notional,
        )
        self._probability_engine = ProbabilityDecisionEngine(self._base_decision_config)

        execution_cfg = ExecutionEngineConfig(
            requote_interval_s=config.safety.requote_interval_seconds,
            max_intent_age_s=config.safety.max_intent_age_seconds,
            entry_block_window_s=config.safety.entry_block_window_seconds,
            max_data_age_s=config.safety.max_data_age_seconds,
            max_clock_drift_s=config.safety.max_clock_drift_seconds,
            require_book_match=config.safety.require_book_match,
            fee_must_be_zero=config.safety.fee_must_be_zero,
            max_entries_per_market=config.risk.max_entries_per_market,
            latch_kill_switch=config.safety.kill_switch_latch,
        )
        adapter = (
            DryRunExecutionAdapter(id_prefix="dry")
            if config.mode == RuntimeMode.DRY_RUN
            else PolymarketLiveExecutionAdapter(
                host=config.polymarket.clob_rest_base_url,
                auth=config.polymarket_live_auth,
            )
        )
        self._execution_engine = LimitOrderExecutionEngine(
            adapter=adapter,
            config=execution_cfg,
        )

        rv_long_estimator = RealizedVolEstimator(
            rest_base_url=config.binance.rest_base_url,
            symbol=config.binance.symbol,
            interval=config.volatility.rv_interval,
            lookback_hours=config.volatility.rv_lookback_hours,
            refresh_seconds=config.volatility.rv_refresh_seconds,
            fallback_sigma=config.volatility.rv_fallback,
            floor_sigma=config.volatility.rv_floor,
            ewma_half_life=config.volatility.rv_ewma_half_life,
        )
        rv_short_estimator = RealizedVolEstimator(
            rest_base_url=config.binance.rest_base_url,
            symbol=config.binance.symbol,
            interval=config.volatility.rv_short_interval,
            lookback_hours=config.volatility.rv_short_lookback_hours,
            refresh_seconds=config.volatility.rv_refresh_seconds,
            fallback_sigma=config.volatility.rv_fallback,
            floor_sigma=config.volatility.rv_floor,
            ewma_half_life=config.volatility.rv_short_ewma_half_life,
        )
        self._rv_estimator = AdaptiveRealizedVolEstimator(
            long_horizon=rv_long_estimator,
            short_horizon=rv_short_estimator,
            tau_switch_seconds=config.volatility.rv_tau_switch_seconds,
            floor_sigma=config.volatility.rv_floor,
            fallback_sigma=config.volatility.rv_fallback,
        )
        self._iv_estimator: DeribitVolatilityIndexEstimator | None = None
        if config.volatility.enable_deribit_iv and config.volatility.iv_override is None:
            self._iv_estimator = DeribitVolatilityIndexEstimator(
                currency=config.volatility.deribit_iv_currency,
                resolution_minutes=config.volatility.deribit_iv_resolution_minutes,
                lookback_hours=config.volatility.deribit_iv_lookback_hours,
                refresh_seconds=config.volatility.deribit_iv_refresh_seconds,
                floor=config.volatility.deribit_iv_floor,
                cap=config.volatility.deribit_iv_cap,
            )
        self._policy_controller: PolicyBanditController | None = None
        if config.policy.enabled:
            self._policy_controller = PolicyBanditController(
                config=PolicyBanditConfig(
                    enabled=config.policy.enabled,
                    shadow_mode=config.policy.shadow_mode,
                    exploration_epsilon=config.policy.exploration_epsilon,
                    ucb_c=config.policy.ucb_c,
                    reward_turnover_lambda=config.policy.reward_turnover_lambda,
                    reward_risk_penalty=config.policy.reward_risk_penalty,
                    vol_ratio_threshold=config.policy.vol_ratio_threshold,
                    spread_tight_threshold=config.policy.spread_tight_threshold,
                    dataset_path=config.logging.root_dir / config.policy.dataset_csv_name,
                    state_path=config.logging.root_dir / config.policy.state_json_name,
                ),
                base_decision_config=self._base_decision_config,
            )
        self._latest_policy_selection: PolicySelection | None = None

        self._market: MarketCandidate | None = None
        self._orderbook: ClobOrderbookAdapter | None = None
        self._underlying: BinanceUnderlyingAdapter | None = None
        self._fee_state: FeeState | None = None
        self._paper_settlement: PaperSettlementCoordinator | None = None
        self._paper_result_tracker: PaperResultTracker | None = None
        self._paper_settlement_interval = timedelta(seconds=15)
        self._next_paper_settlement_check: datetime | None = None
        self._token_tick_sizes: dict[str, float] = {}
        self._near_expiry_cleanup_done = False
        self._next_heartbeat_check: datetime | None = None
        self._next_open_order_reconcile_check: datetime | None = None
        self._next_position_reconcile_check: datetime | None = None
        self._external_kill_switch_latched = False
        self._position_mismatch_blocked = False
        self._live_auto_claim_worker: LiveAutoClaimWorker | None = None
        self._next_auto_claim_check: datetime | None = None
        self._auto_claim_init_failed = False
        self._positions_client: DataApiPositionsClient | None = None
        self._live_balance_monitor: LiveBalanceMonitor | None = None
        self._arb_one_leg_started_at: datetime | None = None
        self._live_order_matched_sizes: dict[str, float] = {}
        self._live_market_side_exposure_tokens: dict[tuple[str, OutcomeSide], float] = {}

        if config.mode == RuntimeMode.DRY_RUN:
            resolver = PolymarketMarketResolutionResolver(
                gamma_base_url=config.polymarket.gamma_base_url,
                clob_rest_base_url=config.polymarket.clob_rest_base_url,
            )
            self._paper_settlement = PaperSettlementCoordinator(
                execution_csv_path=config.logging.execution_csv_path,
                reporter=self._reporter,
                resolver=resolver,
            )
            self._paper_result_tracker = PaperResultTracker(
                initial_bankroll=config.risk.bankroll,
                result_json_path=config.logging.root_dir / "result.json",
            )
        elif config.mode == RuntimeMode.LIVE:
            self._paper_result_tracker = PaperResultTracker(
                initial_bankroll=config.risk.bankroll,
                result_json_path=config.logging.root_dir / "result.json",
                fill_statuses=("live_fill",),
                settle_statuses=("live_settle",),
            )
            self._positions_client = DataApiPositionsClient(
                base_url=config.position_reconcile.data_api_base_url,
            )
            if config.risk.max_live_drawdown > 0.0:
                self._live_balance_monitor = LiveBalanceMonitor(
                    auth=config.polymarket_live_auth,
                    host=config.polymarket.clob_rest_base_url,
                    refresh_seconds=config.risk.live_balance_refresh_seconds,
                    max_drawdown=config.risk.max_live_drawdown,
                )

    async def run(self, stop_event: asyncio.Event) -> None:
        await self._bootstrap_runtime()
        tick_count = 0

        try:
            while not stop_event.is_set():
                if self._market is None:
                    activated = await self._activate_market()
                    if not activated:
                        try:
                            await asyncio.wait_for(
                                stop_event.wait(),
                                timeout=self._config.loop.tick_seconds,
                            )
                        except asyncio.TimeoutError:
                            continue
                        continue

                tick_count += 1
                completed = await self._tick_once(tick_count=tick_count)
                if completed:
                    market = self._market
                    if market is not None:
                        LOGGER.info(
                            "Market expired (id=%s slug=%s). Rotating to next market.",
                            market.market_id,
                            market.slug,
                        )
                    else:
                        LOGGER.info("Market expired. Rotating to next market.")
                    await self._deactivate_market()
                    continue

                if self._config.loop.max_ticks > 0 and tick_count >= self._config.loop.max_ticks:
                    LOGGER.info("Reached max ticks (%s).", self._config.loop.max_ticks)
                    break

                try:
                    await asyncio.wait_for(stop_event.wait(), timeout=self._config.loop.tick_seconds)
                except asyncio.TimeoutError:
                    continue
        finally:
            await self._deactivate_market()

    async def _bootstrap_runtime(self) -> None:
        if self._paper_result_tracker is not None:
            if self._config.fresh_start:
                LOGGER.info("fresh_start enabled: ignoring previous executions.csv state")
                _reset_execution_csv(self._config.logging.execution_csv_path)
                self._paper_result_tracker.write_snapshot()
                if self._policy_controller is not None:
                    self._policy_controller.reset()
            else:
                self._paper_result_tracker.bootstrap_from_csv(self._config.logging.execution_csv_path)

    async def _initialize(self) -> None:
        await self._bootstrap_runtime()
        activated = await self._activate_market()
        if not activated:
            raise RuntimeError("No active BTC 1H Up/Down market found from Gamma.")

    async def _activate_market(self) -> bool:
        market = await self._discovery.find_active_btc_1h_up_down_market(
            require_rule_match=self._config.polymarket.require_rule_match,
            preferred_slug=self._config.polymarket.market_slug,
        )
        if market is None:
            LOGGER.info("No active BTC 1H Up/Down market found yet. Retrying.")
            return False
        if not market.rule_check.allowed:
            reasons = ", ".join(market.rule_check.reasons) or "rule mismatch"
            raise RuntimeError(f"Market rule validation failed: {reasons}")

        self._market = market
        self._orderbook = ClobOrderbookAdapter(
            token_ids=[market.token_ids.up_token_id, market.token_ids.down_token_id],
            rest_base_url=self._config.polymarket.clob_rest_base_url,
            ws_url=self._config.polymarket.clob_ws_url,
            enable_websocket=self._config.loop.enable_websocket,
        )
        self._underlying = BinanceUnderlyingAdapter(
            symbol=self._config.binance.symbol,
            interval=self._config.binance.interval,
            rest_base_url=self._config.binance.rest_base_url,
            ws_url=self._config.binance.ws_url,
            enable_websocket=self._config.loop.enable_websocket,
        )

        try:
            await asyncio.gather(self._orderbook.start(), self._underlying.start())
            await asyncio.gather(
                self._orderbook.wait_until_ready(timeout_seconds=self._config.loop.feed_ready_timeout_seconds),
                self._underlying.wait_until_ready(timeout_seconds=self._config.loop.feed_ready_timeout_seconds),
            )
        except Exception:
            await self._deactivate_market()
            raise

        now_ts = utc_now()
        if not self._external_kill_switch_latched:
            self._execution_engine.reset_kill_switch()
        self._near_expiry_cleanup_done = False
        self._set_market_tick_sizes(self._discover_market_tick_sizes(market))
        self._next_heartbeat_check = now_ts
        self._next_open_order_reconcile_check = now_ts
        self._next_position_reconcile_check = now_ts
        self._next_auto_claim_check = now_ts
        await self._refresh_fee_state(force=True, now=now_ts)
        await self._reconcile_paper_settlements(force=True, now=now_ts)
        LOGGER.info(
            "Selected market id=%s slug=%s start=%s end=%s up_token=%s down_token=%s tick_up=%s tick_down=%s",
            market.market_id,
            market.slug,
            market.timing.start.isoformat(),
            market.timing.end.isoformat(),
            market.token_ids.up_token_id,
            market.token_ids.down_token_id,
            self._token_tick_sizes.get(market.token_ids.up_token_id),
            self._token_tick_sizes.get(market.token_ids.down_token_id),
        )
        return True

    async def _shutdown(self) -> None:
        await self._deactivate_market()

    async def _deactivate_market(self) -> None:
        if self._config.mode == RuntimeMode.LIVE:
            try:
                self._execution_engine.cancel_all_intents(reason="shutdown")
            except Exception as exc:
                LOGGER.warning("shutdown_local_cancel_failed: %s", exc)

        tasks: list[asyncio.Future[object]] = []
        if self._orderbook is not None:
            tasks.append(asyncio.ensure_future(self._orderbook.stop()))
        if self._underlying is not None:
            tasks.append(asyncio.ensure_future(self._underlying.stop()))
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        self._orderbook = None
        self._underlying = None
        self._fee_state = None
        self._token_tick_sizes = {}
        self._near_expiry_cleanup_done = False
        self._next_heartbeat_check = None
        self._next_open_order_reconcile_check = None
        self._next_position_reconcile_check = None
        self._next_auto_claim_check = None
        self._latest_policy_selection = None
        self._position_mismatch_blocked = False
        self._arb_one_leg_started_at = None
        self._live_order_matched_sizes = {}
        self._live_market_side_exposure_tokens = {}
        self._market = None

    async def _tick_once(self, *, tick_count: int) -> bool:
        market = self._require_market()
        now = utc_now()
        if now >= market.timing.end:
            return True

        self._run_live_heartbeat(now=now)
        self._reconcile_live_open_orders(now=now)
        self._run_live_auto_claim(now=now)
        await self._reconcile_live_positions(now=now)
        self._run_live_drawdown_guard(now=now)

        await self._refresh_fee_state(now=now)
        max_fee_rate = self._fee_state.max_fee_rate if self._fee_state is not None else 0.0

        book = await self._require_orderbook().get_snapshot()
        underlying = await self._require_underlying().get_snapshot()
        up_quote = book.quotes.get(market.token_ids.up_token_id)
        down_quote = book.quotes.get(market.token_ids.down_token_id)

        data_ready = _is_data_ready(up_quote, down_quote, underlying)
        seconds_to_expiry = max((market.timing.end - now).total_seconds(), 0.0)
        self._enforce_near_expiry_cleanup(
            market_id=market.market_id,
            seconds_to_expiry=seconds_to_expiry,
            now=now,
        )
        tau_years = seconds_to_expiry / SECONDS_PER_YEAR

        decision: DecisionOutcome | None = None
        arb_decision: CompleteSetArbDecision | None = None
        sigma = self._config.volatility.rv_fallback
        rv_long = sigma
        rv_short = sigma
        dynamic_iv: float | None = self._config.volatility.iv_override
        selected_profile_id = "P2_BALANCED"
        applied_profile_id = "P2_BALANCED"
        self._latest_policy_selection = None
        decision_bankroll = self._decision_bankroll()
        entry_block_reason = self._entry_block_reason(now=now, market_id=market.market_id)
        circuit_breaker = evaluate_circuit_breaker(
            CircuitBreakerSnapshot(
                daily_realized_pnl=self._daily_realized_pnl(now),
                max_daily_loss=self._config.risk.max_daily_loss,
                decision_bankroll=decision_bankroll,
                market_notional=self._market_effective_notional(market.market_id),
                max_market_notional=self._config.risk.max_market_notional,
                worst_case_loss=self._worst_case_loss_estimate(),
                max_worst_case_loss=self._config.risk.max_worst_case_loss,
            )
        )
        if circuit_breaker.triggered:
            should_kill = True
            if (
                KillSwitchReason.DAILY_LOSS_LIMIT in circuit_breaker.reasons
                and not self._config.safety.hard_kill_on_daily_loss
            ):
                should_kill = False
            if should_kill:
                self._trigger_external_kill_switch(
                    reason=circuit_breaker.reasons[0],
                    now=now,
                    message=(
                        "Risk circuit breaker triggered "
                        f"reasons={','.join(reason.value for reason in circuit_breaker.reasons)}"
                    ),
                )
        if data_ready and tau_years > 0.0:
            rv_snapshot = await self._rv_estimator.get_snapshot(
                now=now,
                seconds_to_expiry=seconds_to_expiry,
            )
            rv_long = rv_snapshot.long_sigma
            rv_short = rv_snapshot.short_sigma
            sigma = rv_snapshot.blended_sigma
            if dynamic_iv is None and self._iv_estimator is not None:
                dynamic_iv = await self._iv_estimator.get_iv(now=now)
            decision_engine = self._probability_engine
            if self._policy_controller is not None:
                policy_selection = self._policy_controller.select_profile(
                    timestamp=now,
                    market_id=market.market_id,
                    seconds_to_expiry=seconds_to_expiry,
                    rv_long=rv_long,
                    rv_short=rv_short,
                    sigma=sigma,
                    iv=dynamic_iv,
                    spot=underlying.last_price or 0.0,
                    strike=underlying.candle_open or 0.0,
                    bid_up=up_quote.best_bid or 0.0,
                    ask_up=up_quote.best_ask or 0.0,
                    bid_down=down_quote.best_bid or 0.0,
                    ask_down=down_quote.best_ask or 0.0,
                    bankroll_at_entry=decision_bankroll,
                    allow_exploration=self._config.mode == RuntimeMode.DRY_RUN,
                )
                self._latest_policy_selection = policy_selection
                selected_profile_id = policy_selection.chosen_profile_id
                applied_profile_id = policy_selection.applied_profile_id
                decision_engine = ProbabilityDecisionEngine(
                    self._policy_controller.decision_config_for(applied_profile_id)
                )
            decision = decision_engine.decide(
                spot=underlying.last_price or 0.0,
                strike=underlying.candle_open or 0.0,
                tau=tau_years,
                rv=sigma,
                iv=dynamic_iv,
                ask_up=up_quote.best_ask or 0.0,
                ask_down=down_quote.best_ask or 0.0,
                bankroll=decision_bankroll,
            )
            sigma = decision.sigma
            if (
                self._config.complete_set_arb.enabled
                and entry_block_reason is None
                and up_quote is not None
                and down_quote is not None
                and up_quote.best_ask is not None
                and down_quote.best_ask is not None
            ):
                remaining = self._market_remaining_notional(market.market_id)
                arb_max_notional = min(self._config.complete_set_arb.max_notional, remaining)
                arb_decision = decide_complete_set_arb(
                    ask_up=up_quote.best_ask,
                    ask_down=down_quote.best_ask,
                    bankroll=decision_bankroll,
                    min_profit=self._config.complete_set_arb.min_profit,
                    max_notional=arb_max_notional,
                    min_order_notional=self._config.risk.min_order_notional,
                )
                if arb_decision.should_trade:
                    selected_profile_id = "ARB_COMPLETE_SET"
                    applied_profile_id = "ARB_COMPLETE_SET"

        safety = SafetySurface(
            fee_rate=max_fee_rate,
            data_age_s=_compute_data_age_seconds(now=now, up_quote=up_quote, down_quote=down_quote, underlying=underlying),
            clock_drift_s=_compute_clock_drift_seconds(now=now, underlying=underlying),
            book_is_consistent=_is_book_consistent(up_quote, down_quote),
        )

        if arb_decision is not None and arb_decision.should_trade:
            up_signal, down_signal = self._build_complete_set_arb_signals(
                market_id=market.market_id,
                now=now,
                seconds_to_expiry=seconds_to_expiry,
                up_quote=up_quote,
                down_quote=down_quote,
                arb=arb_decision,
                entry_block_reason=entry_block_reason,
            )
        else:
            up_signal = self._build_signal(
                market_id=market.market_id,
                token_id=market.token_ids.up_token_id,
                side_label="up",
                now=now,
                seconds_to_expiry=seconds_to_expiry,
                quote=up_quote,
                decision=decision,
                data_ready=data_ready,
                entry_block_reason=entry_block_reason,
            )
            down_signal = self._build_signal(
                market_id=market.market_id,
                token_id=market.token_ids.down_token_id,
                side_label="down",
                now=now,
                seconds_to_expiry=seconds_to_expiry,
                quote=down_quote,
                decision=decision,
                data_ready=data_ready,
                entry_block_reason=entry_block_reason,
            )

        all_actions: list[ExecutionAction] = []
        for signal in (up_signal, down_signal):
            result = self._execution_engine.process_signal(signal, safety, now=now)
            all_actions.extend(result.actions)
        stale_result = self._execution_engine.sweep_stale_intents(now=now)
        all_actions.extend(stale_result.actions)
        self._guard_complete_set_partial_state(
            now=now,
            market_id=market.market_id,
            seconds_to_expiry=seconds_to_expiry,
            up_quote=up_quote,
            down_quote=down_quote,
            safety=safety,
        )

        signal_map = {
            up_signal.token_id: up_signal,
            down_signal.token_id: down_signal,
        }
        for action in all_actions:
            self._report_action(
                action=action,
                signal_map=signal_map,
                market=market,
                underlying=underlying,
                up_quote=up_quote,
                down_quote=down_quote,
                tau_years=tau_years,
                sigma=sigma,
                decision=decision,
            )

        self._log_tick_summary(
            tick_count=tick_count,
            tau_years=tau_years,
            decision=decision,
            up_quote=up_quote,
            down_quote=down_quote,
            fee_rate=max_fee_rate,
            action_count=len(all_actions),
            selected_profile_id=selected_profile_id,
            applied_profile_id=applied_profile_id,
            arb_decision=arb_decision,
        )
        await self._reconcile_paper_settlements(now=now)
        return False

    def _ensure_live_auto_claim_worker(self) -> LiveAutoClaimWorker | None:
        if self._config.mode != RuntimeMode.LIVE:
            return None
        if not self._config.auto_claim.enabled:
            return None
        if self._auto_claim_init_failed:
            return None
        if self._live_auto_claim_worker is not None:
            return self._live_auto_claim_worker

        adapter = self._execution_engine._adapter
        if not isinstance(adapter, PolymarketLiveExecutionAdapter):
            self._auto_claim_init_failed = True
            LOGGER.warning("auto_claim_disabled reason=live_adapter_unavailable")
            return None
        try:
            self._live_auto_claim_worker = build_live_auto_claim_worker(
                auth=self._config.polymarket_live_auth,
                adapter=adapter,
                rpc_url=self._config.auto_claim.polygon_rpc_url,
                data_api_base_url=self._config.auto_claim.data_api_base_url,
                size_threshold=self._config.auto_claim.size_threshold,
                cooldown_seconds=self._config.auto_claim.cooldown_seconds,
                tx_timeout_seconds=self._config.auto_claim.tx_timeout_seconds,
            )
        except Exception as exc:
            self._auto_claim_init_failed = True
            LOGGER.warning("auto_claim_disabled reason=%s", exc)
            return None
        return self._live_auto_claim_worker

    def _run_live_auto_claim(self, *, now: datetime, force: bool = False) -> None:
        if self._config.mode != RuntimeMode.LIVE:
            return
        worker = self._ensure_live_auto_claim_worker()
        if worker is None:
            return
        if not force and self._next_auto_claim_check is not None and now < self._next_auto_claim_check:
            return

        try:
            summary = worker.run_once()
        except Exception as exc:
            LOGGER.warning("auto_claim_loop_failed error=%s", exc)
            interval = max(5.0, self._config.auto_claim.interval_seconds)
            self._next_auto_claim_check = now + timedelta(seconds=interval)
            return

        if summary.targets > 0 or summary.attempted > 0:
            LOGGER.info(
                (
                    "auto_claim_scan checked_positions=%s targets=%s attempted=%s "
                    "claimed=%s errors=%s skipped_cooldown=%s"
                ),
                summary.checked_positions,
                summary.targets,
                summary.attempted,
                summary.claimed,
                summary.errors,
                summary.skipped_cooldown,
            )
        for attempt in summary.attempts:
            if attempt.success:
                LOGGER.info(
                    "auto_claim_success condition_id=%s index_sets=%s tx_hash=%s",
                    attempt.condition_id,
                    ",".join(str(item) for item in attempt.index_sets),
                    attempt.tx_hash,
                )
                continue
            LOGGER.warning(
                "auto_claim_failed condition_id=%s index_sets=%s tx_hash=%s error=%s",
                attempt.condition_id,
                ",".join(str(item) for item in attempt.index_sets),
                attempt.tx_hash,
                attempt.error,
            )

        interval = max(5.0, self._config.auto_claim.interval_seconds)
        self._next_auto_claim_check = now + timedelta(seconds=interval)

    def _run_live_drawdown_guard(self, *, now: datetime) -> None:
        if self._config.mode != RuntimeMode.LIVE:
            return
        monitor = self._live_balance_monitor
        if monitor is None:
            return
        try:
            check = monitor.maybe_refresh(now=now)
        except Exception as exc:
            LOGGER.warning("live_drawdown_refresh_failed error=%s", exc)
            return
        if check is None:
            return
        if check.triggered:
            self._trigger_external_kill_switch(
                reason=KillSwitchReason.LIVE_DRAWDOWN_LIMIT,
                now=now,
                message=(
                    "Live drawdown limit reached "
                    f"baseline={check.baseline.bankroll:.6f} "
                    f"current={check.snapshot.bankroll:.6f} "
                    f"drawdown={check.drawdown:.6f} "
                    f"max={self._config.risk.max_live_drawdown:.6f}"
                ),
            )

    async def _reconcile_live_positions(self, *, now: datetime, force: bool = False) -> None:
        if self._config.mode != RuntimeMode.LIVE:
            return
        if self._positions_client is None:
            return
        market = self._market
        if market is None:
            return
        if not force and self._next_position_reconcile_check is not None and now < self._next_position_reconcile_check:
            return

        user_address = self._config.polymarket_live_auth.funder
        if not isinstance(user_address, str) or not user_address.strip():
            interval = max(1.0, self._config.position_reconcile.interval_seconds)
            self._next_position_reconcile_check = now + timedelta(seconds=interval)
            return

        try:
            positions = await self._positions_client.fetch_positions(
                user_address=user_address,
                size_threshold=self._config.position_reconcile.size_threshold,
            )
            snapshot = extract_position_snapshot(
                positions,
                up_token_id=market.token_ids.up_token_id,
                down_token_id=market.token_ids.down_token_id,
                size_threshold=self._config.position_reconcile.size_threshold,
            )
        except Exception as exc:
            LOGGER.warning("position_reconcile_failed error=%s", exc)
            interval = max(1.0, self._config.position_reconcile.interval_seconds)
            self._next_position_reconcile_check = now + timedelta(seconds=interval)
            return

        mismatch = evaluate_position_mismatch(
            snapshot=snapshot,
            local_up_size=self._market_side_exposure_tokens(market.market_id, OutcomeSide.UP),
            local_down_size=self._market_side_exposure_tokens(market.market_id, OutcomeSide.DOWN),
            size_threshold=self._config.position_reconcile.size_threshold,
            relative_tolerance=self._config.position_reconcile.size_relative_tolerance,
        )
        policy = self._config.position_reconcile.mismatch_policy
        adopt_policy = self._config.position_reconcile.adopt_existing_positions_policy
        if self._config.position_reconcile.adopt_existing_positions and mismatch.reason == "wallet_only_exposure":
            policy = adopt_policy
        if mismatch.mismatch:
            if policy == "warn":
                LOGGER.warning(
                    (
                        "position_reconcile_mismatch policy=warn reason=%s adopt=%s "
                        "wallet_up=%.6f wallet_down=%.6f local_up=%.6f local_down=%.6f "
                        "up_diff=%.6f down_diff=%.6f"
                    ),
                    mismatch.reason,
                    str(self._config.position_reconcile.adopt_existing_positions).lower(),
                    mismatch.wallet_up_size,
                    mismatch.wallet_down_size,
                    mismatch.local_up_size,
                    mismatch.local_down_size,
                    mismatch.up_diff,
                    mismatch.down_diff,
                )
            elif policy == "block":
                self._position_mismatch_blocked = True
                LOGGER.warning(
                    (
                        "position_reconcile_mismatch policy=block reason=%s adopt=%s "
                        "wallet_up=%.6f wallet_down=%.6f local_up=%.6f local_down=%.6f "
                        "up_diff=%.6f down_diff=%.6f"
                    ),
                    mismatch.reason,
                    str(self._config.position_reconcile.adopt_existing_positions).lower(),
                    mismatch.wallet_up_size,
                    mismatch.wallet_down_size,
                    mismatch.local_up_size,
                    mismatch.local_down_size,
                    mismatch.up_diff,
                    mismatch.down_diff,
                )
            else:
                self._position_mismatch_blocked = True
                self._trigger_external_kill_switch(
                    reason=KillSwitchReason.POSITION_MISMATCH,
                    now=now,
                    message=(
                        "position_reconcile_mismatch "
                        f"reason={mismatch.reason} "
                        f"wallet_up={mismatch.wallet_up_size:.6f} "
                        f"wallet_down={mismatch.wallet_down_size:.6f} "
                        f"local_up={mismatch.local_up_size:.6f} "
                        f"local_down={mismatch.local_down_size:.6f}"
                    ),
                )
        elif self._position_mismatch_blocked:
            self._position_mismatch_blocked = False
            LOGGER.info("position_reconcile_recovered")

        interval = max(1.0, self._config.position_reconcile.interval_seconds)
        self._next_position_reconcile_check = now + timedelta(seconds=interval)

    def _guard_complete_set_partial_state(
        self,
        *,
        now: datetime,
        market_id: str,
        seconds_to_expiry: float,
        up_quote: BestQuote | None,
        down_quote: BestQuote | None,
        safety: SafetySurface,
    ) -> None:
        if not self._config.complete_set_arb.enabled:
            self._arb_one_leg_started_at = None
            return
        up_token_id = str(up_quote.token_id).strip() if up_quote is not None else ""
        down_token_id = str(down_quote.token_id).strip() if down_quote is not None else ""
        has_up = self._has_active_intent_for_token(market_id=market_id, token_id=up_token_id)
        has_down = self._has_active_intent_for_token(market_id=market_id, token_id=down_token_id)
        one_leg_only = has_up ^ has_down
        if not one_leg_only:
            self._arb_one_leg_started_at = None
            return
        if self._arb_one_leg_started_at is None:
            self._arb_one_leg_started_at = now
            return
        elapsed = (now - self._arb_one_leg_started_at).total_seconds()
        if elapsed < self._config.complete_set_arb.fill_timeout_seconds:
            return
        recovered = self._attempt_complete_set_recovery(
            now=now,
            market_id=market_id,
            seconds_to_expiry=seconds_to_expiry,
            up_quote=up_quote,
            down_quote=down_quote,
            safety=safety,
        )
        if recovered:
            self._arb_one_leg_started_at = now
            return
        self._trigger_external_kill_switch(
            reason=KillSwitchReason.POSITION_MISMATCH,
            now=now,
            message="complete_set_one_leg_timeout",
        )
        self._arb_one_leg_started_at = now

    def _attempt_complete_set_recovery(
        self,
        *,
        now: datetime,
        market_id: str,
        seconds_to_expiry: float,
        up_quote: BestQuote | None,
        down_quote: BestQuote | None,
        safety: SafetySurface,
    ) -> bool:
        if self._external_kill_switch_latched or self._execution_engine.kill_switch.active:
            return False

        try:
            self._execution_engine.cancel_all_intents(reason="complete_set_recovery")
        except Exception as exc:
            LOGGER.warning("complete_set_recovery_cancel_failed error=%s", exc)

        up_exposure = max(0.0, self._market_side_exposure_tokens(market_id, OutcomeSide.UP))
        down_exposure = max(0.0, self._market_side_exposure_tokens(market_id, OutcomeSide.DOWN))
        imbalance = up_exposure - down_exposure
        if abs(imbalance) <= 1e-9:
            LOGGER.warning("complete_set_recovery_skipped reason=no_detected_exposure")
            return False

        if imbalance > 0.0:
            completion_ok = self._place_recovery_order(
                now=now,
                market_id=market_id,
                side=OutcomeSide.DOWN,
                token_id=down_quote.token_id if down_quote is not None else "",
                order_side=VenueOrderSide.BUY,
                quote_price=down_quote.best_ask if down_quote is not None else None,
                size=imbalance,
                seconds_to_expiry=seconds_to_expiry,
                safety=safety,
                reason="complete_set_recovery_complete_missing_down",
            )
            if completion_ok:
                return True
            unwind_ok = self._place_recovery_order(
                now=now,
                market_id=market_id,
                side=OutcomeSide.UP,
                token_id=up_quote.token_id if up_quote is not None else "",
                order_side=VenueOrderSide.SELL,
                quote_price=up_quote.best_bid if up_quote is not None else None,
                size=imbalance,
                seconds_to_expiry=seconds_to_expiry,
                safety=safety,
                reason="complete_set_recovery_unwind_up",
            )
            return unwind_ok

        completion_ok = self._place_recovery_order(
            now=now,
            market_id=market_id,
            side=OutcomeSide.UP,
            token_id=up_quote.token_id if up_quote is not None else "",
            order_side=VenueOrderSide.BUY,
            quote_price=up_quote.best_ask if up_quote is not None else None,
            size=abs(imbalance),
            seconds_to_expiry=seconds_to_expiry,
            safety=safety,
            reason="complete_set_recovery_complete_missing_up",
        )
        if completion_ok:
            return True
        unwind_ok = self._place_recovery_order(
            now=now,
            market_id=market_id,
            side=OutcomeSide.DOWN,
            token_id=down_quote.token_id if down_quote is not None else "",
            order_side=VenueOrderSide.SELL,
            quote_price=down_quote.best_bid if down_quote is not None else None,
            size=abs(imbalance),
            seconds_to_expiry=seconds_to_expiry,
            safety=safety,
            reason="complete_set_recovery_unwind_down",
        )
        return unwind_ok

    def _place_recovery_order(
        self,
        *,
        now: datetime,
        market_id: str,
        side: OutcomeSide,
        token_id: str,
        order_side: VenueOrderSide,
        quote_price: float | None,
        size: float,
        seconds_to_expiry: float,
        safety: SafetySurface,
        reason: str,
    ) -> bool:
        normalized_token = str(token_id).strip()
        if not normalized_token:
            LOGGER.warning("%s skipped reason=missing_token_id", reason)
            return False
        if quote_price is None or quote_price <= 0.0:
            LOGGER.warning("%s skipped reason=missing_price", reason)
            return False
        if size <= 0.0:
            LOGGER.warning("%s skipped reason=non_positive_size", reason)
            return False
        tick_size = self._token_tick_sizes.get(normalized_token, 0.0001)
        desired_price = _round_down_to_tick(quote_price, tick_size)
        emergency_seconds_to_expiry = max(
            seconds_to_expiry,
            self._config.safety.entry_block_window_seconds + 1.0,
        )
        signal = IntentSignal(
            market_id=market_id,
            token_id=normalized_token,
            side=side,
            edge=1.0,
            min_edge=-1.0,
            desired_price=desired_price,
            size=size,
            seconds_to_expiry=emergency_seconds_to_expiry,
            signal_ts=now,
            allow_entry=True,
            order_side=order_side,
        )
        try:
            result = self._execution_engine.process_signal(signal, safety, now=now)
        except Exception as exc:
            LOGGER.warning("%s failed error=%s", reason, exc)
            return False
        placed = any(action.action_type == ExecutionActionType.PLACE for action in result.actions)
        if placed:
            LOGGER.warning(
                "%s placed token=%s side=%s order_side=%s size=%.6f price=%.6f",
                reason,
                normalized_token,
                side.value,
                order_side.value,
                size,
                desired_price,
            )
            return True
        LOGGER.warning("%s skipped reason=no_place_action", reason)
        return False

    async def _refresh_fee_state(self, *, force: bool = False, now: datetime | None = None) -> None:
        market = self._require_market()
        now_ts = now or utc_now()
        if not force and self._fee_state is not None:
            elapsed = (now_ts - self._fee_state.checked_at).total_seconds()
            if elapsed < self._config.safety.fee_check_interval_seconds:
                return

        try:
            _, statuses = await self._fee_guard.any_trade_blocking_fee(
                [market.token_ids.up_token_id, market.token_ids.down_token_id]
            )
            max_fee_rate = max((status.fee_rate for status in statuses.values()), default=0.0)
        except Exception as exc:
            LOGGER.warning("Fee-rate check failed; blocking trading this tick: %s", exc)
            max_fee_rate = 1.0
        self._fee_state = FeeState(max_fee_rate=max_fee_rate, checked_at=now_ts)

    def _build_signal(
        self,
        *,
        market_id: str,
        token_id: str,
        side_label: str,
        now: datetime,
        seconds_to_expiry: float,
        quote: BestQuote | None,
        decision: DecisionOutcome | None,
        data_ready: bool,
        entry_block_reason: str | None = None,
    ) -> IntentSignal:
        side = OutcomeSide.UP if side_label == "up" else OutcomeSide.DOWN
        allow_entry = data_ready and entry_block_reason is None
        if decision is None or quote is None or quote.best_ask is None:
            return IntentSignal(
                market_id=market_id,
                token_id=quote.token_id if quote is not None else token_id,
                side=side,
                edge=0.0,
                min_edge=self._config.decision.edge_min,
                desired_price=None,
                size=0.0,
                seconds_to_expiry=max(seconds_to_expiry, 0.0),
                signal_ts=now,
                allow_entry=allow_entry,
                order_side=VenueOrderSide.BUY,
            )

        side_intent = decision.up if side_label == "up" else decision.down
        edge_buffer = (
            self._config.decision.edge_buffer_up
            if side_label == "up"
            else self._config.decision.edge_buffer_down
        )
        desired_price = _limit_price(
            ask=quote.best_ask,
            fair_probability=side_intent.probability,
            edge_buffer=edge_buffer,
            tick_size=self._token_tick_sizes.get(quote.token_id),
        )
        share_size = 0.0
        if side_intent.order_size > 0.0 and desired_price is not None and desired_price > 0.0:
            capped_notional = self._cap_order_notional_for_market(
                market_id=market_id,
                suggested_notional=side_intent.order_size,
            )
            share_size = capped_notional / desired_price

        return IntentSignal(
            market_id=market_id,
            token_id=quote.token_id,
            side=side,
            edge=side_intent.edge,
            min_edge=self._config.decision.edge_min,
            desired_price=desired_price,
            size=share_size,
            seconds_to_expiry=max(seconds_to_expiry, 0.0),
            signal_ts=now,
            allow_entry=allow_entry,
            order_side=VenueOrderSide.BUY,
        )

    def _build_complete_set_arb_signals(
        self,
        *,
        market_id: str,
        now: datetime,
        seconds_to_expiry: float,
        up_quote: BestQuote | None,
        down_quote: BestQuote | None,
        arb: CompleteSetArbDecision,
        entry_block_reason: str | None,
    ) -> tuple[IntentSignal, IntentSignal]:
        allow_entry = entry_block_reason is None
        up_price = (
            _round_down_to_tick(
                up_quote.best_ask,
                self._token_tick_sizes.get(up_quote.token_id, 0.0001),
            )
            if up_quote is not None and up_quote.best_ask is not None
            else None
        )
        down_price = (
            _round_down_to_tick(
                down_quote.best_ask,
                self._token_tick_sizes.get(down_quote.token_id, 0.0001),
            )
            if down_quote is not None and down_quote.best_ask is not None
            else None
        )
        up_signal = IntentSignal(
            market_id=market_id,
            token_id=up_quote.token_id if up_quote is not None else "",
            side=OutcomeSide.UP,
            edge=arb.profit_per_pair,
            min_edge=self._config.complete_set_arb.min_profit,
            desired_price=up_price,
            size=arb.pair_tokens,
            seconds_to_expiry=max(seconds_to_expiry, 0.0),
            signal_ts=now,
            allow_entry=allow_entry,
            order_side=VenueOrderSide.BUY,
        )
        down_signal = IntentSignal(
            market_id=market_id,
            token_id=down_quote.token_id if down_quote is not None else "",
            side=OutcomeSide.DOWN,
            edge=arb.profit_per_pair,
            min_edge=self._config.complete_set_arb.min_profit,
            desired_price=down_price,
            size=arb.pair_tokens,
            seconds_to_expiry=max(seconds_to_expiry, 0.0),
            signal_ts=now,
            allow_entry=allow_entry,
            order_side=VenueOrderSide.BUY,
        )
        return up_signal, down_signal

    def _report_action(
        self,
        *,
        action: ExecutionAction,
        signal_map: dict[str, IntentSignal],
        market: MarketCandidate,
        underlying: UnderlyingSnapshot,
        up_quote: BestQuote | None,
        down_quote: BestQuote | None,
        tau_years: float,
        sigma: float,
        decision: DecisionOutcome | None,
    ) -> None:
        side_signal: IntentSignal | None = None
        if action.token_id is not None:
            side_signal = signal_map.get(str(action.token_id).strip())
        if side_signal is None and action.side is not None:
            if action.side == OutcomeSide.UP:
                side_signal = next((signal for signal in signal_map.values() if signal.side == OutcomeSide.UP), None)
            elif action.side == OutcomeSide.DOWN:
                side_signal = next((signal for signal in signal_map.values() if signal.side == OutcomeSide.DOWN), None)
        record = ExecutionLogRecord(
            timestamp=utc_now(),
            market_id=market.market_id,
            K=underlying.candle_open or 0.0,
            S_t=underlying.last_price or 0.0,
            tau=tau_years,
            sigma=sigma,
            q_up=decision.q_up if decision is not None else 0.5,
            bid_up=up_quote.best_bid if up_quote and up_quote.best_bid is not None else 0.0,
            ask_up=up_quote.best_ask if up_quote and up_quote.best_ask is not None else 0.0,
            bid_down=down_quote.best_bid if down_quote and down_quote.best_bid is not None else 0.0,
            ask_down=down_quote.best_ask if down_quote and down_quote.best_ask is not None else 0.0,
            edge=_edge_for_action(action=action, decision=decision, side_signal=side_signal),
            order_id=action.order_id or "-",
            side=_side_to_label(action.side),
            price=side_signal.desired_price if side_signal and side_signal.desired_price is not None else 0.0,
            size=side_signal.size if side_signal is not None else 0.0,
            status=action.action_type.value,
            settlement_outcome=None,
            pnl=None,
        )
        self._reporter.append(record)
        if self._paper_result_tracker is not None:
            self._paper_result_tracker.on_record(record)

        paper_fill_record = self._build_paper_fill_record(
            action=action,
            side_signal=side_signal,
            base_record=record,
            up_quote=up_quote,
            down_quote=down_quote,
        )
        if paper_fill_record is not None:
            self._execution_engine.mark_order_filled(paper_fill_record.order_id)
            self._reporter.append(paper_fill_record)
            if self._paper_result_tracker is not None:
                self._paper_result_tracker.on_record(paper_fill_record)
            if self._policy_controller is not None:
                self._policy_controller.on_fill(
                    paper_fill_record,
                    selection=self._latest_policy_selection,
                )

    def _build_paper_fill_record(
        self,
        *,
        action: ExecutionAction,
        side_signal: IntentSignal | None,
        base_record: ExecutionLogRecord,
        up_quote: BestQuote | None,
        down_quote: BestQuote | None,
    ) -> ExecutionLogRecord | None:
        if self._config.mode != RuntimeMode.DRY_RUN:
            return None
        if action.action_type != ExecutionActionType.PLACE:
            return None
        if side_signal is None or side_signal.desired_price is None:
            return None
        if side_signal.size <= 0.0:
            return None

        best_ask: float | None = None
        normalized_action_token = str(action.token_id or "").strip()
        if up_quote is not None and str(up_quote.token_id).strip() == normalized_action_token:
            best_ask = up_quote.best_ask
        elif down_quote is not None and str(down_quote.token_id).strip() == normalized_action_token:
            best_ask = down_quote.best_ask
        elif action.side == OutcomeSide.UP:
            best_ask = up_quote.best_ask if up_quote is not None else None
        elif action.side == OutcomeSide.DOWN:
            best_ask = down_quote.best_ask if down_quote is not None else None

        if not _should_mark_dry_run_fill(limit_price=side_signal.desired_price, best_ask=best_ask):
            return None

        return ExecutionLogRecord(
            timestamp=base_record.timestamp,
            market_id=base_record.market_id,
            K=base_record.K,
            S_t=base_record.S_t,
            tau=base_record.tau,
            sigma=base_record.sigma,
            q_up=base_record.q_up,
            bid_up=base_record.bid_up,
            ask_up=base_record.ask_up,
            bid_down=base_record.bid_down,
            ask_down=base_record.ask_down,
            edge=base_record.edge,
            order_id=base_record.order_id,
            side=base_record.side,
            price=base_record.price,
            size=base_record.size,
            status="paper_fill",
            settlement_outcome=None,
            pnl=None,
        )

    async def _reconcile_paper_settlements(self, *, force: bool = False, now: datetime | None = None) -> None:
        if self._paper_settlement is None:
            return
        now_ts = now or utc_now()
        if (
            not force
            and self._next_paper_settlement_check is not None
            and now_ts < self._next_paper_settlement_check
        ):
            return
        try:
            appended_records = await self._paper_settlement.reconcile_records()
            for record in appended_records:
                if self._paper_result_tracker is not None:
                    self._paper_result_tracker.on_record(record)
                if self._policy_controller is not None:
                    reward = self._policy_controller.on_settlement(
                        record,
                        risk_limit_breach=(
                            self._external_kill_switch_latched
                            or self._execution_engine.kill_switch.active
                        ),
                    )
                    if reward is not None:
                        LOGGER.info("policy_reward order_id=%s reward=%.8f", record.order_id, reward)
            appended = len(appended_records)
            if appended > 0:
                LOGGER.info("paper_settle_appended=%s", appended)
        except Exception as exc:
            LOGGER.warning("Paper settlement reconciliation failed: %s", exc)
        finally:
            self._next_paper_settlement_check = now_ts + self._paper_settlement_interval

    def _log_tick_summary(
        self,
        *,
        tick_count: int,
        tau_years: float,
        decision: DecisionOutcome | None,
        arb_decision: CompleteSetArbDecision | None,
        up_quote: BestQuote | None,
        down_quote: BestQuote | None,
        fee_rate: float,
        action_count: int,
        selected_profile_id: str,
        applied_profile_id: str,
    ) -> None:
        q_up = decision.q_up if decision is not None else 0.5
        if arb_decision is not None and arb_decision.should_trade:
            edge_up = arb_decision.profit_per_pair
            edge_down = arb_decision.profit_per_pair
            metrics = {
                "up_kelly": 0.0,
                "down_kelly": 0.0,
                "up_notional": arb_decision.notional_up,
                "down_notional": arb_decision.notional_down,
            }
        else:
            edge_up = decision.up.edge if decision is not None else 0.0
            edge_down = decision.down.edge if decision is not None else 0.0
            metrics = _tick_decision_metrics(decision)
        LOGGER.info(
            (
                "tick=%s q_up=%.4f tau_years=%.8f ask_up=%.4f ask_down=%.4f "
                "edge_up=%.4f edge_down=%.4f "
                "kelly_up=%.4f kelly_down=%.4f notional_up=%.2f notional_down=%.2f "
                "arb_active=%s ask_sum=%.4f arb_profit=%.4f "
                "fee=%.6f actions=%s kill_switch=%s policy_selected=%s policy_applied=%s"
            ),
            tick_count,
            q_up,
            tau_years,
            up_quote.best_ask if up_quote and up_quote.best_ask is not None else 0.0,
            down_quote.best_ask if down_quote and down_quote.best_ask is not None else 0.0,
            edge_up,
            edge_down,
            metrics["up_kelly"],
            metrics["down_kelly"],
            metrics["up_notional"],
            metrics["down_notional"],
            arb_decision.should_trade if arb_decision is not None else False,
            arb_decision.ask_sum if arb_decision is not None else 0.0,
            arb_decision.profit_per_pair if arb_decision is not None else 0.0,
            fee_rate,
            action_count,
            self._execution_engine.kill_switch.active,
            selected_profile_id,
            applied_profile_id,
        )

    def _require_market(self) -> MarketCandidate:
        if self._market is None:
            raise RuntimeError("market not initialized")
        return self._market

    def _require_orderbook(self) -> ClobOrderbookAdapter:
        if self._orderbook is None:
            raise RuntimeError("orderbook feed not initialized")
        return self._orderbook

    def _require_underlying(self) -> BinanceUnderlyingAdapter:
        if self._underlying is None:
            raise RuntimeError("underlying feed not initialized")
        return self._underlying

    def _set_market_tick_sizes(self, tick_sizes: Mapping[str, float]) -> None:
        normalized: dict[str, float] = {}
        for token_id, tick in tick_sizes.items():
            key = str(token_id).strip()
            if not key:
                continue
            if tick <= 0.0:
                continue
            normalized[key] = tick
        self._token_tick_sizes = normalized
        if not normalized:
            return
        min_tick = min(normalized.values())
        self._execution_engine.set_price_epsilon(max(min_tick / 2.0, 1e-9))

    def _discover_market_tick_sizes(self, market: MarketCandidate) -> dict[str, float]:
        resolved = _extract_tick_sizes_from_market(market)
        adapter = self._execution_engine._adapter
        for token_id in (market.token_ids.up_token_id, market.token_ids.down_token_id):
            if token_id in resolved:
                continue
            tick_size = adapter.get_tick_size(token_id)
            if tick_size is None or tick_size <= 0.0:
                continue
            resolved[token_id] = tick_size
        return resolved

    def _run_live_heartbeat(self, *, now: datetime, force: bool = False) -> None:
        if self._config.mode != RuntimeMode.LIVE:
            return
        if not force and self._next_heartbeat_check is not None and now < self._next_heartbeat_check:
            return

        heartbeat_ok = self._execution_engine._adapter.send_heartbeat()
        if not heartbeat_ok:
            self._trigger_external_kill_switch(
                reason=KillSwitchReason.HEARTBEAT_FAILED,
                now=now,
                message="Heartbeat failed",
            )

        interval = max(1.0, self._config.safety.heartbeat_interval_seconds)
        self._next_heartbeat_check = now + timedelta(seconds=interval)

    def _reconcile_live_open_orders(self, *, now: datetime, force: bool = False) -> None:
        if self._config.mode != RuntimeMode.LIVE:
            return
        if not force and self._next_open_order_reconcile_check is not None and now < self._next_open_order_reconcile_check:
            return

        try:
            venue_open_ids = self._list_open_order_ids_with_retry()
        except Exception:
            self._trigger_external_kill_switch(
                reason=KillSwitchReason.ORDER_RECONCILIATION_FAILED,
                now=now,
                message="Open-order reconciliation failed",
            )
            interval = max(1.0, self._config.safety.open_order_reconcile_interval_seconds)
            self._next_open_order_reconcile_check = now + timedelta(seconds=interval)
            return

        active_intents = self._execution_engine.active_intents()
        local_open_ids = {intent.order_id for intent in active_intents.values()}
        intent_by_order_id = {intent.order_id: intent for intent in active_intents.values()}
        shared_open_ids = local_open_ids & venue_open_ids
        orphaned_ids = venue_open_ids - local_open_ids
        missing_ids = local_open_ids - venue_open_ids

        if orphaned_ids:
            adapter = self._execution_engine._adapter
            if not self._config.safety.cancel_orphan_orders:
                self._trigger_external_kill_switch(
                    reason=KillSwitchReason.ORDER_RECONCILIATION_FAILED,
                    now=now,
                    message="Found orphaned open order(s); manual review required",
                    cancel_venue_orders=False,
                )
            else:
                for order_id in orphaned_ids:
                    try:
                        result = adapter.cancel_order(order_id)
                    except Exception:
                        self._trigger_external_kill_switch(
                            reason=KillSwitchReason.ORDER_RECONCILIATION_FAILED,
                            now=now,
                            message="Failed to cancel orphaned open order",
                        )
                        break
                    if result.canceled:
                        continue
                    self._trigger_external_kill_switch(
                        reason=KillSwitchReason.ORDER_RECONCILIATION_FAILED,
                        now=now,
                        message="Found orphaned open order that could not be canceled",
                    )
                    break

        for order_id in shared_open_ids:
            intent = intent_by_order_id.get(order_id)
            if intent is None:
                continue
            try:
                fill_snapshot = self._poll_live_fill_snapshot_with_retry(order_id=order_id, intent=intent)
            except Exception:
                self._trigger_external_kill_switch(
                    reason=KillSwitchReason.ORDER_RECONCILIATION_FAILED,
                    now=now,
                    message="Open-order fill polling failed",
                )
                interval = max(1.0, self._config.safety.open_order_reconcile_interval_seconds)
                self._next_open_order_reconcile_check = now + timedelta(seconds=interval)
                return
            if fill_snapshot is None:
                continue
            self._append_live_fill_record(
                now=now,
                order_id=order_id,
                intent=intent,
                fill_snapshot=fill_snapshot,
            )
            if fill_snapshot.matched_size >= max(0.0, intent.size) - 1e-9:
                self._execution_engine.mark_order_filled(order_id)
                self._live_order_matched_sizes.pop(order_id, None)

        for order_id in missing_ids:
            intent = intent_by_order_id.get(order_id)
            if intent is None:
                self._execution_engine.drop_intent(order_id)
                self._live_order_matched_sizes.pop(order_id, None)
                continue
            try:
                fill_snapshot = self._poll_live_fill_snapshot_with_retry(order_id=order_id, intent=intent)
            except Exception:
                self._trigger_external_kill_switch(
                    reason=KillSwitchReason.ORDER_RECONCILIATION_FAILED,
                    now=now,
                    message="Missing-order fill polling failed",
                )
                interval = max(1.0, self._config.safety.open_order_reconcile_interval_seconds)
                self._next_open_order_reconcile_check = now + timedelta(seconds=interval)
                return
            if fill_snapshot is not None:
                self._append_live_fill_record(
                    now=now,
                    order_id=order_id,
                    intent=intent,
                    fill_snapshot=fill_snapshot,
                )
                if fill_snapshot.matched_size >= max(0.0, intent.size) - 1e-9:
                    self._execution_engine.mark_order_filled(order_id)
                else:
                    self._execution_engine.drop_intent(order_id)
            else:
                self._execution_engine.drop_intent(order_id)
            self._live_order_matched_sizes.pop(order_id, None)

        if orphaned_ids or missing_ids:
            LOGGER.warning(
                "open_order_reconciliation orphaned=%s missing=%s",
                len(orphaned_ids),
                len(missing_ids),
            )

        interval = max(1.0, self._config.safety.open_order_reconcile_interval_seconds)
        self._next_open_order_reconcile_check = now + timedelta(seconds=interval)

    def _list_open_order_ids_with_retry(self) -> set[str]:
        adapter = self._execution_engine._adapter
        last_error: Exception | None = None
        for _ in range(max(1, LIVE_RECON_MAX_RETRIES)):
            try:
                return adapter.list_open_order_ids()
            except Exception as exc:
                last_error = exc
                continue
        if last_error is not None:
            raise last_error
        return set()

    def _poll_live_fill_snapshot_with_retry(
        self,
        *,
        order_id: str,
        intent: ActiveIntent,
    ) -> LiveFillSnapshot | None:
        last_error: Exception | None = None
        for _ in range(max(1, LIVE_RECON_MAX_RETRIES)):
            try:
                return self._poll_live_fill_snapshot(order_id=order_id, intent=intent)
            except Exception as exc:
                last_error = exc
                continue
        if last_error is not None:
            raise last_error
        return None

    def _poll_live_fill_snapshot(self, *, order_id: str, intent: ActiveIntent) -> LiveFillSnapshot | None:
        if self._config.mode != RuntimeMode.LIVE:
            return None
        adapter = self._execution_engine._adapter
        get_order = getattr(adapter, "get_order", None)
        if not callable(get_order):
            return None
        order_payload = get_order(order_id)
        if not isinstance(order_payload, Mapping):
            return None

        order_view = _extract_live_order_view(order_payload)
        if order_view is None:
            return None

        matched_size = _extract_float_by_keys(order_view, ("size_matched", "matched_size")) or 0.0
        price = _extract_float_by_keys(order_view, ("price",))
        status = _extract_text_by_keys(order_view, ("status",))
        associated_trade_ids = _extract_trade_ids_by_keys(order_view, ("associate_trades",))

        trade_price, trade_size = self._poll_live_trade_details(
            order_id=order_id,
            trade_ids=associated_trade_ids,
        )
        if trade_size is not None and trade_size > 0.0:
            matched_size = trade_size
        if trade_price is not None and trade_price > 0.0:
            price = trade_price

        if matched_size <= 0.0 and _is_live_order_filled_status(status):
            matched_size = intent.size
        if matched_size <= 0.0:
            return None
        bounded_size = min(max(0.0, matched_size), max(0.0, intent.size))
        if bounded_size <= 0.0:
            return None
        return LiveFillSnapshot(matched_size=bounded_size, price=price)

    def _poll_live_trade_details(
        self,
        *,
        order_id: str,
        trade_ids: Sequence[str],
    ) -> tuple[float | None, float | None]:
        if not trade_ids:
            return None, None
        adapter = self._execution_engine._adapter
        get_trade = getattr(adapter, "get_trade", None)
        if not callable(get_trade):
            return None, None

        total_size = 0.0
        total_notional = 0.0
        fallback_price: float | None = None
        for trade_id in trade_ids:
            try:
                payload = get_trade(trade_id)
            except Exception:
                continue
            if not isinstance(payload, Mapping):
                continue
            trade_view = _extract_live_trade_view(payload)
            if trade_view is None:
                continue
            price = _extract_float_by_keys(trade_view, ("price",))
            if price is not None and price > 0.0:
                fallback_price = price
            matched_amount = _extract_trade_match_amount(trade_view, order_id=order_id)
            if price is None or price <= 0.0:
                continue
            if matched_amount is None or matched_amount <= 0.0:
                continue
            total_size += matched_amount
            total_notional += matched_amount * price

        if total_size > 0.0:
            return (total_notional / total_size), total_size
        return fallback_price, None

    def _append_live_fill_record(
        self,
        *,
        now: datetime,
        order_id: str,
        intent: ActiveIntent,
        fill_snapshot: LiveFillSnapshot,
    ) -> None:
        normalized_order_id = str(order_id).strip()
        if not normalized_order_id:
            return
        bounded_matched = min(max(0.0, fill_snapshot.matched_size), max(0.0, intent.size))
        previous_matched = self._live_order_matched_sizes.get(normalized_order_id, 0.0)
        if bounded_matched <= previous_matched + 1e-12:
            return
        delta_size = bounded_matched - previous_matched
        self._live_order_matched_sizes[normalized_order_id] = bounded_matched

        fill_price = fill_snapshot.price if fill_snapshot.price is not None and fill_snapshot.price > 0.0 else intent.price
        record = ExecutionLogRecord(
            timestamp=now,
            market_id=intent.market_id,
            K=0.0,
            S_t=0.0,
            tau=0.0,
            sigma=0.0,
            q_up=0.5,
            bid_up=0.0,
            ask_up=0.0,
            bid_down=0.0,
            ask_down=0.0,
            edge=intent.edge,
            order_id=normalized_order_id,
            side=_side_to_label(intent.side),
            price=fill_price,
            size=bounded_matched,
            status="live_fill",
            settlement_outcome=None,
            pnl=None,
        )
        self._reporter.append(record)
        if self._paper_result_tracker is not None:
            self._paper_result_tracker.on_record(record)
        direction = 1.0 if intent.order_side == VenueOrderSide.BUY else -1.0
        self._apply_live_market_side_exposure(
            market_id=intent.market_id,
            side=intent.side,
            delta_tokens=delta_size * direction,
        )

    def _enforce_near_expiry_cleanup(
        self,
        *,
        market_id: str,
        seconds_to_expiry: float,
        now: datetime,
    ) -> bool:
        if self._near_expiry_cleanup_done:
            return False
        threshold = max(0.0, self._config.safety.entry_block_window_seconds)
        if seconds_to_expiry > threshold:
            return False

        try:
            self._execution_engine.cancel_all_intents(reason="near_expiry_cleanup")
        except Exception as exc:
            LOGGER.warning("near_expiry_cleanup failed to cancel local intents: %s", exc)
        if self._config.mode == RuntimeMode.LIVE:
            self._cancel_all_venue_orders(now=now)
        self._near_expiry_cleanup_done = True
        LOGGER.info("near_expiry_cleanup_applied market_id=%s seconds_to_expiry=%.2f", market_id, seconds_to_expiry)
        return True

    def _cancel_all_venue_orders(self, *, now: datetime) -> None:
        if self._config.mode != RuntimeMode.LIVE:
            return
        try:
            canceled_count = self._execution_engine._adapter.cancel_all_orders()
        except Exception:
            self._trigger_external_kill_switch(
                reason=KillSwitchReason.ORDER_RECONCILIATION_FAILED,
                now=now,
                message="cancel_all_orders failed",
            )
            return
        LOGGER.warning("cancel_all_orders_applied count=%s", canceled_count)

    def _trigger_external_kill_switch(
        self,
        *,
        reason: KillSwitchReason,
        now: datetime,
        message: str,
        cancel_venue_orders: bool = True,
    ) -> None:
        if self._external_kill_switch_latched:
            return
        self._external_kill_switch_latched = True
        try:
            self._execution_engine.latch_kill_switch(reason=reason, now=now)
        except Exception as exc:
            LOGGER.warning("kill_switch_latch_local_cancel_failed: %s", exc)
        if cancel_venue_orders:
            self._cancel_all_venue_orders(now=now)
        LOGGER.error("%s: kill switch latched.", message)

    def _decision_bankroll(self) -> float:
        tracker = self._paper_result_tracker
        base = tracker.available_bankroll() if tracker is not None else self._config.risk.bankroll
        return max(0.0, base - self._open_intent_notional_total())

    def _worst_case_loss_estimate(self) -> float:
        tracker = self._paper_result_tracker
        unsettled_notional = tracker.unsettled_notional() if tracker is not None else 0.0
        return max(0.0, unsettled_notional + self._open_intent_notional_total())

    def _entry_block_reason(self, *, now: datetime, market_id: str) -> str | None:
        if self._external_kill_switch_latched:
            return "external_kill_switch_latched"
        if self._position_mismatch_blocked:
            return "position_mismatch"

        tracker = self._paper_result_tracker
        if tracker is None:
            return None

        daily_loss_cap = self._config.risk.max_daily_loss
        if daily_loss_cap > 0.0:
            daily_realized = tracker.daily_realized_pnl(now)
            if daily_realized <= -daily_loss_cap:
                return "daily_loss_cap_reached"

        entry_limit = self._config.risk.max_entries_per_market
        if entry_limit > 0 and tracker.market_entry_count(market_id) >= entry_limit:
            return "market_entry_limit"

        return None

    def _cap_order_notional_for_market(self, *, market_id: str, suggested_notional: float) -> float:
        capped = max(0.0, suggested_notional)
        market_cap = self._config.risk.max_market_notional
        if market_cap <= 0.0:
            return capped
        tracker = self._paper_result_tracker
        used_notional = tracker.market_unsettled_notional(market_id) if tracker is not None else 0.0
        used_notional += self._open_intent_notional_for_market(market_id)
        remaining = max(0.0, market_cap - used_notional)
        return min(capped, remaining)

    def _daily_realized_pnl(self, now: datetime) -> float:
        tracker = self._paper_result_tracker
        if tracker is None:
            return 0.0
        return tracker.daily_realized_pnl(now)

    def _market_effective_notional(self, market_id: str) -> float:
        tracker = self._paper_result_tracker
        unsettled = tracker.market_unsettled_notional(market_id) if tracker is not None else 0.0
        return unsettled + self._open_intent_notional_for_market(market_id)

    def _market_remaining_notional(self, market_id: str) -> float:
        cap = max(0.0, self._config.risk.max_market_notional)
        if cap <= 0.0:
            return max(0.0, self._decision_bankroll())
        used = self._market_effective_notional(market_id)
        return max(0.0, cap - used)

    def _open_intent_notional_total(self) -> float:
        return sum(max(0.0, intent.price * intent.size) for intent in self._execution_engine.active_intents().values())

    def _open_intent_notional_for_market(self, market_id: str) -> float:
        normalized_market_id = str(market_id).strip()
        if not normalized_market_id:
            return 0.0
        return sum(
            max(0.0, intent.price * intent.size)
            for intent in self._execution_engine.active_intents().values()
            if intent.market_id == normalized_market_id
        )

    def _has_local_market_exposure(self, market_id: str) -> bool:
        if self._open_intent_notional_for_market(market_id) > 0.0:
            return True
        if self._market_side_exposure_tokens(market_id, OutcomeSide.UP) > 0.0:
            return True
        if self._market_side_exposure_tokens(market_id, OutcomeSide.DOWN) > 0.0:
            return True
        tracker = self._paper_result_tracker
        if tracker is None:
            return False
        return tracker.market_unsettled_notional(market_id) > 0.0

    def _has_active_intent_for_token(self, *, market_id: str, token_id: str) -> bool:
        normalized_market_id = str(market_id).strip()
        normalized_token_id = str(token_id).strip()
        if not normalized_market_id or not normalized_token_id:
            return False
        return (normalized_market_id, normalized_token_id) in self._execution_engine.active_intents()

    def _apply_live_market_side_exposure(self, *, market_id: str, side: OutcomeSide, delta_tokens: float) -> None:
        normalized_market_id = str(market_id).strip()
        if not normalized_market_id:
            return
        key = (normalized_market_id, side)
        next_value = self._live_market_side_exposure_tokens.get(key, 0.0) + float(delta_tokens)
        if next_value <= 1e-12:
            self._live_market_side_exposure_tokens.pop(key, None)
            return
        self._live_market_side_exposure_tokens[key] = next_value

    def _market_side_exposure_tokens(self, market_id: str, side: OutcomeSide) -> float:
        normalized_market_id = str(market_id).strip()
        if not normalized_market_id:
            return 0.0
        return max(0.0, self._live_market_side_exposure_tokens.get((normalized_market_id, side), 0.0))


def install_signal_handlers(stop_event: asyncio.Event) -> None:
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, stop_event.set)
        except NotImplementedError:
            signal.signal(sig, lambda _signum, _frame: stop_event.set())


async def run_async(config: AppConfig) -> None:
    stop_event = asyncio.Event()
    install_signal_handlers(stop_event)
    app = PM1HEdgeTraderApp(config)
    LOGGER.info("Starting trader mode=%s", config.mode.value)
    await app.run(stop_event)
    LOGGER.info("Shutdown complete.")


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    live_auth = load_polymarket_live_auth_from_env(os.environ)
    selected_mode = RuntimeMode(args.mode)
    live_balance: LiveBalanceSnapshot | None = None
    effective_bankroll = args.bankroll
    if selected_mode == RuntimeMode.LIVE:
        validate_live_mode_credentials(AppConfig(mode=RuntimeMode.LIVE, polymarket_live_auth=live_auth))
        live_balance = fetch_live_balance_snapshot(
            auth=live_auth,
            host="https://clob.polymarket.com",
        )
        effective_bankroll = live_balance.bankroll
    config = build_config(
        mode=selected_mode,
        binance_symbol=args.binance_symbol,
        market_slug=args.market_slug,
        tick_seconds=args.tick_seconds,
        max_ticks=args.max_ticks,
        bankroll=effective_bankroll,
        edge_min=args.edge_min,
        edge_buffer=args.edge_buffer,
        cost_rate=args.cost_rate,
        kelly_fraction=args.kelly_fraction,
        f_cap=args.f_cap,
        min_order_notional=args.min_order_notional,
        max_market_notional=args.max_market_notional,
        max_daily_loss=args.max_daily_loss,
        max_worst_case_loss=args.max_worst_case_loss,
        max_live_drawdown=args.max_live_drawdown,
        live_balance_refresh_seconds=args.live_balance_refresh_seconds,
        max_entries_per_market=args.max_entries_per_market,
        rv_fallback=args.rv_fallback,
        rv_interval=args.rv_interval,
        rv_ewma_half_life=args.rv_ewma_half_life,
        rv_short_interval=args.rv_short_interval,
        rv_short_lookback_hours=args.rv_short_lookback_hours,
        rv_short_ewma_half_life=args.rv_short_ewma_half_life,
        rv_tau_switch_seconds=args.rv_tau_switch_seconds,
        sigma_weight=args.sigma_weight,
        iv_override=args.iv,
        enable_deribit_iv=args.enable_deribit_iv,
        deribit_iv_resolution_minutes=args.deribit_iv_resolution_minutes,
        deribit_iv_lookback_hours=args.deribit_iv_lookback_hours,
        deribit_iv_refresh_seconds=args.deribit_iv_refresh_seconds,
        enable_policy_bandit=args.enable_policy_bandit,
        policy_shadow_mode=args.policy_shadow_mode,
        policy_exploration_epsilon=args.policy_exploration_epsilon,
        policy_ucb_c=args.policy_ucb_c,
        policy_reward_turnover_lambda=args.policy_reward_turnover_lambda,
        policy_reward_risk_penalty=args.policy_reward_risk_penalty,
        policy_vol_ratio_threshold=args.policy_vol_ratio_threshold,
        policy_spread_tight_threshold=args.policy_spread_tight_threshold,
        enable_auto_claim=not args.disable_auto_claim,
        auto_claim_interval_seconds=args.auto_claim_interval_seconds,
        auto_claim_size_threshold=args.auto_claim_size_threshold,
        auto_claim_cooldown_seconds=args.auto_claim_cooldown_seconds,
        auto_claim_tx_timeout_seconds=args.auto_claim_tx_timeout_seconds,
        polygon_rpc_url=args.polygon_rpc_url,
        data_api_base_url=args.data_api_base_url,
        position_reconcile_interval_seconds=args.position_reconcile_interval_seconds,
        position_mismatch_policy=args.position_mismatch_policy,
        position_size_threshold=args.position_size_threshold,
        position_size_relative_tolerance=args.position_size_relative_tolerance,
        adopt_existing_positions=args.adopt_existing_positions,
        adopt_existing_positions_policy=args.adopt_existing_positions_policy,
        hard_kill_on_daily_loss=args.hard_kill_on_daily_loss,
        enable_complete_set_arb=args.enable_complete_set_arb,
        arb_min_profit=args.arb_min_profit,
        arb_max_notional=args.arb_max_notional,
        arb_fill_timeout_seconds=args.arb_fill_timeout_seconds,
        cancel_orphan_orders=args.cancel_orphan_orders,
        log_dir=args.log_dir,
        enable_websocket=not args.disable_websocket,
        fresh_start=args.fresh_start,
        polymarket_live_auth=live_auth,
    )
    configure_logging(config)
    if live_balance is not None:
        LOGGER.info(
            "live_bankroll_auto balance=%.6f allowance=%.6f bankroll=%.6f",
            live_balance.balance or 0.0,
            live_balance.allowance or 0.0,
            live_balance.bankroll,
        )
    try:
        validate_live_mode_credentials(config)
        asyncio.run(run_async(config))
    except RuntimeError as exc:
        LOGGER.error("%s", exc)
        return 1
    except KeyboardInterrupt:
        LOGGER.info("Interrupted by user.")
    return 0


def _interval_to_seconds(interval: str) -> int:
    text = interval.strip().lower()
    if len(text) < 2:
        raise ValueError(f"Unsupported interval format: {interval!r}")
    unit = text[-1]
    magnitude_text = text[:-1]
    try:
        magnitude = int(magnitude_text)
    except ValueError as exc:
        raise ValueError(f"Unsupported interval format: {interval!r}") from exc
    if magnitude <= 0:
        raise ValueError(f"Interval magnitude must be positive: {interval!r}")

    if unit == "m":
        return magnitude * 60
    if unit == "h":
        return magnitude * 60 * 60
    if unit == "d":
        return magnitude * 24 * 60 * 60
    raise ValueError(f"Unsupported interval unit: {interval!r}")


def _compute_kline_limit(*, lookback_hours: int, interval_seconds: int) -> int:
    if interval_seconds <= 0:
        return 9
    lookback_seconds = max(1, lookback_hours) * 60 * 60
    periods = max(2, int(ceil(lookback_seconds / interval_seconds)))
    return max(9, periods + 1)


def _ewma_std(values: Sequence[float], *, half_life: float) -> float:
    if len(values) < 1:
        return 0.0
    if half_life <= 0.0:
        return _sample_std(values)
    decay = 0.5 ** (1.0 / half_life)
    variance = 0.0
    initialized = False
    for value in values:
        squared = value * value
        if not initialized:
            variance = squared
            initialized = True
            continue
        variance = (decay * variance) + ((1.0 - decay) * squared)
    return sqrt(max(variance, 0.0))


def _tau_short_weight(seconds_to_expiry: float, tau_switch_seconds: float) -> float:
    switch = max(tau_switch_seconds, 1e-9)
    tte = max(seconds_to_expiry, 0.0)
    raw = (switch - tte) / switch
    return min(1.0, max(0.0, raw))


def _extract_deribit_iv_close(payload: object) -> float | None:
    if not isinstance(payload, Mapping):
        return None
    result = payload.get("result")
    if not isinstance(result, Mapping):
        return None
    rows = result.get("data")
    if not isinstance(rows, list):
        return None
    for row in reversed(rows):
        if not isinstance(row, list) or len(row) < 5:
            continue
        close = _safe_float(row[4])
        if close is None or close <= 0.0:
            continue
        return close
    return None


def _extract_closes(payload: object) -> list[float]:
    if not isinstance(payload, list):
        return []
    closes: list[float] = []
    for row in payload:
        if not isinstance(row, list) or len(row) < 5:
            continue
        close_value = _safe_float(row[4])
        if close_value is None or close_value <= 0.0:
            continue
        closes.append(close_value)
    return closes


def _log_returns(prices: Sequence[float]) -> list[float]:
    if len(prices) < 2:
        return []
    returns: list[float] = []
    for index in range(1, len(prices)):
        prev = prices[index - 1]
        cur = prices[index]
        if prev <= 0.0 or cur <= 0.0:
            continue
        returns.append(log(cur / prev))
    return returns


def _sample_std(values: Sequence[float]) -> float:
    if len(values) < 2:
        return 0.0
    mean = sum(values) / len(values)
    variance = sum((value - mean) ** 2 for value in values) / (len(values) - 1)
    return sqrt(max(variance, 0.0))


def _safe_float(value: object) -> float | None:
    if isinstance(value, (float, int)):
        return float(value)
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        try:
            return float(text)
        except ValueError:
            return None
    return None


def _is_data_ready(
    up_quote: BestQuote | None,
    down_quote: BestQuote | None,
    underlying: UnderlyingSnapshot,
) -> bool:
    if up_quote is None or down_quote is None:
        return False
    if up_quote.best_bid is None or up_quote.best_ask is None:
        return False
    if down_quote.best_bid is None or down_quote.best_ask is None:
        return False
    if underlying.candle_open is None or underlying.last_price is None:
        return False
    return True


def _is_book_consistent(up_quote: BestQuote | None, down_quote: BestQuote | None) -> bool:
    for quote in (up_quote, down_quote):
        if quote is None:
            return False
        if quote.best_bid is None or quote.best_ask is None:
            return False
        if quote.best_bid < 0.0 or quote.best_bid > 1.0:
            return False
        if quote.best_ask < 0.0 or quote.best_ask > 1.0:
            return False
        if quote.best_bid > quote.best_ask:
            return False
    return True


def _compute_data_age_seconds(
    *,
    now: datetime,
    up_quote: BestQuote | None,
    down_quote: BestQuote | None,
    underlying: UnderlyingSnapshot,
) -> float:
    ages: list[float] = []
    if up_quote is not None:
        ages.append((now - up_quote.updated_at).total_seconds())
    if down_quote is not None:
        ages.append((now - down_quote.updated_at).total_seconds())
    if underlying.price_time is not None:
        ages.append((now - underlying.price_time).total_seconds())
    if not ages:
        return 1e9
    return max(ages)


def _compute_clock_drift_seconds(*, now: datetime, underlying: UnderlyingSnapshot) -> float:
    """
    Conservative clock drift proxy.

    We treat feed timestamps significantly in the future as local clock drift.
    """
    future_offsets: list[float] = []
    for feed_ts in (underlying.price_time, underlying.candle_open_time):
        if feed_ts is None:
            continue
        offset = (feed_ts - now).total_seconds()
        if offset > 0.0:
            future_offsets.append(offset)
    return max(future_offsets) if future_offsets else 0.0


def _limit_price(
    *,
    ask: float,
    fair_probability: float,
    edge_buffer: float,
    tick_size: float | None = None,
) -> float | None:
    target = fair_probability - edge_buffer
    if target <= 0.0:
        return None
    price = min(ask, target)
    # Bound to valid probability price ticks.
    bounded = min(0.999, max(0.001, price))
    tick = tick_size if tick_size is not None and tick_size > 0.0 else 0.0001
    return _round_down_to_tick(bounded, tick)


def _edge_for_action(
    action: ExecutionAction,
    decision: DecisionOutcome | None,
    side_signal: IntentSignal | None,
) -> float:
    if side_signal is not None:
        return float(side_signal.edge)
    if decision is None or action.side is None:
        return 0.0
    if action.side == OutcomeSide.UP:
        return decision.up.edge
    return decision.down.edge


def _side_to_label(side: OutcomeSide | None) -> str:
    if side is None:
        return "none"
    if side == OutcomeSide.UP:
        return "up"
    return "down"


def _tick_decision_metrics(decision: DecisionOutcome | None) -> dict[str, float]:
    if decision is None:
        return {
            "up_kelly": 0.0,
            "down_kelly": 0.0,
            "up_notional": 0.0,
            "down_notional": 0.0,
        }
    return {
        "up_kelly": decision.up.kelly_fraction,
        "down_kelly": decision.down.kelly_fraction,
        "up_notional": decision.up.order_size,
        "down_notional": decision.down.order_size,
    }


def _extract_live_order_view(payload: Mapping[str, Any]) -> Mapping[str, Any] | None:
    order = payload.get("order")
    if isinstance(order, Mapping):
        return order
    if _extract_text_by_keys(payload, ("status",)) is not None:
        return payload
    for key in ("data", "result"):
        nested = payload.get(key)
        if isinstance(nested, Mapping):
            nested_order = _extract_live_order_view(nested)
            if nested_order is not None:
                return nested_order
    return None


def _extract_live_trade_view(payload: Mapping[str, Any]) -> Mapping[str, Any] | None:
    if _extract_text_by_keys(payload, ("id", "trade_id", "tradeID")) is not None:
        return payload
    for key in ("trade", "data", "result"):
        nested = payload.get(key)
        if isinstance(nested, Mapping):
            nested_trade = _extract_live_trade_view(nested)
            if nested_trade is not None:
                return nested_trade
        if isinstance(nested, list):
            for item in nested:
                if not isinstance(item, Mapping):
                    continue
                nested_trade = _extract_live_trade_view(item)
                if nested_trade is not None:
                    return nested_trade
    return None


def _extract_text_by_keys(payload: Mapping[str, Any], keys: Sequence[str]) -> str | None:
    for key in keys:
        value = payload.get(key)
        if isinstance(value, str):
            normalized = value.strip()
            if normalized:
                return normalized
    return None


def _extract_float_by_keys(payload: Mapping[str, Any], keys: Sequence[str]) -> float | None:
    for key in keys:
        value = payload.get(key)
        parsed = _safe_float(value)
        if parsed is not None:
            return parsed
    return None


def _extract_trade_ids_by_keys(payload: Mapping[str, Any], keys: Sequence[str]) -> list[str]:
    for key in keys:
        value = payload.get(key)
        if isinstance(value, list):
            collected: list[str] = []
            for item in value:
                if not isinstance(item, str):
                    continue
                normalized = item.strip()
                if normalized:
                    collected.append(normalized)
            if collected:
                return collected
        if isinstance(value, str):
            normalized = value.strip()
            if normalized:
                return [normalized]
    return []


def _extract_trade_match_amount(trade_view: Mapping[str, Any], *, order_id: str) -> float | None:
    maker_orders = trade_view.get("maker_orders")
    if isinstance(maker_orders, list):
        for maker_order in maker_orders:
            if not isinstance(maker_order, Mapping):
                continue
            maker_order_id = _extract_text_by_keys(maker_order, ("order_id", "id", "orderId"))
            if maker_order_id is None or maker_order_id != order_id:
                continue
            matched = _extract_float_by_keys(maker_order, ("matched_amount", "size", "amount"))
            if matched is not None and matched > 0.0:
                return matched
    return _extract_float_by_keys(trade_view, ("matched_amount", "size", "amount"))


def _is_live_order_filled_status(status: str | None) -> bool:
    if status is None:
        return False
    normalized = status.strip().lower()
    return normalized in {"matched", "filled", "executed", "confirmed", "mined"}


def _reset_execution_csv(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=ExecutionLogRecord.csv_fields())
        writer.writeheader()


def _should_mark_dry_run_fill(*, limit_price: float, best_ask: float | None) -> bool:
    if best_ask is None:
        return False
    return (limit_price + 1e-9) >= best_ask


def _round_down_to_tick(value: float, tick_size: float) -> float:
    if tick_size <= 0.0:
        return round(value, 4)
    steps = int((value + 1e-12) / tick_size)
    rounded = steps * tick_size
    bounded = min(0.999, max(0.001, rounded))
    precision = max(0, _decimal_places(tick_size))
    return round(bounded, precision)


def _decimal_places(value: float) -> int:
    text = f"{value:.10f}".rstrip("0")
    if "." not in text:
        return 0
    return len(text.split(".", maxsplit=1)[1])


def _extract_tick_sizes_from_market(market: MarketCandidate) -> dict[str, float]:
    tick_sizes: dict[str, float] = {}
    raw = market.raw_market if isinstance(market.raw_market, Mapping) else {}
    token_rows = raw.get("tokens")
    if isinstance(token_rows, list):
        for token in token_rows:
            if not isinstance(token, Mapping):
                continue
            token_id = _extract_token_id(token)
            tick_size = _extract_tick_size(token)
            if token_id is None or tick_size is None:
                continue
            tick_sizes[token_id] = tick_size

    default_tick = _extract_tick_size(raw)
    if default_tick is not None:
        for token_id in (market.token_ids.up_token_id, market.token_ids.down_token_id):
            tick_sizes.setdefault(token_id, default_tick)

    return tick_sizes


def _extract_token_id(payload: Mapping[str, Any]) -> str | None:
    for key in ("token_id", "tokenId", "clobTokenId", "asset_id", "id"):
        value = payload.get(key)
        if isinstance(value, str):
            normalized = value.strip()
            if normalized:
                return normalized
    return None


def _extract_tick_size(payload: Mapping[str, Any]) -> float | None:
    for key in (
        "tickSize",
        "tick_size",
        "priceIncrement",
        "price_increment",
        "minPriceIncrement",
        "min_price_increment",
        "minimumTickSize",
        "minimum_tick_size",
        "minimumPriceIncrement",
    ):
        value = payload.get(key)
        parsed = _safe_float(value)
        if parsed is not None and parsed > 0.0:
            return parsed
    return None


if __name__ == "__main__":
    raise SystemExit(main())
