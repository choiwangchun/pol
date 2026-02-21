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
from math import log, sqrt
from pathlib import Path
from typing import Sequence

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
    DryRunExecutionAdapter,
    ExecutionAction,
    ExecutionActionType,
    ExecutionConfig as ExecutionEngineConfig,
    IntentSignal,
    LimitOrderExecutionEngine,
    PolymarketLiveExecutionAdapter,
    SafetySurface,
    Side,
)
from .feeds import BinanceUnderlyingAdapter, ClobOrderbookAdapter
from .feeds.http_client import AsyncJsonHttpClient
from .feeds.models import BestQuote, UnderlyingSnapshot
from .logger import CSVExecutionReporter, ExecutionLogRecord
from .markets import FeeRateGuardClient, GammaMarketDiscoveryClient, MarketCandidate
from .paper import PaperResultTracker, PaperSettlementCoordinator, PolymarketMarketResolutionResolver

LOGGER = logging.getLogger("pm1h_edge_trader")
SECONDS_PER_YEAR = 365.0 * 24.0 * 60.0 * 60.0


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
        "--kelly-fraction",
        type=float,
        default=0.25,
        help="Fractional Kelly multiplier.",
    )
    parser.add_argument("--f-cap", type=float, default=0.05, help="Max bankroll fraction per side.")
    parser.add_argument(
        "--min-order-notional",
        type=float,
        default=5.0,
        help="Minimum order notional in quote currency.",
    )
    parser.add_argument(
        "--rv-fallback",
        type=float,
        default=0.55,
        help="Fallback annualized RV when estimator fails.",
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
        "--disable-websocket",
        action="store_true",
        help="Disable websocket and use REST polling only.",
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


class RealizedVolEstimator:
    """Computes annualized realized volatility from Binance 1h closes."""

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
        http_client: AsyncJsonHttpClient | None = None,
    ) -> None:
        self._rest_base_url = rest_base_url.rstrip("/")
        self._symbol = symbol.upper()
        self._interval = interval
        self._lookback_hours = max(8, lookback_hours)
        self._refresh_seconds = max(5.0, refresh_seconds)
        self._fallback_sigma = max(fallback_sigma, floor_sigma)
        self._floor_sigma = max(0.0, floor_sigma)
        self._http = http_client or AsyncJsonHttpClient()
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
                    "limit": self._lookback_hours + 1,
                },
            )
            closes = _extract_closes(rows)
            returns = _log_returns(closes)
            if len(returns) < 2:
                return self._fallback_sigma
            hourly_vol = _sample_std(returns)
            annualized = hourly_vol * sqrt(24.0 * 365.0)
            if annualized <= 0.0:
                return self._fallback_sigma
            return annualized
        except Exception:
            return self._fallback_sigma


class PM1HEdgeTraderApp:
    """Orchestrates market discovery, feeds, decisioning, execution, and logging."""

    def __init__(self, config: AppConfig) -> None:
        self._config = config

        self._discovery = GammaMarketDiscoveryClient(base_url=config.polymarket.gamma_base_url)
        self._fee_guard = FeeRateGuardClient(base_url=config.polymarket.clob_rest_base_url)
        self._reporter = CSVExecutionReporter(config.logging.execution_csv_path)

        decision_cfg = ProbabilityDecisionConfig(
            sigma_weight=config.volatility.sigma_weight,
            edge_min=config.decision.edge_min,
            edge_buffer_up=config.decision.edge_buffer_up,
            edge_buffer_down=config.decision.edge_buffer_down,
            kelly_fraction=config.risk.kelly_fraction,
            f_cap=config.risk.f_cap,
            min_order_size=config.risk.min_order_notional,
        )
        self._probability_engine = ProbabilityDecisionEngine(decision_cfg)

        execution_cfg = ExecutionEngineConfig(
            requote_interval_s=config.safety.requote_interval_seconds,
            max_intent_age_s=config.safety.max_intent_age_seconds,
            entry_block_window_s=config.safety.entry_block_window_seconds,
            max_data_age_s=config.safety.max_data_age_seconds,
            max_clock_drift_s=config.safety.max_clock_drift_seconds,
            require_book_match=config.safety.require_book_match,
            fee_must_be_zero=config.safety.fee_must_be_zero,
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

        self._rv_estimator = RealizedVolEstimator(
            rest_base_url=config.binance.rest_base_url,
            symbol=config.binance.symbol,
            interval=config.binance.interval,
            lookback_hours=config.volatility.rv_lookback_hours,
            refresh_seconds=config.volatility.rv_refresh_seconds,
            fallback_sigma=config.volatility.rv_fallback,
            floor_sigma=config.volatility.rv_floor,
        )

        self._market: MarketCandidate | None = None
        self._orderbook: ClobOrderbookAdapter | None = None
        self._underlying: BinanceUnderlyingAdapter | None = None
        self._fee_state: FeeState | None = None
        self._paper_settlement: PaperSettlementCoordinator | None = None
        self._paper_result_tracker: PaperResultTracker | None = None
        self._paper_settlement_interval = timedelta(seconds=15)
        self._next_paper_settlement_check: datetime | None = None

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

        self._execution_engine.reset_kill_switch()
        await self._refresh_fee_state(force=True, now=utc_now())
        await self._reconcile_paper_settlements(force=True, now=utc_now())
        LOGGER.info(
            "Selected market id=%s slug=%s start=%s end=%s up_token=%s down_token=%s",
            market.market_id,
            market.slug,
            market.timing.start.isoformat(),
            market.timing.end.isoformat(),
            market.token_ids.up_token_id,
            market.token_ids.down_token_id,
        )
        return True

    async def _shutdown(self) -> None:
        await self._deactivate_market()

    async def _deactivate_market(self) -> None:
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
        self._market = None

    async def _tick_once(self, *, tick_count: int) -> bool:
        market = self._require_market()
        now = utc_now()
        if now >= market.timing.end:
            return True

        await self._refresh_fee_state(now=now)
        max_fee_rate = self._fee_state.max_fee_rate if self._fee_state is not None else 0.0

        book = await self._require_orderbook().get_snapshot()
        underlying = await self._require_underlying().get_snapshot()
        up_quote = book.quotes.get(market.token_ids.up_token_id)
        down_quote = book.quotes.get(market.token_ids.down_token_id)

        data_ready = _is_data_ready(up_quote, down_quote, underlying)
        tau_years = max((market.timing.end - now).total_seconds(), 0.0) / SECONDS_PER_YEAR

        decision: DecisionOutcome | None = None
        sigma = self._config.volatility.rv_fallback
        if data_ready and tau_years > 0.0:
            sigma = await self._rv_estimator.get_sigma(now=now)
            decision = self._probability_engine.decide(
                spot=underlying.last_price or 0.0,
                strike=underlying.candle_open or 0.0,
                tau=tau_years,
                rv=sigma,
                iv=self._config.volatility.iv_override,
                ask_up=up_quote.best_ask or 0.0,
                ask_down=down_quote.best_ask or 0.0,
                bankroll=self._config.risk.bankroll,
            )
            sigma = decision.sigma

        safety = SafetySurface(
            fee_rate=max_fee_rate,
            data_age_s=_compute_data_age_seconds(now=now, up_quote=up_quote, down_quote=down_quote, underlying=underlying),
            clock_drift_s=_compute_clock_drift_seconds(now=now, underlying=underlying),
            book_is_consistent=_is_book_consistent(up_quote, down_quote),
        )

        up_signal = self._build_signal(
            market_id=market.market_id,
            token_id=market.token_ids.up_token_id,
            side_label="up",
            now=now,
            seconds_to_expiry=(market.timing.end - now).total_seconds(),
            quote=up_quote,
            decision=decision,
            data_ready=data_ready,
        )
        down_signal = self._build_signal(
            market_id=market.market_id,
            token_id=market.token_ids.down_token_id,
            side_label="down",
            now=now,
            seconds_to_expiry=(market.timing.end - now).total_seconds(),
            quote=down_quote,
            decision=decision,
            data_ready=data_ready,
        )

        all_actions: list[ExecutionAction] = []
        for signal in (up_signal, down_signal):
            result = self._execution_engine.process_signal(signal, safety, now=now)
            all_actions.extend(result.actions)
        stale_result = self._execution_engine.sweep_stale_intents(now=now)
        all_actions.extend(stale_result.actions)

        signal_map = {
            Side.BUY: up_signal,
            Side.SELL: down_signal,
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
        )
        await self._reconcile_paper_settlements(now=now)
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
    ) -> IntentSignal:
        side = Side.BUY if side_label == "up" else Side.SELL
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
                allow_entry=False,
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
        )
        share_size = 0.0
        if side_intent.order_size > 0.0 and desired_price is not None and desired_price > 0.0:
            share_size = side_intent.order_size / desired_price

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
            allow_entry=data_ready,
        )

    def _report_action(
        self,
        *,
        action: ExecutionAction,
        signal_map: dict[Side, IntentSignal],
        market: MarketCandidate,
        underlying: UnderlyingSnapshot,
        up_quote: BestQuote | None,
        down_quote: BestQuote | None,
        tau_years: float,
        sigma: float,
        decision: DecisionOutcome | None,
    ) -> None:
        side_signal = signal_map.get(action.side) if action.side is not None else None
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
            edge=_edge_for_action(action=action, decision=decision),
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
            self._reporter.append(paper_fill_record)
            if self._paper_result_tracker is not None:
                self._paper_result_tracker.on_record(paper_fill_record)

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
        if action.side == Side.BUY:
            best_ask = up_quote.best_ask if up_quote is not None else None
        elif action.side == Side.SELL:
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
        up_quote: BestQuote | None,
        down_quote: BestQuote | None,
        fee_rate: float,
        action_count: int,
    ) -> None:
        q_up = decision.q_up if decision is not None else 0.5
        edge_up = decision.up.edge if decision is not None else 0.0
        edge_down = decision.down.edge if decision is not None else 0.0
        metrics = _tick_decision_metrics(decision)
        LOGGER.info(
            (
                "tick=%s q_up=%.4f tau_years=%.8f ask_up=%.4f ask_down=%.4f "
                "edge_up=%.4f edge_down=%.4f "
                "kelly_up=%.4f kelly_down=%.4f notional_up=%.2f notional_down=%.2f "
                "fee=%.6f actions=%s kill_switch=%s"
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
            fee_rate,
            action_count,
            self._execution_engine.kill_switch.active,
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
    config = build_config(
        mode=RuntimeMode(args.mode),
        binance_symbol=args.binance_symbol,
        market_slug=args.market_slug,
        tick_seconds=args.tick_seconds,
        max_ticks=args.max_ticks,
        bankroll=args.bankroll,
        edge_min=args.edge_min,
        edge_buffer=args.edge_buffer,
        kelly_fraction=args.kelly_fraction,
        f_cap=args.f_cap,
        min_order_notional=args.min_order_notional,
        rv_fallback=args.rv_fallback,
        sigma_weight=args.sigma_weight,
        iv_override=args.iv,
        log_dir=args.log_dir,
        enable_websocket=not args.disable_websocket,
        fresh_start=args.fresh_start,
        polymarket_live_auth=live_auth,
    )
    configure_logging(config)
    try:
        validate_live_mode_credentials(config)
        asyncio.run(run_async(config))
    except RuntimeError as exc:
        LOGGER.error("%s", exc)
        return 1
    except KeyboardInterrupt:
        LOGGER.info("Interrupted by user.")
    return 0


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


def _limit_price(*, ask: float, fair_probability: float, edge_buffer: float) -> float | None:
    target = fair_probability - edge_buffer
    if target <= 0.0:
        return None
    price = min(ask, target)
    # Bound to valid probability price ticks.
    bounded = min(0.999, max(0.001, price))
    return round(bounded, 4)


def _edge_for_action(action: ExecutionAction, decision: DecisionOutcome | None) -> float:
    if decision is None or action.side is None:
        return 0.0
    if action.side == Side.BUY:
        return decision.up.edge
    return decision.down.edge


def _side_to_label(side: Side | None) -> str:
    if side is None:
        return "none"
    if side == Side.BUY:
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


def _reset_execution_csv(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=ExecutionLogRecord.csv_fields())
        writer.writeheader()


def _should_mark_dry_run_fill(*, limit_price: float, best_ask: float | None) -> bool:
    if best_ask is None:
        return False
    return (limit_price + 1e-9) >= best_ask


if __name__ == "__main__":
    raise SystemExit(main())
