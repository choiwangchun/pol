"""Manual one-shot position entry CLI for dry-run/live workflows."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

from .config import (
    AppConfig,
    RuntimeMode,
    load_polymarket_live_auth_from_env,
    validate_live_mode_credentials,
)
from .execution import OrderRequest, Side
from .execution.polymarket_live_adapter import PolymarketLiveExecutionAdapter
from .feeds import BinanceUnderlyingAdapter, ClobOrderbookAdapter
from .logger import CSVExecutionReporter, ExecutionLogRecord
from .markets import GammaMarketDiscoveryClient
from .paper import PaperResultTracker


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Manual one-shot position entry")
    parser.add_argument(
        "--mode",
        choices=[mode.value for mode in RuntimeMode],
        default=RuntimeMode.DRY_RUN.value,
        help="Execution mode: dry-run or live.",
    )
    parser.add_argument(
        "--direction",
        choices=["up", "down"],
        required=True,
        help="Target position token direction.",
    )
    parser.add_argument(
        "--usd",
        type=float,
        required=True,
        help="Target notional in USD.",
    )
    parser.add_argument(
        "--price",
        type=float,
        default=None,
        help="Optional limit price. If omitted, best ask is used.",
    )
    parser.add_argument(
        "--market-slug",
        default=None,
        help="Optional exact Polymarket market slug override.",
    )
    parser.add_argument(
        "--binance-symbol",
        default="BTCUSDT",
        help="Underlying symbol for snapshot logging.",
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
        help="Directory for execution/result logs.",
    )
    return parser


async def run_once(args: argparse.Namespace) -> dict[str, object]:
    mode = RuntimeMode(args.mode)
    direction = str(args.direction).strip().lower()
    target_notional = max(0.0, float(args.usd))
    if target_notional <= 0.0:
        raise RuntimeError("--usd must be positive.")

    auth = load_polymarket_live_auth_from_env(os.environ)
    if mode == RuntimeMode.LIVE:
        validate_live_mode_credentials(
            AppConfig(
                mode=RuntimeMode.LIVE,
                polymarket_live_auth=auth,
            )
        )

    discovery = GammaMarketDiscoveryClient()
    market = await discovery.find_active_btc_1h_up_down_market(
        require_rule_match=True,
        preferred_slug=args.market_slug,
    )
    if market is None:
        raise RuntimeError("No active BTC 1H Up/Down market found.")

    orderbook = ClobOrderbookAdapter(
        token_ids=[market.token_ids.up_token_id, market.token_ids.down_token_id],
        enable_websocket=not args.disable_websocket,
    )
    underlying = BinanceUnderlyingAdapter(
        symbol=args.binance_symbol,
        interval="1h",
        enable_websocket=not args.disable_websocket,
    )

    try:
        await asyncio.gather(orderbook.start(), underlying.start())
        await asyncio.gather(
            orderbook.wait_until_ready(timeout_seconds=10.0),
            underlying.wait_until_ready(timeout_seconds=10.0),
        )
        book = await orderbook.get_snapshot()
        under = await underlying.get_snapshot()
    finally:
        await asyncio.gather(orderbook.stop(), underlying.stop(), return_exceptions=True)

    if direction == "up":
        token_id = market.token_ids.up_token_id
        side_label = "up"
    else:
        token_id = market.token_ids.down_token_id
        side_label = "down"
    quote = book.quotes.get(token_id)
    if quote is None or quote.best_ask is None or quote.best_ask <= 0.0:
        raise RuntimeError("Orderbook best ask is unavailable for selected direction.")

    price = float(args.price) if args.price is not None else float(quote.best_ask)
    if price <= 0.0 or price > 1.0:
        raise RuntimeError("Invalid limit price. Must be in (0, 1].")
    size = target_notional / price
    if size <= 0.0:
        raise RuntimeError("Computed order size is non-positive.")

    now = utc_now()
    reporter = CSVExecutionReporter(args.log_dir / "executions.csv")
    k_value = float(under.candle_open or 0.0)
    s_value = float(under.last_price or 0.0)
    q_up = 0.5
    bid_up = 0.0
    ask_up = 0.0
    bid_down = 0.0
    ask_down = 0.0
    up_quote = book.quotes.get(market.token_ids.up_token_id)
    down_quote = book.quotes.get(market.token_ids.down_token_id)
    if up_quote is not None:
        bid_up = float(up_quote.best_bid or 0.0)
        ask_up = float(up_quote.best_ask or 0.0)
    if down_quote is not None:
        bid_down = float(down_quote.best_bid or 0.0)
        ask_down = float(down_quote.best_ask or 0.0)

    if mode == RuntimeMode.DRY_RUN:
        order_id = f"manual-{int(now.timestamp() * 1000)}"
        record = ExecutionLogRecord(
            timestamp=now,
            market_id=market.market_id,
            K=k_value,
            S_t=s_value,
            tau=max((market.timing.end - now).total_seconds(), 0.0) / (365.0 * 24.0 * 3600.0),
            sigma=0.0,
            q_up=q_up,
            bid_up=bid_up,
            ask_up=ask_up,
            bid_down=bid_down,
            ask_down=ask_down,
            edge=0.0,
            order_id=order_id,
            side=side_label,
            price=price,
            size=size,
            status="paper_fill",
            settlement_outcome=None,
            pnl=None,
        )
        reporter.append(record)
        tracker = PaperResultTracker(
            initial_bankroll=_resolve_initial_bankroll(args.log_dir / "result.json", default_bankroll=1000.0),
            result_json_path=args.log_dir / "result.json",
        )
        tracker.bootstrap_from_csv(args.log_dir / "executions.csv")
        tracker.on_record(record)
        return {
            "mode": mode.value,
            "market_id": market.market_id,
            "order_id": order_id,
            "direction": direction,
            "price": round(price, 6),
            "size": round(size, 8),
            "status": "paper_fill",
        }

    adapter = PolymarketLiveExecutionAdapter(
        host="https://clob.polymarket.com",
        auth=auth,
    )
    request = OrderRequest(
        market_id=market.market_id,
        token_id=token_id,
        side=Side.BUY,
        price=price,
        size=size,
        submitted_at=now,
    )
    handle = adapter.place_limit_order(request)
    live_record = ExecutionLogRecord(
        timestamp=now,
        market_id=market.market_id,
        K=k_value,
        S_t=s_value,
        tau=max((market.timing.end - now).total_seconds(), 0.0) / (365.0 * 24.0 * 3600.0),
        sigma=0.0,
        q_up=q_up,
        bid_up=bid_up,
        ask_up=ask_up,
        bid_down=bid_down,
        ask_down=ask_down,
        edge=0.0,
        order_id=handle.order_id,
        side=side_label,
        price=price,
        size=size,
        status="place",
        settlement_outcome=None,
        pnl=None,
    )
    reporter.append(live_record)
    return {
        "mode": mode.value,
        "market_id": market.market_id,
        "order_id": handle.order_id,
        "direction": direction,
        "price": round(price, 6),
        "size": round(size, 8),
        "status": "place",
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    try:
        result = asyncio.run(run_once(args))
    except RuntimeError as exc:
        print(json.dumps({"ok": False, "error": str(exc)}, ensure_ascii=True))
        return 1
    except Exception as exc:  # pragma: no cover - defensive for CLI
        print(json.dumps({"ok": False, "error": f"{exc.__class__.__name__}: {exc}"}, ensure_ascii=True))
        return 1
    print(json.dumps({"ok": True, "result": result}, ensure_ascii=True))
    return 0


def _resolve_initial_bankroll(result_json_path: Path, *, default_bankroll: float) -> float:
    if not result_json_path.exists():
        return default_bankroll
    try:
        payload = json.loads(result_json_path.read_text(encoding="utf-8"))
    except Exception:
        return default_bankroll
    value = payload.get("initial_bankroll")
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default_bankroll
    return parsed if parsed > 0.0 else default_bankroll


if __name__ == "__main__":
    raise SystemExit(main())
