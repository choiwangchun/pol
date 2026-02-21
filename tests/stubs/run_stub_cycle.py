from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from pm1h_edge_trader.execution import (
    DryRunExecutionAdapter,
    ExecutionAction,
    ExecutionConfig,
    IntentSignal,
    LimitOrderExecutionEngine,
    SafetySurface,
    Side,
)


def _parse_bool(value: str | None, *, default: bool) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _parse_side(value: str) -> Side:
    normalized = value.strip().lower()
    if normalized == "buy":
        return Side.BUY
    if normalized == "sell":
        return Side.SELL
    raise ValueError(f"unsupported side: {value}")


def _signal_from_env(env: dict[str, str], now: datetime) -> IntentSignal:
    return IntentSignal(
        market_id=env.get("MARKET_ID", "demo-btc-1h"),
        token_id=env.get("TOKEN_ID", "demo-token-up"),
        side=_parse_side(env.get("SIDE", "buy")),
        edge=float(env.get("EDGE", "0.03")),
        min_edge=float(env.get("MIN_EDGE", "0.01")),
        desired_price=float(env.get("DESIRED_PRICE", "0.55")),
        size=float(env.get("ORDER_SIZE", "5")),
        seconds_to_expiry=float(env.get("SECONDS_TO_EXPIRY", "600")),
        signal_ts=now,
        allow_entry=_parse_bool(env.get("ALLOW_ENTRY"), default=True),
    )


def _safety_from_env(env: dict[str, str]) -> SafetySurface:
    return SafetySurface(
        fee_rate=float(env.get("FEE_RATE", "0.0")),
        data_age_s=float(env.get("DATA_AGE_S", "1")),
        clock_drift_s=float(env.get("CLOCK_DRIFT_S", "0")),
        book_is_consistent=_parse_bool(env.get("BOOK_IS_CONSISTENT"), default=True),
    )


def _execution_config_from_env(env: dict[str, str]) -> ExecutionConfig:
    return ExecutionConfig(
        max_data_age_s=float(env.get("MAX_DATA_AGE_S", "3")),
        max_clock_drift_s=float(env.get("MAX_CLOCK_DRIFT_S", "1")),
        fee_must_be_zero=_parse_bool(env.get("FEE_MUST_BE_ZERO"), default=True),
        require_book_match=_parse_bool(env.get("REQUIRE_BOOK_MATCH"), default=True),
        entry_block_window_s=float(env.get("ENTRY_BLOCK_WINDOW_S", "60")),
    )


def _serialize_action(action: ExecutionAction) -> str:
    return (
        f"action={action.action_type.value}"
        f" market_id={action.market_id}"
        f" side={(action.side.value if action.side else '-')}"
        f" order_id={action.order_id or '-'}"
        f" reason={action.reason or '-'}"
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run one stub trading cycle for local MVP validation."
    )
    parser.add_argument("--edge", type=float, help="Override EDGE.")
    parser.add_argument("--fee", type=float, help="Override FEE_RATE.")
    parser.add_argument("--data-age-s", type=float, help="Override DATA_AGE_S.")
    parser.add_argument("--allow-entry", choices=["true", "false"], help="Override ALLOW_ENTRY.")
    parser.add_argument(
        "--log-file",
        default=os.getenv("LOG_FILE"),
        help="Optional output file for structured logs.",
    )
    return parser


def main() -> int:
    args = _build_parser().parse_args()
    env = dict(os.environ)

    if args.edge is not None:
        env["EDGE"] = str(args.edge)
    if args.fee is not None:
        env["FEE_RATE"] = str(args.fee)
    if args.data_age_s is not None:
        env["DATA_AGE_S"] = str(args.data_age_s)
    if args.allow_entry is not None:
        env["ALLOW_ENTRY"] = args.allow_entry

    now = datetime.now(tz=timezone.utc)
    adapter = DryRunExecutionAdapter(id_prefix="mvp", now_fn=lambda: now)
    engine = LimitOrderExecutionEngine(
        adapter=adapter,
        config=_execution_config_from_env(env),
        now_fn=lambda: now,
    )
    signal = _signal_from_env(env, now=now)
    safety = _safety_from_env(env)
    result = engine.process_signal(signal, safety, now=now)

    summary = f"kill_switch_active={result.kill_switch_active} actions={len(result.actions)}"
    lines = [summary]
    lines.extend(_serialize_action(action) for action in result.actions)
    for line in lines:
        print(line)

    if args.log_file:
        log_dir = os.path.dirname(args.log_file)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
        with open(args.log_file, "a", encoding="utf-8") as log_file:
            for line in lines:
                log_file.write(line + "\n")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
