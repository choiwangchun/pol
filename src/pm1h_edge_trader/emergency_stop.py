"""Emergency stop helper for live operations.

This tool cancels all venue open orders and verifies residual open orders.
"""

from __future__ import annotations

import argparse
import json
import os
from collections.abc import Callable
from typing import Sequence

from .config import (
    AppConfig,
    RuntimeMode,
    PolymarketLiveAuthConfig,
    load_polymarket_live_auth_from_env,
    validate_live_mode_credentials,
)
from .execution.polymarket_live_adapter import PolymarketLiveExecutionAdapter


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Emergency stop for PM-1H live bot")
    parser.add_argument(
        "--host",
        default="https://clob.polymarket.com",
        help="Polymarket CLOB host URL.",
    )
    parser.add_argument(
        "--retries",
        type=int,
        default=3,
        help="Retry count for open-order listing checks.",
    )
    return parser


def _list_open_order_ids_with_retry(*, adapter: object, retries: int) -> set[str]:
    list_open_order_ids = getattr(adapter, "list_open_order_ids", None)
    if not callable(list_open_order_ids):
        raise RuntimeError("Live adapter does not support list_open_order_ids().")

    attempts = max(1, int(retries))
    last_error: Exception | None = None
    for _ in range(attempts):
        try:
            value = list_open_order_ids()
        except Exception as exc:
            last_error = exc
            continue
        if isinstance(value, set):
            return {str(item).strip() for item in value if str(item).strip()}
        if isinstance(value, (list, tuple)):
            return {str(item).strip() for item in value if str(item).strip()}
        return set()
    if last_error is not None:
        raise RuntimeError(f"open_order_list_failed: {last_error.__class__.__name__}: {last_error}") from last_error
    return set()


def perform_emergency_stop(
    *,
    auth: PolymarketLiveAuthConfig,
    host: str,
    retries: int = 3,
    adapter_factory: Callable[..., object] = PolymarketLiveExecutionAdapter,
) -> dict[str, object]:
    adapter = adapter_factory(host=host, auth=auth)
    open_before_ids = _list_open_order_ids_with_retry(adapter=adapter, retries=retries)

    cancel_all_orders = getattr(adapter, "cancel_all_orders", None)
    if not callable(cancel_all_orders):
        raise RuntimeError("Live adapter does not support cancel_all_orders().")
    canceled_count = int(cancel_all_orders())

    open_after_ids = _list_open_order_ids_with_retry(adapter=adapter, retries=retries)
    residual_order_ids = sorted(open_after_ids)
    ok = len(residual_order_ids) == 0
    return {
        "ok": ok,
        "host": host,
        "open_orders_before": len(open_before_ids),
        "open_orders_after": len(open_after_ids),
        "cancel_all_reported": canceled_count,
        "residual_order_ids": residual_order_ids,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    auth = load_polymarket_live_auth_from_env(os.environ)
    try:
        validate_live_mode_credentials(
            AppConfig(mode=RuntimeMode.LIVE, polymarket_live_auth=auth)
        )
        result = perform_emergency_stop(
            auth=auth,
            host=args.host,
            retries=args.retries,
        )
    except RuntimeError as exc:
        print(json.dumps({"ok": False, "error": str(exc)}, ensure_ascii=True))
        return 1
    except Exception as exc:  # pragma: no cover
        print(json.dumps({"ok": False, "error": f"{exc.__class__.__name__}: {exc}"}, ensure_ascii=True))
        return 1

    print(json.dumps(result, ensure_ascii=True))
    return 0 if bool(result.get("ok")) else 2


if __name__ == "__main__":
    raise SystemExit(main())
