"""Live balance snapshot helpers for bankroll sizing."""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from typing import Callable, Sequence

from .config import PolymarketLiveAuthConfig, load_polymarket_live_auth_from_env, validate_live_mode_credentials
from .execution.polymarket_live_adapter import PolymarketLiveExecutionAdapter


@dataclass(frozen=True)
class LiveBalanceSnapshot:
    balance: float | None
    allowance: float | None
    bankroll: float


def fetch_live_balance_snapshot(
    *,
    auth: PolymarketLiveAuthConfig,
    host: str,
    adapter_factory: Callable[..., object] = PolymarketLiveExecutionAdapter,
) -> LiveBalanceSnapshot:
    adapter = adapter_factory(host=host, auth=auth)
    balance, allowance = adapter.get_collateral_balance_allowance(refresh=True)
    candidates = [value for value in (balance, allowance) if value is not None and value > 0.0]
    if not candidates:
        raise RuntimeError("Live collateral balance/allowance is unavailable or zero.")
    bankroll = min(candidates)
    return LiveBalanceSnapshot(balance=balance, allowance=allowance, bankroll=bankroll)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Fetch live collateral balance snapshot")
    parser.add_argument("--host", default="https://clob.polymarket.com", help="Polymarket CLOB host URL.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    auth = load_polymarket_live_auth_from_env(os.environ)
    try:
        from .config import AppConfig, RuntimeMode

        validate_live_mode_credentials(AppConfig(mode=RuntimeMode.LIVE, polymarket_live_auth=auth))
        snapshot = fetch_live_balance_snapshot(auth=auth, host=args.host)
    except RuntimeError as exc:
        print(json.dumps({"ok": False, "error": str(exc)}, ensure_ascii=True))
        return 1
    except Exception as exc:  # pragma: no cover
        print(json.dumps({"ok": False, "error": f"{exc.__class__.__name__}: {exc}"}, ensure_ascii=True))
        return 1

    print(
        json.dumps(
            {
                "ok": True,
                "result": {
                    "balance": snapshot.balance,
                    "allowance": snapshot.allowance,
                    "bankroll": snapshot.bankroll,
                },
            },
            ensure_ascii=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
