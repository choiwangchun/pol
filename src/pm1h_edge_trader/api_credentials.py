"""Generate or derive Polymarket API credentials from wallet auth."""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

from .config import (
    AppConfig,
    RuntimeMode,
    load_polymarket_live_auth_from_env,
    validate_live_mode_credentials,
)

try:
    from py_clob_client.client import ClobClient
except Exception:  # pragma: no cover - optional runtime dependency during import
    ClobClient = None  # type: ignore[assignment]


@dataclass(frozen=True)
class DerivedApiCreds:
    api_key: str
    api_secret: str
    api_passphrase: str


def derive_api_credentials(
    *,
    host: str,
    auth,
    client_factory: Callable[..., object] | None = None,
) -> DerivedApiCreds:
    factory = client_factory
    if factory is None:
        if ClobClient is None:
            raise RuntimeError("py-clob-client is required to derive API credentials.")
        factory = ClobClient

    client = factory(
        host=host,
        chain_id=auth.chain_id,
        key=auth.private_key,
        signature_type=auth.signature_type,
        funder=auth.funder,
    )
    creds = client.create_or_derive_api_creds()
    if creds is None:
        raise RuntimeError("create_or_derive_api_creds returned empty credentials.")
    api_key = str(getattr(creds, "api_key", "") or "").strip()
    api_secret = str(getattr(creds, "api_secret", "") or "").strip()
    api_passphrase = str(getattr(creds, "api_passphrase", "") or "").strip()
    if not api_key or not api_secret or not api_passphrase:
        raise RuntimeError("Derived API credentials are incomplete.")
    return DerivedApiCreds(
        api_key=api_key,
        api_secret=api_secret,
        api_passphrase=api_passphrase,
    )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate/derive Polymarket API credentials")
    parser.add_argument("--host", default="https://clob.polymarket.com", help="Polymarket CLOB host URL.")
    parser.add_argument(
        "--format",
        choices=["env", "json"],
        default="env",
        help="Output format.",
    )
    parser.add_argument(
        "--write-env",
        action="store_true",
        help="Write derived credentials into .env (or --env-file path).",
    )
    parser.add_argument(
        "--env-file",
        type=Path,
        default=Path(".env"),
        help="Path for --write-env.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    auth = load_polymarket_live_auth_from_env(os.environ)
    try:
        validate_live_mode_credentials(AppConfig(mode=RuntimeMode.LIVE, polymarket_live_auth=auth))
        creds = derive_api_credentials(host=args.host, auth=auth)
        if args.write_env:
            _upsert_env_vars(
                path=args.env_file,
                values={
                    "POLYMARKET_API_KEY": creds.api_key,
                    "POLYMARKET_API_SECRET": creds.api_secret,
                    "POLYMARKET_API_PASSPHRASE": creds.api_passphrase,
                },
            )
    except RuntimeError as exc:
        if args.format == "json":
            print(json.dumps({"ok": False, "error": str(exc)}, ensure_ascii=True))
        else:
            print(f"ERROR: {exc}")
        return 1
    except Exception as exc:  # pragma: no cover
        if args.format == "json":
            print(json.dumps({"ok": False, "error": f"{exc.__class__.__name__}: {exc}"}, ensure_ascii=True))
        else:
            print(f"ERROR: {exc.__class__.__name__}: {exc}")
        return 1

    if args.format == "json":
        print(
            json.dumps(
                {
                    "ok": True,
                    "result": {
                        "POLYMARKET_API_KEY": creds.api_key,
                        "POLYMARKET_API_SECRET": creds.api_secret,
                        "POLYMARKET_API_PASSPHRASE": creds.api_passphrase,
                        "env_file_updated": bool(args.write_env),
                        "env_file_path": str(args.env_file),
                    },
                },
                ensure_ascii=True,
            )
        )
        return 0

    print(f"POLYMARKET_API_KEY={creds.api_key}")
    print(f"POLYMARKET_API_SECRET={creds.api_secret}")
    print(f"POLYMARKET_API_PASSPHRASE={creds.api_passphrase}")
    if args.write_env:
        print(f"Updated {args.env_file}")
    return 0


def _upsert_env_vars(*, path: Path, values: dict[str, str]) -> None:
    existing_lines: list[str] = []
    if path.exists():
        existing_lines = path.read_text(encoding="utf-8").splitlines()

    remaining = dict(values)
    output: list[str] = []
    for line in existing_lines:
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in line:
            output.append(line)
            continue
        key, _ = line.split("=", 1)
        normalized = key.strip()
        if normalized in remaining:
            output.append(f"{normalized}={remaining.pop(normalized)}")
        else:
            output.append(line)

    for key in ("POLYMARKET_API_KEY", "POLYMARKET_API_SECRET", "POLYMARKET_API_PASSPHRASE"):
        if key in remaining:
            output.append(f"{key}={remaining[key]}")

    text = "\n".join(output).rstrip("\n") + "\n"
    path.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
