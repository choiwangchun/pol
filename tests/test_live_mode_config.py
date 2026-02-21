from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from pm1h_edge_trader.config import (  # noqa: E402
    PolymarketLiveAuthConfig,
    RuntimeMode,
    build_config,
    load_polymarket_live_auth_from_env,
    validate_live_mode_credentials,
)


class LiveModeConfigTests(unittest.TestCase):
    def _build_config(self, *, mode: RuntimeMode, auth: PolymarketLiveAuthConfig):
        with TemporaryDirectory() as tmp_dir:
            return build_config(
                mode=mode,
                binance_symbol="BTCUSDT",
                market_slug=None,
                tick_seconds=1.0,
                max_ticks=1,
                bankroll=10_000.0,
                edge_min=0.015,
                edge_buffer=0.010,
                kelly_fraction=0.25,
                f_cap=0.05,
                min_order_notional=5.0,
                rv_fallback=0.55,
                sigma_weight=1.0,
                iv_override=None,
                log_dir=Path(tmp_dir),
                enable_websocket=False,
                polymarket_live_auth=auth,
            )

    def test_load_live_auth_defaults_signature_and_chain(self) -> None:
        env = {
            "POLYMARKET_PRIVATE_KEY": "0xabc",
            "POLYMARKET_FUNDER": "0xdef",
        }
        auth = load_polymarket_live_auth_from_env(env)
        self.assertEqual(auth.private_key, "0xabc")
        self.assertEqual(auth.funder, "0xdef")
        self.assertEqual(auth.signature_type, 1)
        self.assertEqual(auth.chain_id, 137)

    def test_load_live_auth_optional_api_creds(self) -> None:
        env = {
            "POLYMARKET_PRIVATE_KEY": "0xabc",
            "POLYMARKET_FUNDER": "0xdef",
            "POLYMARKET_API_KEY": "api-key",
            "POLYMARKET_API_SECRET": "api-secret",
            "POLYMARKET_API_PASSPHRASE": "api-pass",
            "POLYMARKET_SIGNATURE_TYPE": "2",
            "POLYMARKET_CHAIN_ID": "80002",
        }
        auth = load_polymarket_live_auth_from_env(env)
        self.assertEqual(auth.api_key, "api-key")
        self.assertEqual(auth.api_secret, "api-secret")
        self.assertEqual(auth.api_passphrase, "api-pass")
        self.assertEqual(auth.signature_type, 2)
        self.assertEqual(auth.chain_id, 80002)

    def test_validate_live_mode_fails_when_required_creds_missing(self) -> None:
        config = self._build_config(
            mode=RuntimeMode.LIVE,
            auth=PolymarketLiveAuthConfig(),
        )
        with self.assertRaises(RuntimeError) as context:
            validate_live_mode_credentials(config)
        message = str(context.exception)
        self.assertIn("POLYMARKET_PRIVATE_KEY", message)
        self.assertIn("POLYMARKET_FUNDER", message)

    def test_validate_live_mode_passes_when_required_creds_present(self) -> None:
        config = self._build_config(
            mode=RuntimeMode.LIVE,
            auth=PolymarketLiveAuthConfig(
                private_key=os.getenv("TEST_POLYMARKET_PRIVATE_KEY", "0xabc"),
                funder=os.getenv("TEST_POLYMARKET_FUNDER", "0xdef"),
            ),
        )
        validate_live_mode_credentials(config)

    def test_validate_ignored_for_dry_run(self) -> None:
        config = self._build_config(
            mode=RuntimeMode.DRY_RUN,
            auth=PolymarketLiveAuthConfig(),
        )
        validate_live_mode_credentials(config)


if __name__ == "__main__":
    unittest.main()
