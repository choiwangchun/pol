from __future__ import annotations

import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from pm1h_edge_trader.api_credentials import (  # noqa: E402
    _upsert_env_vars,
    derive_api_credentials,
)


class _FakeCreds:
    def __init__(self) -> None:
        self.api_key = "k-1"
        self.api_secret = "s-1"
        self.api_passphrase = "p-1"


class _FakeClient:
    def __init__(self, **kwargs) -> None:  # type: ignore[no-untyped-def]
        self.kwargs = kwargs

    def create_or_derive_api_creds(self) -> object:
        return _FakeCreds()


class _Auth:
    chain_id = 137
    private_key = "0xabc"
    signature_type = 1
    funder = "0xfunder"


class ApiCredentialsTests(unittest.TestCase):
    def test_derive_api_credentials_returns_values(self) -> None:
        creds = derive_api_credentials(
            host="https://clob.polymarket.com",
            auth=_Auth(),
            client_factory=_FakeClient,
        )
        self.assertEqual(creds.api_key, "k-1")
        self.assertEqual(creds.api_secret, "s-1")
        self.assertEqual(creds.api_passphrase, "p-1")

    def test_upsert_env_vars_updates_or_appends(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            env_path = Path(tmp_dir) / ".env"
            env_path.write_text(
                "POLYMARKET_API_KEY=old-k\nEXISTING=1\n",
                encoding="utf-8",
            )
            _upsert_env_vars(
                path=env_path,
                values={
                    "POLYMARKET_API_KEY": "new-k",
                    "POLYMARKET_API_SECRET": "new-s",
                    "POLYMARKET_API_PASSPHRASE": "new-p",
                },
            )
            text = env_path.read_text(encoding="utf-8")
            self.assertIn("POLYMARKET_API_KEY=new-k", text)
            self.assertIn("POLYMARKET_API_SECRET=new-s", text)
            self.assertIn("POLYMARKET_API_PASSPHRASE=new-p", text)
            self.assertIn("EXISTING=1", text)


if __name__ == "__main__":
    unittest.main()
