from __future__ import annotations

import math
import sys
import unittest
from datetime import datetime, timezone
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from pm1h_edge_trader.main import (  # noqa: E402
    SECONDS_PER_YEAR,
    AdaptiveRealizedVolEstimator,
    DeribitVolatilityIndexEstimator,
    RealizedVolEstimator,
    _ewma_std,
    _interval_to_seconds,
    _log_returns,
    _sample_std,
    _tau_short_weight,
)


class _FakeHttpClient:
    def __init__(self, payloads: list[object]) -> None:
        self._payloads = list(payloads)
        self.calls: list[tuple[str, dict[str, object]]] = []

    async def get_json(self, url: str, *, params=None, headers=None, timeout_seconds=None):  # type: ignore[no-untyped-def]
        self.calls.append((url, dict(params or {})))
        if not self._payloads:
            return {}
        return self._payloads.pop(0)


class _StaticSigmaEstimator:
    def __init__(self, sigma: float) -> None:
        self._sigma = sigma

    async def get_sigma(self, *, now: datetime) -> float:
        return self._sigma


def _kline_rows_from_closes(closes: list[float]) -> list[list[object]]:
    rows: list[list[object]] = []
    for close in closes:
        rows.append([0, "0", "0", "0", str(close), "0"])
    return rows


class VolatilityPipelineTests(unittest.IsolatedAsyncioTestCase):
    def test_interval_to_seconds_parses_minute_and_hour_formats(self) -> None:
        self.assertEqual(_interval_to_seconds("5m"), 300)
        self.assertEqual(_interval_to_seconds("15m"), 900)
        self.assertEqual(_interval_to_seconds("1h"), 3600)
        self.assertEqual(_interval_to_seconds("2h"), 7200)

    def test_ewma_std_emphasizes_recent_shock(self) -> None:
        returns = [0.0, 0.0, 0.1]
        sample = _sample_std(returns)
        ewma = _ewma_std(returns, half_life=1.0)
        self.assertGreater(ewma, sample)

    async def test_realized_vol_estimator_uses_interval_aware_annualization(self) -> None:
        closes = [100.0, 101.0, 99.0, 100.0, 102.0]
        fake_http = _FakeHttpClient([_kline_rows_from_closes(closes)])
        estimator = RealizedVolEstimator(
            rest_base_url="https://api.binance.com",
            symbol="BTCUSDT",
            interval="5m",
            lookback_hours=12,
            refresh_seconds=1.0,
            fallback_sigma=0.55,
            floor_sigma=0.10,
            ewma_half_life=0.0,
            http_client=fake_http,
        )
        now = datetime(2026, 2, 23, 8, 0, tzinfo=timezone.utc)

        sigma = await estimator.get_sigma(now=now)

        expected_std = _sample_std(_log_returns(closes))
        expected = expected_std * math.sqrt(SECONDS_PER_YEAR / _interval_to_seconds("5m"))
        self.assertTrue(math.isclose(sigma, expected, rel_tol=1e-12, abs_tol=1e-12))

    async def test_adaptive_rv_blends_long_short_by_time_to_expiry(self) -> None:
        estimator = AdaptiveRealizedVolEstimator(
            long_horizon=_StaticSigmaEstimator(0.4),
            short_horizon=_StaticSigmaEstimator(0.8),
            tau_switch_seconds=1800.0,
            floor_sigma=0.1,
            fallback_sigma=0.55,
        )
        now = datetime(2026, 2, 23, 8, 0, tzinfo=timezone.utc)

        sigma = await estimator.get_sigma(now=now, seconds_to_expiry=900.0)

        expected = math.sqrt((0.5 * 0.4 * 0.4) + (0.5 * 0.8 * 0.8))
        self.assertTrue(math.isclose(sigma, expected, rel_tol=1e-12, abs_tol=1e-12))
        self.assertTrue(math.isclose(_tau_short_weight(0.0, 1800.0), 1.0, abs_tol=1e-12))
        self.assertTrue(math.isclose(_tau_short_weight(1800.0, 1800.0), 0.0, abs_tol=1e-12))

    async def test_deribit_iv_estimator_parses_latest_close(self) -> None:
        payload = {
            "result": {
                "data": [
                    [1771849620000, 55.0, 56.0, 54.0, 55.5],
                    [1771849680000, 55.5, 56.0, 55.2, 57.5],
                ],
                "continuation": None,
            }
        }
        fake_http = _FakeHttpClient([payload])
        estimator = DeribitVolatilityIndexEstimator(
            currency="BTC",
            resolution_minutes=60,
            lookback_hours=48,
            refresh_seconds=1.0,
            floor=0.05,
            cap=3.0,
            http_client=fake_http,
        )

        iv = await estimator.get_iv(now=datetime(2026, 2, 23, 8, 0, tzinfo=timezone.utc))

        self.assertTrue(math.isclose(iv or 0.0, 0.575, rel_tol=1e-12, abs_tol=1e-12))

    async def test_deribit_iv_estimator_returns_none_on_empty_data(self) -> None:
        fake_http = _FakeHttpClient([{"result": {"data": [], "continuation": None}}])
        estimator = DeribitVolatilityIndexEstimator(
            currency="BTC",
            resolution_minutes=60,
            lookback_hours=48,
            refresh_seconds=1.0,
            floor=0.05,
            cap=3.0,
            http_client=fake_http,
        )

        iv = await estimator.get_iv(now=datetime(2026, 2, 23, 8, 0, tzinfo=timezone.utc))

        self.assertIsNone(iv)


if __name__ == "__main__":
    unittest.main()
