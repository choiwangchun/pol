from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from pm1h_edge_trader.engine import (  # noqa: E402
    DecisionConfig,
    DecisionEngine,
    blended_sigma,
    compute_probabilities,
    evaluate_edges,
    fractional_kelly,
    kelly_order_size,
)


def _assert_close(test_case: unittest.TestCase, lhs: float, rhs: float, tol: float = 1e-12) -> None:
    test_case.assertTrue(math.isclose(lhs, rhs, rel_tol=tol, abs_tol=tol), f"{lhs} != {rhs}")


class MathEngineTests(unittest.TestCase):
    def test_compute_probabilities_nominal_case(self) -> None:
        q_up, q_down = compute_probabilities(spot=100.0, strike=100.0, sigma=0.2, tau=1.0)
        _assert_close(self, q_up, 0.460172162722971)
        _assert_close(self, q_down, 0.539827837277029)
        _assert_close(self, q_up + q_down, 1.0)

    def test_compute_probabilities_stable_for_tau_or_sigma_near_zero(self) -> None:
        self.assertEqual(compute_probabilities(spot=101.0, strike=100.0, sigma=0.2, tau=0.0), (1.0, 0.0))
        self.assertEqual(compute_probabilities(spot=99.0, strike=100.0, sigma=0.2, tau=0.0), (0.0, 1.0))
        self.assertEqual(compute_probabilities(spot=100.0, strike=100.0, sigma=0.0, tau=1.0), (1.0, 0.0))
        self.assertEqual(compute_probabilities(spot=99.0, strike=100.0, sigma=0.0, tau=1.0), (0.0, 1.0))

    def test_blended_sigma_with_iv_and_fallback(self) -> None:
        sigma = blended_sigma(rv=0.3, iv=0.5, w=0.25)
        _assert_close(self, sigma, math.sqrt(0.25 * 0.3**2 + 0.75 * 0.5**2))

        sigma_fallback = blended_sigma(rv=0.3, iv=None, w=0.25)
        _assert_close(self, sigma_fallback, 0.3)


    def test_evaluate_edges_with_buffers(self) -> None:
        edge = evaluate_edges(
            q_up=0.55,
            q_down=0.45,
            ask_up=0.50,
            ask_down=0.47,
            buffer_up=0.01,
            buffer_down=0.02,
        )
        _assert_close(self, edge.edge_up, 0.05)
        _assert_close(self, edge.edge_up_after_buffer, 0.04)
        _assert_close(self, edge.edge_down, -0.02)
        _assert_close(self, edge.edge_down_after_buffer, -0.04)

    def test_evaluate_edges_applies_cost_before_buffer(self) -> None:
        edge = evaluate_edges(
            q_up=0.60,
            q_down=0.40,
            ask_up=0.55,
            ask_down=0.35,
            cost_up=0.01,
            cost_down=0.02,
            buffer_up=0.005,
            buffer_down=0.005,
        )
        _assert_close(self, edge.edge_up, 0.04)
        _assert_close(self, edge.edge_up_after_buffer, 0.035)
        _assert_close(self, edge.edge_down, 0.03)
        _assert_close(self, edge.edge_down_after_buffer, 0.025)


    def test_fractional_kelly_and_cap(self) -> None:
        f = fractional_kelly(q=0.60, c=0.55, kelly_fraction=0.5, f_cap=0.2)
        _assert_close(self, f, 0.05555555555555556)

        self.assertEqual(fractional_kelly(q=0.55, c=0.55, kelly_fraction=1.0, f_cap=1.0), 0.0)
        _assert_close(self, fractional_kelly(q=0.95, c=0.20, kelly_fraction=1.0, f_cap=0.1), 0.1)


    def test_kelly_order_size_honors_min_order_size(self) -> None:
        small = kelly_order_size(
            bankroll=1_000.0,
            q=0.60,
            c=0.55,
            kelly_fraction=0.5,
            f_cap=1.0,
            min_order_size=100.0,
        )
        self.assertEqual(small, 0.0)

        large = kelly_order_size(
            bankroll=3_000.0,
            q=0.60,
            c=0.55,
            kelly_fraction=0.5,
            f_cap=1.0,
            min_order_size=100.0,
        )
        _assert_close(self, large, 166.66666666666669)


    def test_decision_engine_trade_and_no_trade_intents(self) -> None:
        engine = DecisionEngine(
            DecisionConfig(
                sigma_weight=0.5,
                edge_min=0.01,
                edge_buffer_up=0.005,
                edge_buffer_down=0.005,
                kelly_fraction=0.5,
                f_cap=0.2,
                min_order_size=10.0,
            )
        )

        outcome = engine.decide(
            spot=110.0,
            strike=100.0,
            tau=0.1,
            rv=0.2,
            iv=0.2,
            ask_up=0.50,
            ask_down=0.55,
            bankroll=1_000.0,
        )
        self.assertTrue(outcome.up.should_trade)
        _assert_close(self, outcome.up.order_size, 200.0)
        self.assertFalse(outcome.down.should_trade)
        self.assertEqual(outcome.down.reason, "edge_below_min")

        no_trade = engine.decide(
            spot=100.0,
            strike=100.0,
            tau=1.0,
            rv=0.2,
            iv=0.2,
            ask_up=0.46,
            ask_down=0.54,
            bankroll=1_000.0,
        )
        self.assertFalse(no_trade.up.should_trade)
        self.assertFalse(no_trade.down.should_trade)


if __name__ == "__main__":
    unittest.main()
