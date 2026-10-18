import unittest

import pandas as pd

from streamlit_app.cnr_hedge_transport.contract import simulate_contract


class ContractScenarioTests(unittest.TestCase):
    def setUp(self):
        months = pd.period_range("2025-01", periods=13, freq="M")
        self.levels = pd.DataFrame(
            {"cnr": [100] + [110] * 11 + [120],
             "brent": [50] + [51] * 11 + [52]},
            index=months,
        )
        self.start = pd.Period("2025-02", freq="M")

    def test_monthly_clause_and_cash_timing(self):
        result = simulate_contract(
            self.levels, self.start, 12000, 0.2, 0.5, "monthly", "monthly"
        )
        self.assertAlmostEqual(result.annual_invoice_eur, 12260)
        self.assertAlmostEqual(result.surcharge_eur, 260)
        self.assertEqual(result.first_contracts, 0)
        self.assertEqual(result.ledger.mini_contracts.iloc[-1], 0)
        self.assertAlmostEqual(result.ledger.cash_invoice_eur.iloc[0], 1020)
        self.assertAlmostEqual(result.ledger.cash_invoice_eur.sum(), 12260)
        self.assertAlmostEqual(result.ledger.invoice_gap_eur.iloc[0], 20)
        self.assertAlmostEqual(
            result.ledger.total_pnl_eur.iloc[0],
            result.ledger.proxy_pnl_eur.iloc[0] - 20,
        )
        self.assertAlmostEqual(result.ledger.total_pnl_eur.sum(), -result.residual_before_costs_eur)

    def test_annual_anniversary_and_average_are_distinct(self):
        anniversary = simulate_contract(
            self.levels, self.start, 12000, 0.2, 0.5,
            "monthly", "annual", annual_reference="anniversary"
        )
        average = simulate_contract(
            self.levels, self.start, 12000, 0.2, 0.5,
            "annual", "annual", annual_reference="average"
        )
        self.assertAlmostEqual(anniversary.surcharge_eur, 480)
        self.assertAlmostEqual(average.surcharge_eur, 260)
        self.assertAlmostEqual(anniversary.ledger.cash_invoice_eur.iloc[0], 1000)
        self.assertAlmostEqual(anniversary.ledger.cash_invoice_eur.iloc[-1], 1480)
        self.assertTrue((average.ledger.cash_invoice_eur.iloc[:-1] == 0).all())
        self.assertAlmostEqual(average.ledger.cash_invoice_eur.iloc[-1], 12260)
        self.assertAlmostEqual(average.ledger.invoice_gap_eur.sum(), average.surcharge_eur)
        self.assertAlmostEqual(
            average.ledger.total_pnl_eur.sum(), -average.residual_before_costs_eur
        )

    def test_position_reduces_as_months_fix(self):
        result = simulate_contract(
            self.levels, self.start, 120000, 0.2, 0.5, "monthly", "monthly"
        )
        self.assertEqual(result.first_contracts, 2)
        self.assertEqual(result.ledger.mini_contracts.iloc[-1], 0)
        self.assertTrue(result.ledger.mini_contracts.is_monotonic_decreasing)

    def test_small_budget_cannot_be_hedged_with_floor_rounding(self):
        result = simulate_contract(
            self.levels, self.start, 1200, 0.2, 0.5, "annual", "monthly"
        )
        self.assertEqual(result.first_contracts, 0)
        self.assertEqual(result.proxy_pnl_eur, 0)

    def test_continuous_position_overhedges_small_remaining_exposure(self):
        result = simulate_contract(
            self.levels, self.start, 12000, 0.2, 0.5,
            "monthly", "monthly", rounding="continuous",
        )
        self.assertTrue((result.ledger.mini_contracts == 1).all())
        self.assertGreater(
            result.ledger.mini_contracts.iloc[-1] / result.ledger.target_contracts.iloc[-1],
            1,
        )

    def test_missing_month_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "Treize mois"):
            simulate_contract(
                self.levels.drop(pd.Period("2025-05", freq="M")),
                self.start, 12000, 0.2, 0.5, "monthly", "monthly",
            )


if __name__ == "__main__":
    unittest.main()
