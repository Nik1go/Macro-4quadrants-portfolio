"""Vérifications des flux de factures et du suivi d'une échéance ICE fixe."""

import unittest

import pandas as pd

from streamlit_app.cnr_hedge_transport.backtest import IceHistory, _ice_expiry, run_strip_backtest


class StripBacktestTests(unittest.TestCase):
    def setUp(self):
        days = pd.to_datetime(["2025-01-03", "2025-01-31", "2025-02-03", "2025-02-28"])
        prices = pd.DataFrame(1.0, index=days, columns=range(1, 13))
        prices.loc[:, 1] = [80.0, 85.0, 96.0, 98.0]
        prices.loc[:, 2] = [90.0, 94.0, 101.0, 103.0]
        contracts = pd.DataFrame(
            {
                "expiry": pd.to_datetime(["2025-01-31", "2025-02-28"]),
                "ticker": ["COH25 Comdty", "COJ25 Comdty"],
            },
            index=pd.PeriodIndex(["2025-03", "2025-04"], freq="M"),
        )
        self.history = IceHistory(prices, prices.copy(), prices.copy(), contracts)
        self.cnr = pd.Series(
            [100.0, 120.0],
            index=pd.PeriodIndex(["2024-12", "2025-02"], freq="M"),
        )
        self.positions = pd.DataFrame(
            {"month": ["2025-02"], "invoice_eur": [1_000.0], "barrels": [100.0]}
        )

    def test_follows_same_contract_across_co_rank_change(self):
        result = run_strip_backtest(
            self.history,
            self.cnr,
            entry_date=pd.Timestamp("2025-01-03"),
            base_cnr_month="2024-12",
            diesel_weight=0.2,
            positions=self.positions,
            eurusd=1.0,
        )
        row = result.monthly.iloc[0]
        self.assertEqual(row["ticker"], "COJ25 Comdty")
        self.assertEqual((row["rank_entry"], row["rank_exit"]), (2, 1))
        self.assertEqual(result.daily["price_usd_bbl"].tolist(), [90.0, 94.0, 96.0, 98.0])
        self.assertAlmostEqual(row["pnl_usd"], 800.0)
        self.assertAlmostEqual(row["surcharge_eur"], 40.0)
        self.assertAlmostEqual(row["residual_eur"], -760.0)
        self.assertAlmostEqual(row["ice_contracts_fractional"], 0.1)
        self.assertAlmostEqual(row["mini_equivalents"], 1.0)
        self.assertAlmostEqual(row["front_move_usd"], 1_800.0)
        self.assertAlmostEqual(row["curve_effect_usd"], -1_000.0)
        self.assertAlmostEqual(row["front_move_usd"] + row["curve_effect_usd"], row["pnl_usd"])
        self.assertEqual(row["curve_shape"], "contango")

    def test_daily_fx_is_applied_to_variation_margin(self):
        eurusd = pd.Series(
            [2.0, 1.0, 2.0],
            index=pd.to_datetime(["2025-01-03", "2025-02-03", "2025-02-28"]),
        )
        result = run_strip_backtest(
            self.history,
            self.cnr,
            entry_date=pd.Timestamp("2025-01-03"),
            base_cnr_month="2024-12",
            diesel_weight=0.2,
            positions=self.positions,
            eurusd=eurusd,
        )
        # 400 $ au taux 2, 200 $ au taux 1, 200 $ au taux 2.
        self.assertAlmostEqual(result.monthly.iloc[0]["pnl_eur"], 500.0)
        self.assertAlmostEqual(result.monthly.iloc[0]["residual_eur"], -460.0)

    def test_zero_contract_leaves_cnr_bill_exposed(self):
        positions = self.positions.copy()
        positions.loc[0, "barrels"] = 0
        result = run_strip_backtest(
            self.history,
            self.cnr,
            entry_date=pd.Timestamp("2025-01-03"),
            base_cnr_month="2024-12",
            diesel_weight=0.2,
            positions=positions,
            eurusd=1.1,
        )
        self.assertEqual(result.monthly.iloc[0]["pnl_eur"], 0)
        self.assertAlmostEqual(result.monthly.iloc[0]["residual_eur"], 40.0)

    def test_fractional_ice_position_scales_hedge_without_changing_invoice(self):
        positions = self.positions.copy()
        positions.loc[0, "barrels"] = 400.0  # 0,4 future ICE théorique.
        result = run_strip_backtest(
            self.history,
            self.cnr,
            entry_date=pd.Timestamp("2025-01-03"),
            base_cnr_month="2024-12",
            diesel_weight=0.2,
            positions=positions,
            eurusd=1.0,
        )
        row = result.monthly.iloc[0]
        self.assertAlmostEqual(row["ice_contracts_fractional"], 0.4)
        self.assertAlmostEqual(row["pnl_usd"], 3_200.0)
        self.assertAlmostEqual(row["surcharge_eur"], 40.0)
        self.assertAlmostEqual(row["residual_eur"], -3_160.0)

    def test_cnr_invoice_lag_changes_only_invoice_reference(self):
        cnr = pd.concat(
            [self.cnr, pd.Series([110.0], index=pd.PeriodIndex(["2025-01"], freq="M"))]
        )
        result = run_strip_backtest(
            self.history,
            cnr,
            entry_date=pd.Timestamp("2025-01-03"),
            base_cnr_month="2024-12",
            diesel_weight=0.2,
            positions=self.positions,
            eurusd=1.0,
            cnr_lag_months=1,
        )
        self.assertEqual(result.monthly.iloc[0]["cnr_month"], "2025-01")
        self.assertAlmostEqual(result.monthly.iloc[0]["surcharge_eur"], 20.0)
        self.assertAlmostEqual(result.monthly.iloc[0]["pnl_usd"], 800.0)

    def test_ice_expiry_new_year_exception(self):
        self.assertEqual(_ice_expiry(pd.Timestamp("2024-12-31")), pd.Timestamp("2024-12-30"))
        self.assertEqual(_ice_expiry(pd.Timestamp("2022-12-30")), pd.Timestamp("2022-12-29"))
        self.assertEqual(_ice_expiry(pd.Timestamp("2023-12-29")), pd.Timestamp("2023-12-28"))
        self.assertEqual(_ice_expiry(pd.Timestamp("2025-10-31")), pd.Timestamp("2025-10-31"))


if __name__ == "__main__":
    unittest.main()
