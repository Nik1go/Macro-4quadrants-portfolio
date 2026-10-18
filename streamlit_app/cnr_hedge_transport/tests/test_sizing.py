"""Vérifie l'estimation avant achat et sa conversion en barils."""

import unittest
from datetime import date

import pandas as pd

from streamlit_app.cnr_hedge_transport.backtest import IceHistory
from streamlit_app.cnr_hedge_transport.sizing import estimate_pre_entry_beta, recommend_strip_positions


class SizingTests(unittest.TestCase):
    def test_beta_uses_only_months_before_entry(self):
        months = pd.period_range("2020-01", periods=42, freq="M")
        brent = [100.0]
        cnr = [100.0]
        for i in range(1, len(months)):
            change = (i % 7 - 3) / 100
            brent.append(brent[-1] * (1 + change))
            cnr.append(cnr[-1] * (1 + 0.002 + 0.6 * change))
        levels = pd.DataFrame({"cnr": cnr, "brent": brent}, index=months)
        levels.loc["2022-12":, "cnr"] *= 5  # choc connu seulement après la fenêtre de calibration
        calibrated = estimate_pre_entry_beta(levels, date(2023, 1, 3))
        self.assertEqual(str(calibrated.last_month), "2022-11")
        self.assertEqual(calibrated.observations, 34)
        self.assertAlmostEqual(calibrated.beta, 0.6, places=10)

    def test_euro_exposure_is_converted_using_held_future_price(self):
        days = pd.to_datetime(["2025-01-03", "2025-01-31"])
        prices = pd.DataFrame(80.0, index=days, columns=range(1, 13))
        contracts = pd.DataFrame(
            {"expiry": pd.to_datetime(["2025-01-31"]), "ticker": ["COH25 Comdty"]},
            index=pd.PeriodIndex(["2025-03"], freq="M"),
        )
        history = IceHistory(prices, prices.copy(), prices.copy(), contracts)
        invoices = pd.DataFrame({"month": ["2025-01"], "invoice_eur": [50_000.0]})
        result = recommend_strip_positions(
            history, invoices, entry_date=date(2025, 1, 3),
            diesel_weight=0.2, beta=0.6, eurusd=1.1,
        )
        self.assertAlmostEqual(result.loc[0, "diesel_exposure_eur"], 10_000.0)
        self.assertAlmostEqual(result.loc[0, "barrels"], 82.5)
        self.assertAlmostEqual(result.loc[0, "ice_contracts_fractional"], 0.0825)
        self.assertAlmostEqual(result.loc[0, "mini_equivalents"], 0.825)


if __name__ == "__main__":
    unittest.main()
