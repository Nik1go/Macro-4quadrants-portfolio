import unittest

import numpy as np
import pandas as pd

from streamlit_app.cnr_hedge_transport.study import (
    lag_regressions,
    load_fortnightly_study,
    load_official_study,
    load_study_csv,
)


class StudyTests(unittest.TestCase):
    def test_official_cnr_snapshot_regression(self):
        levels = load_official_study()
        self.assertEqual((str(levels.index.min()), str(levels.index.max())), ("2020-01", "2026-07"))
        self.assertAlmostEqual(float(levels.loc[pd.Period("2026-06"), "cnr"]), 229.03)
        self.assertAlmostEqual(float(levels.loc[pd.Period("2026-07"), "cnr"]), 242.15)
        self.assertAlmostEqual(float(levels.loc["2026-01":"2026-07", "cnr"].mean()), 231.39, places=2)
        results, aligned = lag_regressions(levels, max_lag=4)
        best = results.loc[results["R²"].idxmax()]
        self.assertEqual(len(aligned), 74)
        self.assertEqual(int(best["Décalage (mois)"]), 0)
        self.assertAlmostEqual(float(best["β"]), 0.5808736634, places=5)
        self.assertAlmostEqual(float(best["R²"]), 0.685838, places=5)

    def test_official_fortnightly_snapshot(self):
        levels = load_fortnightly_study()
        self.assertEqual((levels.index.min(), levels.index.max()), ("2026-03-H1", "2026-07-H2"))
        self.assertEqual(len(levels), 10)
        self.assertAlmostEqual(float(levels.loc["2026-03-H1", "cnr"]), 237.97)
        self.assertAlmostEqual(float(levels.loc["2026-07-H2", "cnr"]), 256.62)
        results, aligned = lag_regressions(
            levels, max_lag=2, lag_label="Décalage (quinzaines)", min_observations=6
        )
        self.assertEqual(len(aligned), 7)
        self.assertTrue((results["N"] == 7).all())
        self.assertEqual(int(results.loc[results["R²"].idxmax(), "Décalage (quinzaines)"]), 0)
        self.assertAlmostEqual(float(results.loc[0, "R²"]), 0.700296, places=5)

    def test_detects_known_lag_on_a_common_sample(self):
        months = pd.period_range("2020-01", periods=48, freq="M")
        brent_changes = np.array([3 * np.sin(i * 1.7) + 2 * np.cos(i * 0.8) for i in range(48)])
        cnr_changes = np.array([0.4 * brent_changes[i - 2] if i >= 2 else 0 for i in range(48)])
        levels = pd.DataFrame(
            {
                "brent": 80 * np.cumprod(1 + brent_changes / 100),
                "cnr": 100 * np.cumprod(1 + cnr_changes / 100),
            },
            index=months,
        )

        results, aligned = lag_regressions(levels, max_lag=4)

        self.assertEqual(int(results.loc[results["R²"].idxmax(), "Décalage (mois)"]), 2)
        self.assertAlmostEqual(float(results.loc[2, "β"]), 0.4, places=10)
        self.assertAlmostEqual(float(results.loc[2, "R²"]), 1.0, places=10)
        self.assertTrue((results["N"] == len(aligned)).all())

    def test_missing_month_is_not_interpolated(self):
        raw = b"date,cnr,brent\n2020-01,100,80\n2020-03,105,90\n"
        levels = load_study_csv(raw)
        self.assertTrue(pd.isna(levels.loc[pd.Period("2020-02", freq="M"), "cnr"]))

    def test_duplicate_month_is_rejected(self):
        raw = b"date,cnr,brent\n2020-01,100,80\n2020-01-20,105,90\n"
        with self.assertRaisesRegex(ValueError, "Une seule observation"):
            load_study_csv(raw)

    def test_french_semicolon_csv_is_supported(self):
        raw = b"date;cnr;brent\n2020-01;100,5;80,2\n2020-02;101,3;81,4\n"
        levels = load_study_csv(raw)
        self.assertAlmostEqual(levels.loc[pd.Period("2020-01", freq="M"), "cnr"], 100.5)


if __name__ == "__main__":
    unittest.main()
