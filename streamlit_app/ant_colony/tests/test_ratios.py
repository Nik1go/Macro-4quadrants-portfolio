import math
from pathlib import Path
import sys
import unittest
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ratios import portfolio_ratios

class RatioTests(unittest.TestCase):
    def test_known_sample_and_full_sample_downside(self):
        ratios = portfolio_ratios([.01, -.01, .02, 0.])
        self.assertAlmostEqual(ratios["sharpe"], .005 / math.sqrt(.0005 / 3) * math.sqrt(252))
        self.assertAlmostEqual(ratios["sortino"], math.sqrt(252))

    def test_undefined_risk(self):
        ratios = portfolio_ratios([0., 0., 0.])
        self.assertTrue(math.isnan(ratios["sharpe"]))
        self.assertTrue(math.isnan(ratios["sortino"]))
        self.assertTrue(math.isnan(portfolio_ratios([.01, .02])["sortino"]))

    def test_invalid_data(self):
        for values in [[], [.01], [.01, float("nan")]]:
            with self.assertRaises(ValueError):
                portfolio_ratios(values)
