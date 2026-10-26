from pathlib import Path
from itertools import combinations
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from streamlit_app.ant_colony.core import (
    correct_known_prices, prepare_data, validate_euronext_replacement,
)
from streamlit_app.ant_colony.ratios import portfolio_ratios
from streamlit_app.ant_colony.snapshot import read_snapshot, write_snapshot
from streamlit_app.ant_colony.audit_snapshot import local_swaps


class DataQualityTests(unittest.TestCase):
    def test_mid60_known_point_only_and_idempotence(self):
        dates = pd.to_datetime(['2023-11-27', '2023-11-28', '2023-11-29'])
        raw = pd.DataFrame({'^CM100': [13375.97, 14749.16, 13465.81],
                            'ETL.PA': [10., 22., 48.]}, index=dates)
        fixed, log = correct_known_prices(raw)
        self.assertAlmostEqual(fixed.loc['2023-11-28', '^CM100'], 13420.89)
        self.assertEqual(len(log), 1)
        self.assertEqual(raw.iloc[1, 0], 14749.16)
        pd.testing.assert_series_equal(raw['ETL.PA'], fixed['ETL.PA'])
        again, log_again = correct_known_prices(fixed)
        pd.testing.assert_frame_equal(again, fixed)
        self.assertEqual(log_again, [])
        fixed.loc['2023-11-28', '^CM100'] = 13300.
        unchanged, log = correct_known_prices(fixed)
        pd.testing.assert_frame_equal(unchanged, fixed)
        self.assertEqual(log, [])

    def test_missing_neighbour_blocks_estimation(self):
        raw = pd.DataFrame({'^CM100': [14749.16]}, index=pd.to_datetime(['2023-11-28']))
        with self.assertRaises(ValueError):
            correct_known_prices(raw)

    def test_restructuring_exclusions_and_vivendi_legacy_guard(self):
        dates = pd.bdate_range('2022-01-03', '2024-12-18')
        tickers = [f'S{i}' for i in range(10)] + ['ATO.PA', 'EMEIS.PA', 'VIV.PA']
        universe = pd.DataFrame({'ticker': tickers, 'name': tickers})
        prices = pd.DataFrame(100., index=dates, columns=tickers + ['^FCHI', '^CM100'])
        prices.loc['2024-12-09':, 'VIV.PA'] = 22.
        eligible, _, audit = prepare_data(prices, universe)
        self.assertEqual(eligible.columns.tolist(), tickers[:10])
        reasons = audit.set_index('ticker').reason
        self.assertIn('échelles', reasons['ATO.PA'])
        self.assertIn('échelles', reasons['EMEIS.PA'])
        self.assertIn('Vivendi', reasons['VIV.PA'])

    def test_source_overlap_requires_same_scale_and_full_calendar(self):
        dates = pd.bdate_range('2022-01-03', periods=40)
        source = pd.Series(np.linspace(50., 60., 40), index=dates)
        yahoo = source.copy()
        yahoo.iloc[:10] = np.nan
        pd.testing.assert_series_equal(validate_euronext_replacement(yahoo, source, dates), source)
        for invalid in [source * 10., source.iloc[1:]]:
            with self.assertRaises(ValueError):
                validate_euronext_replacement(yahoo, invalid, dates)

    def test_snapshot_persists_estimation_and_its_provenance_once(self):
        dates = pd.bdate_range('2022-01-03', '2023-11-29')
        tickers = [f'S{i}' for i in range(10)]
        universe = pd.DataFrame({'ticker': tickers, 'name': tickers})
        prices = pd.DataFrame(100., index=dates, columns=tickers + ['^FCHI', '^CM100'])
        prices.loc['2023-11-27':'2023-11-29', '^CM100'] = [13375.97, 14749.16, 13465.81]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'snapshot.zip'
            write_snapshot(prices, {}, universe, path)
            fixed, metadata, saved_universe = read_snapshot(path)
            self.assertAlmostEqual(fixed.loc['2023-11-28', '^CM100'], 13420.89)
            self.assertEqual(len(metadata['data_corrections']), 1)
            write_snapshot(fixed, metadata, saved_universe, path)
            self.assertEqual(len(read_snapshot(path)[1]['data_corrections']), 1)

    def test_swap_audit_against_exhaustive_variance_oracle(self):
        rng = np.random.default_rng(11)
        returns = pd.DataFrame(rng.normal(0.001, .01, size=(80, 6)), columns=list('ABCDEF'))
        selected = ['A', 'B', 'C']
        base = returns[selected].mean(axis=1).var(ddof=1)
        actual = local_swaps(returns, selected, -1., 1.)
        for size in (1, 2):
            count, better = 0, 0
            for removed in combinations(selected, size):
                for added in combinations(['D','E','F'], size):
                    candidate = [ticker for ticker in selected if ticker not in removed] + list(added)
                    count += 1
                    better += int(returns[candidate].mean(axis=1).var(ddof=1) < base - 1e-12)
            self.assertEqual(actual[f'{size}_for_{size}']['tested'], count)
            self.assertEqual(actual[f'{size}_for_{size}']['lower_risk'], better)


class RatioRegressionTests(unittest.TestCase):
    def test_positive_portfolio_and_explicit_annual_rate(self):
        daily = np.array([.01, -.005, .007, -.002, .003])
        self.assertGreater(portfolio_ratios(daily)['sharpe'], 0)
        rf = .02
        excess = daily - rf / 252
        actual = portfolio_ratios(daily, risk_free_rate=rf)
        self.assertAlmostEqual(actual['sharpe'], excess.mean() / excess.std(ddof=1) * np.sqrt(252))
        self.assertAlmostEqual(actual['sortino'], excess.mean() / np.sqrt(np.mean(np.minimum(excess, 0)**2)) * np.sqrt(252))
        with self.assertRaises(ValueError):
            portfolio_ratios(daily, risk_free_rate=float('inf'))


if __name__ == '__main__':
    unittest.main()
