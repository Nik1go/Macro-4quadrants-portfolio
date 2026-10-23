from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from streamlit_app.ant_colony.snapshot import read_snapshot, write_snapshot


class SnapshotTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.path = Path(self.temporary.name) / 'market_data.zip'
        index = pd.bdate_range('2022-01-03', periods=300)
        tickers = [f'S{i}' for i in range(11)]
        self.universe = pd.DataFrame({'ticker': tickers, 'name': tickers})
        self.prices = pd.DataFrame(100 * np.cumprod(np.full((300, 13), 1.0001), axis=0),
                                   index=index, columns=tickers + ['^FCHI', '^CM100'])
        self.metadata = {'requested_start': '2022-01-01', 'end_exclusive': '2023-03-01'}

    def test_roundtrip_keeps_prices_universe_and_effective_dates(self):
        saved = write_snapshot(self.prices, self.metadata, self.universe, self.path)
        prices, metadata, universe = read_snapshot(self.path)
        pd.testing.assert_frame_equal(prices, self.prices.rename_axis('Date'), check_freq=False)
        pd.testing.assert_frame_equal(universe, self.universe)
        self.assertEqual(saved, metadata)
        self.assertEqual(metadata['snapshot_last_session'], self.prices.index[-1].date().isoformat())
        self.assertEqual(metadata['snapshot_eligible_assets'], 11)

    def test_old_snapshot_and_repeated_reads_never_download(self):
        write_snapshot(self.prices, self.metadata, self.universe, self.path)
        with patch('streamlit_app.ant_colony.core.fetch_market_data', side_effect=AssertionError('Network forbidden')), \
             patch('yfinance.download', side_effect=AssertionError('Network forbidden')), \
             patch('requests.sessions.Session.request', side_effect=AssertionError('Network forbidden')):
            for _ in range(3):
                _, metadata, _ = read_snapshot(self.path)
                self.assertEqual(metadata['end_exclusive'], '2023-03-01')

    def test_incomplete_acquisition_preserves_last_valid_snapshot(self):
        write_snapshot(self.prices, self.metadata, self.universe, self.path)
        original = self.path.read_bytes()
        invalid = self.prices.copy()
        invalid.loc[invalid.index[0], '^CM100'] = np.nan
        with self.assertRaises(ValueError):
            write_snapshot(invalid, self.metadata, self.universe, self.path)
        self.assertEqual(original, self.path.read_bytes())

    def test_missing_snapshot_fails_without_network_fallback(self):
        with patch('streamlit_app.ant_colony.core.fetch_market_data', side_effect=AssertionError('Network forbidden')):
            with self.assertRaises(FileNotFoundError):
                read_snapshot(self.path)


if __name__ == '__main__':
    unittest.main()
