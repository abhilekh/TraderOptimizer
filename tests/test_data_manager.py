"""
Unit tests for the data_manager module.
"""

import os
import shutil
import unittest
from unittest.mock import MagicMock, patch

import pandas as pd
import pandas.testing as pdt # For DataFrame comparison

from src.data_manager import get_historical_data, resample_data

# Constants for test data and cache
TEST_DATA_DIR = 'test_cache'
TEST_TICKER = 'TEST'
TEST_START_DATE = '2023-01-01'
TEST_END_DATE = '2023-01-05'
TEST_INTERVAL = '1d'

class TestDataManager(unittest.TestCase):
    """Test suite for the data_manager module."""

    def setUp(self):
        """Set up a temporary cache directory for each test."""
        os.makedirs(TEST_DATA_DIR, exist_ok=True)
        # Define a mock DataFrame that yfinance.download would return
        self.mock_df_raw = pd.DataFrame({
            'Open': [100.0, 101.0, 102.0, 103.0, 104.0],
            'High': [100.5, 101.5, 102.5, 103.5, 104.5],
            'Low': [99.5, 100.5, 101.5, 102.5, 103.5],
            'Close': [100.2, 101.2, 102.2, 103.2, 104.2],
            'Volume': [1000, 1100, 1200, 1300, 1400],
            'Adj Close': [99.0, 100.0, 101.0, 102.0, 103.0]
        }, index=pd.to_datetime([
            '2023-01-01', '2023-01-02', '2023-01-03', '2023-01-04', '2023-01-05'
        ]))
        self.mock_df_raw.index.name = 'Date' # Ensure index name is set

        # The expected DataFrame after processing by _process_downloaded_df
        self.expected_processed_df = pd.DataFrame({
            'Open': [100.0, 101.0, 102.0, 103.0, 104.0],
            'High': [100.5, 101.5, 102.5, 103.5, 104.5],
            'Low': [99.5, 100.5, 101.5, 102.5, 103.5],
            'Close': [100.2, 101.2, 102.2, 103.2, 104.2]
        }, index=pd.to_datetime([
            '2023-01-01', '2023-01-02', '2023-01-03', '2023-01-04', '2023-01-05'
        ]))
        self.expected_processed_df.index.name = 'Date'

    def tearDown(self):
        """Clean up the temporary cache directory after each test."""
        if os.path.exists(TEST_DATA_DIR):
            shutil.rmtree(TEST_DATA_DIR)

    @patch('yfinance.download')
    def test_get_historical_data_download(self, mock_yf_download):
        """Test downloading data when no cache is present."""
        mock_yf_download.return_value = self.mock_df_raw.copy()
        
        df = get_historical_data(
            TEST_TICKER, TEST_START_DATE, TEST_END_DATE, TEST_INTERVAL, TEST_DATA_DIR
        )
        mock_yf_download.assert_called_once_with(
            TEST_TICKER, start=TEST_START_DATE, end=TEST_END_DATE,
            interval=TEST_INTERVAL, progress=False, auto_adjust=True
        )
        pdt.assert_frame_equal(df, self.expected_processed_df)
        # Check if cache file was created
        cache_filepath = os.path.join(
            TEST_DATA_DIR, f"{TEST_TICKER}_{TEST_INTERVAL}_2023-01-01_2023-01-05.csv"
        )
        self.assertTrue(os.path.exists(cache_filepath))

    @patch('yfinance.download')
    def test_get_historical_data_from_cache(self, mock_yf_download):
        """Test loading data from cache."""
        # First, create a mock cache file
        cache_filepath = os.path.join(
            TEST_DATA_DIR, f"{TEST_TICKER}_{TEST_INTERVAL}_2023-01-01_2023-01-05.csv"
        )
        self.expected_processed_df.to_csv(cache_filepath)

        df = get_historical_data(
            TEST_TICKER, TEST_START_DATE, TEST_END_DATE, TEST_INTERVAL, TEST_DATA_DIR
        )
        mock_yf_download.assert_not_called() # yfinance should not be called
        pdt.assert_frame_equal(df, self.expected_processed_df)

    @patch('yfinance.download')
    def test_get_historical_data_download_empty(self, mock_yf_download):
        """Test downloading empty DataFrame from yfinance."""
        mock_yf_download.return_value = pd.DataFrame()
        df = get_historical_data(
            TEST_TICKER, TEST_START_DATE, TEST_END_DATE, TEST_INTERVAL, TEST_DATA_DIR
        )
        self.assertTrue(df.empty)
        mock_yf_download.assert_called_once()
        # No cache file should be created for empty data
        cache_filepath = os.path.join(
            TEST_DATA_DIR, f"{TEST_TICKER}_{TEST_INTERVAL}_2023-01-01_2023-01-05.csv"
        )
        self.assertFalse(os.path.exists(cache_filepath))

    @patch('yfinance.download')
    def test_get_historical_data_corrupted_cache(self, mock_yf_download):
        """Test handling of a corrupted cache file (missing columns)."""
        # Create a corrupted cache file (missing 'Close' column)
        corrupted_df = self.mock_df_raw[['Open', 'High', 'Low']].copy()
        cache_filepath = os.path.join(
            TEST_DATA_DIR, f"{TEST_TICKER}_{TEST_INTERVAL}_2023-01-01_2023-01-05.csv"
        )
        corrupted_df.to_csv(cache_filepath)

        mock_yf_download.return_value = self.mock_df_raw.copy()

        df = get_historical_data(
            TEST_TICKER, TEST_START_DATE, TEST_END_DATE, TEST_INTERVAL, TEST_DATA_DIR
        )
        # yfinance should be called to redownload
        mock_yf_download.assert_called_once()
        pdt.assert_frame_equal(df, self.expected_processed_df)
        self.assertTrue(os.path.exists(cache_filepath)) # New, valid cache should be there

    def test_resample_data_weekly(self):
        """Test resampling daily data to weekly."""
        daily_df = pd.DataFrame({
            'Open': [10, 11, 12, 13, 14, 15, 16, 17, 18, 19],
            'High': [12, 13, 14, 15, 16, 17, 18, 19, 20, 21],
            'Low': [9, 10, 11, 12, 13, 14, 15, 16, 17, 18],
            'Close': [11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
        }, index=pd.to_datetime([
            '2023-01-02', '2023-01-03', '2023-01-04', '2023-01-05', '2023-01-06', # Week 1 (Mon-Fri)
            '2023-01-09', '2023-01-10', '2023-01-11', '2023-01-12', '2023-01-13'  # Week 2 (Mon-Fri)
        ]))
        daily_df.index.name = 'Date'

        weekly_df = resample_data(daily_df, timeframe='W')

        expected_weekly_df = pd.DataFrame({
            'Open': [10, 16],
            'High': [16, 21],
            'Low': [9, 15],
            'Close': [15, 20]
        }, index=pd.to_datetime([
            '2023-01-08', # Week ending Sunday (Jan 8 includes Jan 2-6)
            '2023-01-15'  # Week ending Sunday (Jan 15 includes Jan 9-13)
        ]))
        expected_weekly_df.index.name = 'Date'

        pdt.assert_frame_equal(weekly_df, expected_weekly_df)

    def test_resample_data_empty(self):
        """Test resampling an empty DataFrame."""
        df = pd.DataFrame(columns=['Open', 'High', 'Low', 'Close'])
        resampled_df = resample_data(df, timeframe='W')
        self.assertTrue(resampled_df.empty)


if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False)