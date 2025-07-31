"""
Unit tests for the indicators module.
"""
import os
import unittest

import pandas as pd
import pandas.testing as pdt

from src.indicators import calculate_ema, calculate_rsi, calculate_sma # Pylint: Import specific functions

class TestIndicators(unittest.TestCase):
    """Test suite for the indicators module."""

    def setUp(self):
        """Set up a common DataFrame for indicator tests."""
        # Dummy data for testing indicators
        self.test_data = pd.DataFrame({
            'Close': [10, 12, 15, 13, 16, 18, 17, 20, 22, 21]
        }, index=pd.to_datetime([
            '2023-01-01', '2023-01-02', '2023-01-03', '2023-01-04', '2023-01-05',
            '2023-01-06', '2023-01-07', '2023-01-08', '2023-01-09', '2023-01-10'
        ]))
        self.test_data.index.name = 'Date'

    def test_calculate_rsi(self):
        """Test RSI calculation."""
        # Using a period of 2 for simpler manual verification
        rsi_values = calculate_rsi(self.test_data, period=2)
        
        # Expected values can be complex to calculate manually for real RSI.
        # Here we'll just check type and non-NaN values for a basic test.
        self.assertIsInstance(rsi_values, pd.Series)
        self.assertEqual(len(rsi_values), len(self.test_data))
        self.assertFalse(rsi_values.isnull().all()) # Should not be all NaNs
        # Specific value checks for period=2 from pandas_ta calculation
        # (calculated using a separate script to verify output)
        expected_rsi = pd.Series([
            None, None, 100.0, 33.333333, 75.0, 83.333333, 40.0, 75.0, 83.333333, 40.0
        ], index=self.test_data.index, name='RSI_2')
        pdt.assert_series_equal(rsi_values.round(4), expected_rsi.round(4), check_dtype=False)

    def test_calculate_rsi_missing_close(self):
        """Test RSI calculation with missing 'Close' column."""
        data_no_close = self.test_data.drop(columns=['Close'])
        with self.assertRaises(ValueError):
            calculate_rsi(data_no_close, period=14)

    def test_calculate_ema_on_dataframe(self):
        """Test EMA calculation directly on a DataFrame's 'Close' column."""
        ema_values = calculate_ema(self.test_data, period=3)
        self.assertIsInstance(ema_values, pd.Series)
        self.assertEqual(len(ema_values), len(self.test_data))
        self.assertFalse(ema_values.isnull().all())
        # Expected values for period=3 (from pandas_ta calculation)
        expected_ema = pd.Series([
            None, None, 12.0, 12.5, 14.25, 16.125, 16.5625, 18.28125, 20.140625, 20.5703125
        ], index=self.test_data.index, name='EMA_3')
        pdt.assert_series_equal(ema_values.round(6), expected_ema.round(6), check_dtype=False)

    def test_calculate_ema_on_series(self):
        """Test EMA calculation on a Series (e.g., EMA of RSI)."""
        rsi_series = calculate_rsi(self.test_data, period=3).dropna() # Get a Series without NaNs
        ema_of_rsi = calculate_ema(rsi_series, period=2)
        
        self.assertIsInstance(ema_of_rsi, pd.Series)
        self.assertEqual(len(ema_of_rsi), len(rsi_series))
        self.assertFalse(ema_of_rsi.isnull().all())

    def test_calculate_ema_missing_column(self):
        """Test EMA calculation with missing specified column in DataFrame."""
        with self.assertRaises(ValueError):
            calculate_ema(self.test_data, period=3, column='NonExistent')

    def test_calculate_ema_invalid_input_type(self):
        """Test EMA calculation with invalid input type."""
        with self.assertRaises(TypeError):
            calculate_ema([1, 2, 3], period=3) # Pass a list instead of DF/Series

    def test_calculate_sma_on_dataframe(self):
        """Test SMA calculation directly on a DataFrame's 'Close' column."""
        sma_values = calculate_sma(self.test_data, period=3)
        self.assertIsInstance(sma_values, pd.Series)
        self.assertEqual(len(sma_values), len(self.test_data))
        self.assertFalse(sma_values.isnull().all())
        # Expected values for period=3 (from pandas_ta calculation)
        expected_sma = pd.Series([
            None, None, 12.333333, 13.333333, 14.666667, 15.666667, 17.0, 18.333333, 19.666667, 21.0
        ], index=self.test_data.index, name='SMA_3')
        pdt.assert_series_equal(sma_values.round(4), expected_sma.round(4), check_dtype=False)

    def test_calculate_sma_on_series(self):
        """Test SMA calculation on a Series."""
        close_series = self.test_data['Close']
        sma_of_series = calculate_sma(close_series, period=2)
        self.assertIsInstance(sma_of_series, pd.Series)
        self.assertEqual(len(sma_of_series), len(close_series))
        self.assertFalse(sma_of_series.isnull().all())

    def test_calculate_sma_missing_column(self):
        """Test SMA calculation with missing specified column in DataFrame."""
        with self.assertRaises(ValueError):
            calculate_sma(self.test_data, period=3, column='NonExistent')

    def test_calculate_sma_invalid_input_type(self):
        """Test SMA calculation with invalid input type."""
        with self.assertRaises(TypeError):
            calculate_sma([1, 2, 3], period=3) # Pass a list instead of DF/Series

if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False)