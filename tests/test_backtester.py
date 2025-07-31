"""
Unit tests for the backtester module.
"""

import unittest
import os # Import os for path operations and file cleanup
import shutil # Import shutil for rmtree

import pandas as pd
import pandas.testing as pdt

# Adjust path to import modules from the 'src' directory
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Local application imports (from src package)
from src.backtester import analyze_performance, run_backtest # Pylint: Import specific functions

# --- NEW CONSTANT FOR TEST OUTPUT DIRECTORY ---
TEST_OUTPUT_DIR = os.path.join(os.path.dirname(__file__), 'test_output')

class TestBacktester(unittest.TestCase):
    """Test suite for the backtester module."""

    def setUp(self):
        """Set up a DataFrame with signals for backtesting and create test output dir."""
        # Simple data for clear trade simulation
        self.test_data_signals = pd.DataFrame({
            'Close': [100, 102, 105, 103, 106, 104, 107, 109, 108, 110],
            'Signal': [0, 1, 0, 0, -1, 0, 1, 0, -1, 0], # Buy on 102, Sell on 106, Buy on 107, Sell on 108
            'Position': [0, 1, 1, 1, 0, 0, 1, 1, 0, 0] # Expected positions based on signals
        }, index=pd.to_datetime([f'2023-01-{i:02d}' for i in range(1, 11)]))
        self.test_data_signals.index.name = 'Date'

        self.initial_capital = 10000.0

        # Create the test output directory
        os.makedirs(TEST_OUTPUT_DIR, exist_ok=True)

    def tearDown(self):
        """Clean up any files and the test output directory created by tests."""
        # Remove the entire test output directory
        if os.path.exists(TEST_OUTPUT_DIR):
            shutil.rmtree(TEST_OUTPUT_DIR)

    def test_run_backtest_single_trade(self):
        """Test backtest with one buy and one sell trade."""
        # Data: Buy on 102 (Jan 2), Sell on 106 (Jan 5)
        df_for_test = self.test_data_signals.iloc[:5].copy()
        df_for_test.loc[pd.to_datetime('2023-01-05'), 'Signal'] = -1
        df_for_test.loc[pd.to_datetime('2023-01-05'), 'Position'] = 0

        df_equity, trades_df, num_trades = run_backtest(df_for_test, self.initial_capital)

        self.assertIsInstance(df_equity, pd.DataFrame)
        self.assertIsInstance(trades_df, pd.DataFrame)
        self.assertEqual(num_trades, 1)

        # Verify the trade details
        self.assertEqual(len(trades_df), 1)
        trade = trades_df.iloc[0]
        self.assertEqual(trade['Entry_Date'], pd.to_datetime('2023-01-02'))
        self.assertEqual(trade['Entry_Price'], 102.0)
        self.assertEqual(trade['Exit_Date'], pd.to_datetime('2023-01-05'))
        self.assertEqual(trade['Exit_Price'], 106.0)
        self.assertEqual(trade['Shares'], int(self.initial_capital / 102.0)) # Approx 98 shares
        self.assertAlmostEqual(trade['Dollar_PL'], 4 * trade['Shares'], places=2) # 4 dollar profit per share

        # Verify equity curve
        # Initial: 10000
        # Buy on Jan 2 at 102: shares = 10000 // 102 = 98. cash = 10000 - 98*102 = 10000 - 9996 = 4
        # Equity Jan 2: 4 + 98*102 = 10000
        # Equity Jan 3: 4 + 98*105 = 10294 (unrealized gain)
        # Equity Jan 4: 4 + 98*103 = 10098 (unrealized gain)
        # Sell on Jan 5 at 106: cash = 4 + 98*106 = 4 + 10388 = 10392
        # Final Equity: 10392
        
        expected_equity_curve_values = [
            10000.0, 10000.0, 10294.0, 10098.0, 10392.0 # Adjusted based on calculation
        ]
        # Pylint: use check_exact=False for float comparisons
        pdt.assert_series_equal(
            df_equity['Equity_Curve'].iloc[:5],
            pd.Series(expected_equity_curve_values, index=df_for_test.index),
            check_exact=False,
            check_dtype=False
        )


    def test_run_backtest_multiple_trades(self):
        """Test backtest with multiple buy/sell cycles."""
        df_equity, trades_df, num_trades = run_backtest(self.test_data_signals, self.initial_capital)

        self.assertEqual(num_trades, 2) # Two completed trades
        self.assertEqual(len(trades_df), 2)

        # Check first trade
        trade1 = trades_df.iloc[0]
        self.assertEqual(trade1['Entry_Date'], pd.to_datetime('2023-01-02'))
        self.assertEqual(trade1['Exit_Date'], pd.to_datetime('2023-01-05'))

        # Check second trade
        trade2 = trades_df.iloc[1]
        self.assertEqual(trade2['Entry_Date'], pd.to_datetime('2023-01-07'))
        self.assertEqual(trade2['Exit_Date'], pd.to_datetime('2023-01-09'))
        self.assertAlmostEqual(trade2['Entry_Price'], 107.0, places=2)
        self.assertAlmostEqual(trade2['Exit_Price'], 108.0, places=2)
        # Shares for second trade would be based on capital *after* first trade
        # Capital after first trade: initial_capital + (106-102)*98 = 10000 + 392 = 10392
        # Shares for second trade: 10392 // 107 = 97
        self.assertEqual(trade2['Shares'], 97)
        self.assertAlmostEqual(trade2['Dollar_PL'], (108-107)*97, places=2)

    def test_run_backtest_open_position_at_end(self):
        """Test backtest when a position is open at the end of the data."""
        # Data: Buy on 102 (Jan 2), no final sell signal
        df_for_test = self.test_data_signals.iloc[:4].copy() # Ends with an open position
        
        df_equity, trades_df, num_trades = run_backtest(df_for_test, self.initial_capital)
        
        self.assertEqual(num_trades, 1) # One pseudo-trade for closing at the end
        self.assertEqual(len(trades_df), 1)
        
        trade = trades_df.iloc[0]
        self.assertEqual(trade['Entry_Date'], pd.to_datetime('2023-01-02'))
        self.assertEqual(trade['Exit_Date'], pd.to_datetime('2023-01-04')) # Last date in data
        self.assertAlmostEqual(trade['Exit_Price'], 103.0, places=2)

    def test_run_backtest_no_trades(self):
        """Test backtest with no buy/sell signals."""
        no_signal_df = self.test_data_signals.copy()
        no_signal_df['Signal'] = 0
        no_signal_df['Position'] = 0
        
        df_equity, trades_df, num_trades = run_backtest(no_signal_df, self.initial_capital)
        
        self.assertEqual(num_trades, 0)
        self.assertTrue(trades_df.empty)
        # Equity should just remain flat
        self.assertTrue(all(df_equity['Equity_Curve'] == self.initial_capital))


    def test_analyze_performance_positive(self):
        """Test performance analysis for a profitable scenario."""
        initial = 10000.0
        final = 11000.0
        trades = pd.DataFrame([
            {'Entry_Date': '2023-01-01', 'Entry_Price': 100, 'Exit_Date': '2023-01-02', 'Exit_Price': 110, 'Shares': 10, 'Dollar_PL': 100, 'Percent_PL': 10},
            {'Entry_Date': '2023-01-03', 'Entry_Price': 120, 'Exit_Date': '2023-01-04', 'Exit_Price': 130, 'Shares': 5, 'Dollar_PL': 50, 'Percent_PL': 8.33}
        ])
        
        # --- MODIFICATION: Use TEST_OUTPUT_DIR ---
        metrics_output_path = os.path.join(TEST_OUTPUT_DIR, "test_metrics.txt")
        analyze_performance(initial, final, trades, metrics_output_path)
        
        self.assertTrue(os.path.exists(metrics_output_path))
        self.assertTrue(os.path.exists(metrics_output_path.replace('.txt', '_trades.csv')))

        with open(metrics_output_path, 'r', encoding='utf-8') as f:
            content = f.read()
        self.assertIn("Total Profit/Loss: $1,000.00", content)
        self.assertIn("Total Return (%): 10.00%", content)
        self.assertIn("Number of Trades: 2", content)
        self.assertIn("Winning Trades: 2", content)
        self.assertIn("Win Rate (%): 100.00%", content)
        # No need to manually remove here, tearDown handles directory removal


    def test_analyze_performance_negative(self):
        """Test performance analysis for a losing scenario."""
        initial = 10000.0
        final = 9000.0
        trades = pd.DataFrame([
            {'Entry_Date': '2023-01-01', 'Entry_Price': 100, 'Exit_Date': '2023-01-02', 'Exit_Price': 90, 'Shares': 10, 'Dollar_PL': -100, 'Percent_PL': -10},
            {'Entry_Date': '2023-01-03', 'Entry_Price': 80, 'Exit_Date': '2023-01-04', 'Exit_Price': 75, 'Shares': 5, 'Dollar_PL': -25, 'Percent_PL': -6.25}
        ])

        # --- MODIFICATION: Use TEST_OUTPUT_DIR ---
        metrics_output_path = os.path.join(TEST_OUTPUT_DIR, "test_metrics_negative.txt")
        analyze_performance(initial, final, trades, metrics_output_path)
        
        with open(metrics_output_path, 'r', encoding='utf-8') as f:
            content = f.read()
        self.assertIn("Total Profit/Loss: $-1,000.00", content)
        self.assertIn("Total Return (%): -10.00%", content)


    def test_analyze_performance_no_trades(self):
        """Test performance analysis with no completed trades."""
        initial = 10000.0
        final = 10000.0
        trades = pd.DataFrame(columns=[
            'Entry_Date', 'Entry_Price', 'Exit_Date', 'Exit_Price',
            'Shares', 'Dollar_PL', 'Percent_PL'
        ])

        # --- MODIFICATION: Use TEST_OUTPUT_DIR ---
        metrics_output_path = os.path.join(TEST_OUTPUT_DIR, "test_metrics_no_trades.txt")
        analyze_performance(initial, final, trades, metrics_output_path)
        
        with open(metrics_output_path, 'r', encoding='utf-8') as f:
            content = f.read()
        self.assertIn("No completed trades for analysis.", content)
        # Check that the trades CSV file was NOT created if there are no trades
        self.assertFalse(os.path.exists(metrics_output_path.replace('.txt', '_trades.csv')))


if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False)