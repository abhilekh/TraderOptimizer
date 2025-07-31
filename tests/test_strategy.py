"""
Unit tests for the strategy module.
"""

# Standard library imports
import os
import unittest
from typing import Dict, Any

# Third-party imports
import pandas as pd
import pandas.testing as pdt

# Adjust path to import modules from the 'src' directory
# This allows tests to correctly import modules like 'strategy'
# when running from the project root using 'python -m unittest discover'.
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

# Local application imports (from src package)
from src.strategy import (
    apply_moving_average_crossover_strategy,
    apply_rsi_ema_crossover_strategy
)

class TestStrategy(unittest.TestCase):
    """Test suite for the strategy module."""

    def setUp(self):
        """Set up common test dataframes for strategies."""
        # Dummy data for RSI_EMA_Crossover strategy
        # Extend this data to ensure enough non-NaN values after indicator calculation periods
        # Data designed for clear signals:
        # RSI crosses EMA up, RSI > 50 (buy)
        # EMA crosses RSI up, RSI < 50 (sell)
        self.rsi_ema_test_data = pd.DataFrame({
            'Close': [100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 109, 108],
            'RSI_14': [40, 42, 51, 55, 60, 65, 68, 62, 55, 48, 45, 40, 35],
            'EMA_9': [42, 43, 49, 52, 57, 62, 67, 65, 59, 52, 49, 45, 40]
        })
        # Generate enough dates to match the length of your dummy data.
        # Make it long enough for indicators to populate + some extra.
        num_rows = len(self.rsi_ema_test_data)
        self.rsi_ema_test_data.index = pd.to_datetime([f'2023-01-{i:02d}' for i in range(1, num_rows + 1)])
        self.rsi_ema_test_data.index.name = 'Date'

        # Dummy data for Moving_Average_Crossover strategy
        # Longer data for MA periods
        # Buy on (Short_MA > Long_MA) AND (Prev_Short_MA <= Prev_Long_MA)
        # Sell on (Short_MA < Long_MA) AND (Prev_Short_MA >= Prev_Long_MA)
        self.ma_crossover_test_data = pd.DataFrame({
            'Close': [100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 109, 108, 107, 106],
            'SMA_50': [50, 50, 50, 50, 51, 52, 53, 51, 49, 48, 47, 46, 45, 44, 43],
            'SMA_200': [55, 54, 53, 52, 50, 50, 50, 50, 50, 51, 52, 53, 54, 55, 56]
        })
        num_rows_ma = len(self.ma_crossover_test_data)
        self.ma_crossover_test_data.index = pd.to_datetime([f'2023-01-{i:02d}' for i in range(1, num_rows_ma + 1)])
        self.ma_crossover_test_data.index.name = 'Date'

    def test_rsi_ema_crossover_buy_signal(self):
        """Test buy signal generation for RSI_EMA_Crossover."""
        params: Dict[str, Any] = {"RSI_Period": 14, "EMA_Period": 9}
        df_result = apply_rsi_ema_crossover_strategy(self.rsi_ema_test_data.copy(), params)

        # Expected behavior:
        # 01-01: RSI=40, EMA=42 (RSI<EMA, RSI<50) -> No Signal, Position 0
        # 01-02: RSI=45, EMA=47 (RSI<EMA, RSI<50) -> No Signal, Position 0
        # 01-03: RSI=51, EMA=49 (RSI>EMA AND RSI>50) -> BUY, Position 1
        # 01-04: RSI=55, EMA=52 (RSI>EMA AND RSI>50) -> No Signal, Position 1
        # 01-05: RSI=60, EMA=57 (RSI>EMA AND RSI>50) -> No Signal, Position 1
        self.assertEqual(df_result.loc['2023-01-03', 'Signal'], 1)
        self.assertEqual(df_result.loc['2023-01-03', 'Position'], 1)
        self.assertEqual(df_result.loc['2023-01-05', 'Position'], 1) # Still in position

    def test_rsi_ema_crossover_sell_signal(self):
        """Test sell signal generation for RSI_EMA_Crossover."""
        params: Dict[str, Any] = {"RSI_Period": 14, "EMA_Period": 9}
        df_result = apply_rsi_ema_crossover_strategy(self.rsi_ema_test_data.copy(), params)
        
        # Expected behavior:
        # 01-03: BUY (RSI=51, EMA=49)
        # ... (holding position) ...
        # 01-10: RSI=55, EMA=59 (RSI<EMA, RSI>50) -> No Signal, Position 1
        # 01-11: RSI=48, EMA=52 (RSI<EMA AND RSI<50) -> SELL, Position 0
        self.assertEqual(df_result.loc['2023-01-03', 'Signal'], 1) # Expected buy
        self.assertEqual(df_result.loc['2023-01-11', 'Signal'], -1) # Expected sell (exit)
        self.assertEqual(df_result.loc['2023-01-11', 'Position'], 0) # Should exit position

    def test_rsi_ema_crossover_exit_buy_condition_lost(self):
        """Test exiting long position when buy condition is no longer met."""
        # Create a scenario where buy is triggered, then just one condition breaks (RSI drops below 50)
        test_data_modified = pd.DataFrame({
            'Close':    [100, 101, 102, 103, 104, 105],
            'RSI_14':   [ 40,  45,  51,  55,  48,  52], # RSI goes below 50, but RSI > EMA still true on 01-05
            'EMA_9':    [ 42,  47,  49,  52,  50,  50]
        }, index=pd.to_datetime([f'2023-01-{i:02d}' for i in range(1, 7)]))
        test_data_modified.index.name = 'Date'

        params: Dict[str, Any] = {"RSI_Period": 14, "EMA_Period": 9}
        df_result = apply_rsi_ema_crossover_strategy(test_data_modified.copy(), params)

        # Expected behavior:
        # 01-03: BUY (RSI=51, EMA=49)
        # 01-04: Holding (RSI=55, EMA=52)
        # 01-05: RSI=48, EMA=50. Buy condition (RSI>EMA AND RSI>50) is FALSE.
        #        Explicit sell (EMA>RSI AND RSI<50) is FALSE (EMA not > RSI).
        #        So, it should exit due to `not buy_conditions.iloc[i]`
        self.assertEqual(df_result.loc['2023-01-03', 'Signal'], 1) # Buy
        self.assertEqual(df_result.loc['2023-01-04', 'Position'], 1) # Still in position
        self.assertEqual(df_result.loc['2023-01-05', 'Signal'], -1) # Sell
        self.assertEqual(df_result.loc['2023-01-05', 'Position'], 0) # Exited

    def test_rsi_ema_crossover_missing_indicators(self):
        """Test RSI_EMA_Crossover with missing required indicator columns."""
        df_missing_rsi = self.rsi_ema_test_data.drop(columns=['RSI_14'])
        params: Dict[str, Any] = {"RSI_Period": 14, "EMA_Period": 9}
        df_result = apply_rsi_ema_crossover_strategy(df_missing_rsi, params)
        self.assertTrue(df_result.empty)

    def test_ma_crossover_buy_signal(self):
        """Test buy signal generation for Moving_Average_Crossover."""
        params: Dict[str, Any] = {"Short_MA_Period": 50, "Long_MA_Period": 200}
        df_result = apply_moving_average_crossover_strategy(self.ma_crossover_test_data.copy(), params)
        
        # Expected behavior from ma_crossover_test_data:
        # Date        Close  SMA_50  SMA_200
        # 2023-01-01    100      50       55  (50 < 55)
        # 2023-01-02    101      50       54  (50 < 54)
        # 2023-01-03    102      50       53  (50 < 53)
        # 2023-01-04    103      50       52  (50 < 52)
        # 2023-01-05    104      51       50  (51 > 50) AND (prev_short 50 <= prev_long 52) -> BUY!
        # 2023-01-06    105      52       50  (52 > 50)
        self.assertEqual(df_result.loc['2023-01-05', 'Signal'], 1)
        self.assertEqual(df_result.loc['2023-01-05', 'Position'], 1)
        self.assertEqual(df_result.loc['2023-01-06', 'Position'], 1) # Still in position


    def test_ma_crossover_sell_signal(self):
        """Test sell signal generation for Moving_Average_Crossover."""
        params: Dict[str, Any] = {"Short_MA_Period": 50, "Long_MA_Period": 200}
        df_result = apply_moving_average_crossover_strategy(self.ma_crossover_test_data.copy(), params)

        # Expected behavior from ma_crossover_test_data:
        # ... (Buy on 01-05) ...
        # 2023-01-07    106      53       50  (53 > 50)
        # 2023-01-08    105      51       50  (51 > 50)
        # 2023-01-09    104      49       50  (49 < 50) AND (prev_short 51 >= prev_long 50) -> SELL!
        self.assertEqual(df_result.loc['2023-01-05', 'Signal'], 1) # Expected buy
        self.assertEqual(df_result.loc['2023-01-09', 'Signal'], -1) # Expected sell
        self.assertEqual(df_result.loc['2023-01-09', 'Position'], 0) # Exited

    def test_ma_crossover_missing_indicators(self):
        """Test Moving_Average_Crossover with missing required indicator columns."""
        df_missing_sma = self.ma_crossover_test_data.drop(columns=['SMA_50'])
        params: Dict[str, Any] = {"Short_MA_Period": 50, "Long_MA_Period": 200}
        df_result = apply_moving_average_crossover_strategy(df_missing_sma, params)
        self.assertTrue(df_result.empty)

    def test_strategy_empty_dataframe(self):
        """Test strategies with an empty input DataFrame."""
        empty_df = pd.DataFrame(columns=['Close', 'RSI_14', 'EMA_9'], dtype=float)
        empty_df.index.name = 'Date'
        
        params_rsi: Dict[str, Any] = {"RSI_Period": 14, "EMA_Period": 9}
        result_rsi = apply_rsi_ema_crossover_strategy(empty_df, params_rsi)
        self.assertTrue(result_rsi.empty)

        params_ma: Dict[str, Any] = {"Short_MA_Period": 50, "Long_MA_Period": 200}
        result_ma = apply_moving_average_crossover_strategy(empty_df, params_ma)
        self.assertTrue(result_ma.empty)

if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False)