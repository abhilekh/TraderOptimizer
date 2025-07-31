"""
Integration tests for the runner module, ensuring proper orchestration of the backtest.
"""

import json
import os
import matplotlib.pyplot as plt
import shutil
import unittest
from unittest.mock import MagicMock, patch

import pandas as pd

from src.runner import run_backtest_for_timeframe, calculate_required_indicators # Changed import
from src.config_manager import (
    load_combined_config,
    RUN_CONFIG_FILE,
    STRATEGIES_CONFIG_FILE,
    INDICATORS_CONFIG_FILE
)

# Constants for test directories
TEST_ROOT_DIR = 'temp_test_runner_project'
TEST_CONFIG_DIR = os.path.join(TEST_ROOT_DIR, 'config')
TEST_DATA_DIR = os.path.join(TEST_ROOT_DIR, 'data')
TEST_OUTPUT_DIR = os.path.join(TEST_ROOT_DIR, 'output')

# Mock config file paths
MOCK_RUN_CONFIG_FILE = os.path.join(TEST_CONFIG_DIR, 'run_config.json')
MOCK_STRATEGIES_CONFIG_FILE = os.path.join(TEST_CONFIG_DIR, 'strategies.json')
MOCK_INDICATORS_CONFIG_FILE = os.path.join(TEST_CONFIG_DIR, 'indicators.json')

class TestRunnerIntegration(unittest.TestCase):
    """Test suite for the runner module's integration capabilities."""

    def setUp(self):
        """Set up a temporary project structure and mock configurations."""
        os.makedirs(TEST_CONFIG_DIR, exist_ok=True)
        os.makedirs(TEST_DATA_DIR, exist_ok=True)
        os.makedirs(TEST_OUTPUT_DIR, exist_ok=True)

        # 1. Mock base data that data_manager.get_historical_data would return
        self.mock_daily_df = pd.DataFrame({
            'Open': [100.0, 101.0, 102.0, 103.0, 104.0, 105.0, 106.0, 107.0, 108.0, 109.0, 110.0],
            'High': [100.5, 101.5, 102.5, 103.5, 104.5, 105.5, 106.5, 107.5, 108.5, 109.5, 110.5],
            'Low': [99.5, 100.5, 101.5, 102.5, 103.5, 104.5, 105.5, 106.5, 107.5, 108.5, 109.5],
            'Close': [100.2, 101.2, 102.2, 103.2, 104.2, 105.2, 106.2, 107.2, 108.2, 109.2, 110.2]
        }, index=pd.to_datetime([f'2023-01-{i:02d}' for i in range(1, 12)]))
        self.mock_daily_df.index.name = 'Date'

        # 2. Mock configuration files for config_manager
        self.mock_run_configs = {
            "test_run_rsi": {
                "ticker": "INTC",
                "start_date": "2023-01-01",
                "end_date": "2023-01-11",
                "initial_capital": 1000,
                "cache_dir": TEST_DATA_DIR,
                "output_dir": TEST_OUTPUT_DIR,
                "output_filename_prefix": "intc_rsi_test",
                "show_plot": False, # Don't show plots during tests
                "strategy_id": "test_rsi_ema_strategy",
                "timeframes": ["Daily"]
            },
            "test_run_ma": {
                "ticker": "MSFT",
                "start_date": "2023-01-01",
                "end_date": "2023-01-11",
                "initial_capital": 2000,
                "cache_dir": TEST_DATA_DIR,
                "output_dir": TEST_OUTPUT_DIR,
                "output_filename_prefix": "msft_ma_test",
                "show_plot": False,
                "strategy_id": "test_ma_crossover_strategy",
                "timeframes": ["Daily"]
            }
        }
        self.mock_strategies_configs = {
            "test_rsi_ema_strategy": {
                "strategy_type": "RSI_EMA_Crossover",
                "indicators_needed": [
                    {"indicator_id": "RSI_Period_14"},
                    {"indicator_id": "EMA_Period_9", "on_indicator": "RSI_Period_14"}
                ],
                "params": {"RSI_Period": 14, "EMA_Period": 9} # Provide params needed by strategy function
            },
            "test_ma_crossover_strategy": {
                "strategy_type": "Moving_Average_Crossover",
                "indicators_needed": [
                    {"indicator_id": "SMA_Period_20", "on_close": True},
                    {"indicator_id": "SMA_Period_50", "on_close": True}
                ],
                "params": {"Short_MA_Period": 20, "Long_MA_Period": 50}
            }
        }
        self.mock_indicators_configs = {
            "RSI_Period_14": {"RSI_Period": 14},
            "EMA_Period_9": {"EMA_Period": 9},
            "SMA_Period_20": {"Short_MA_Period": 20},
            "SMA_Period_50": {"Long_MA_Period": 50}
        }
        
        with open(MOCK_RUN_CONFIG_FILE, 'w', encoding='utf-8') as f:
            json.dump(self.mock_run_configs, f)
        with open(MOCK_STRATEGIES_CONFIG_FILE, 'w', encoding='utf-8') as f:
            json.dump(self.mock_strategies_configs, f)
        with open(MOCK_INDICATORS_CONFIG_FILE, 'w', encoding='utf-8') as f:
            json.dump(self.mock_indicators_configs, f)

        # Patch paths in config_manager
        self.patcher_run_cfg_path = patch('src.config_manager.RUN_CONFIG_FILE', MOCK_RUN_CONFIG_FILE) # Changed patch target
        self.patcher_strat_cfg_path = patch('src.config_manager.STRATEGIES_CONFIG_FILE', MOCK_STRATEGIES_CONFIG_FILE) # Changed patch target
        self.patcher_ind_cfg_path = patch('src.config_manager.INDICATORS_CONFIG_FILE', MOCK_INDICATORS_CONFIG_FILE) # Changed patch target

        self.patcher_run_cfg_path.start()
        self.patcher_strat_cfg_path.start()
        self.patcher_ind_cfg_path.start()

        # Patch get_historical_data from data_manager to return our mock DataFrame
        self.patcher_get_hist_data = patch(
            'src.data_manager.get_historical_data', # Changed patch target
            MagicMock(return_value=self.mock_daily_df.copy())
        )
        self.patcher_get_hist_data.start()

    def tearDown(self):
        """Clean up temporary directories and stop patches."""
        self.patcher_get_hist_data.stop()
        self.patcher_run_cfg_path.stop()
        self.patcher_strat_cfg_path.stop()
        self.patcher_ind_cfg_path.stop()
        if os.path.exists(TEST_ROOT_DIR):
            shutil.rmtree(TEST_ROOT_DIR)
        plt.close('all') # Ensure any lingering plots are closed

    def test_runner_rsi_ema_flow(self):
        """
        Test the full flow for an RSI_EMA_Crossover strategy run:
        config loading -> data fetch (mocked) -> indicator calc -> strategy -> backtest -> output.
        """
        run_config = load_combined_config("test_run_rsi")

        # Run the backtest for the specified timeframe (Daily in this case)
        run_backtest_for_timeframe(
            self.mock_daily_df.copy(), # Pass a copy as it gets modified
            run_config,
            "Daily"
        )

        # Verify output files were created
        expected_metrics_file = os.path.join(TEST_OUTPUT_DIR, "intc_rsi_test_RSI_EMA_Crossover_Daily_metrics.txt")
        expected_plot_file = os.path.join(TEST_OUTPUT_DIR, "intc_rsi_test_RSI_EMA_Crossover_Daily_plot.png")
        expected_trades_file = os.path.join(TEST_OUTPUT_DIR, "intc_rsi_test_RSI_EMA_Crossover_Daily_trades.csv")

        self.assertTrue(os.path.exists(expected_metrics_file))
        self.assertTrue(os.path.exists(expected_plot_file))
        self.assertTrue(os.path.exists(expected_trades_file))

        # Basic check of metrics content (more detailed checks in test_backtester)
        with open(expected_metrics_file, 'r', encoding='utf-8') as f:
            content = f.read()
            self.assertIn("Initial Capital: $1,000.00", content)
            self.assertIn("Total Return", content) # Check if performance analysis ran

        # Verify get_historical_data was called correctly
        self.patcher_get_hist_data.assert_called_once_with(
            "INTC", "2023-01-01", "2023-01-11", interval='1d', cache_dir=TEST_DATA_DIR
        )

    def test_runner_ma_crossover_flow(self):
        """
        Test the full flow for a Moving_Average_Crossover strategy run.
        """
        run_config = load_combined_config("test_run_ma")

        run_backtest_for_timeframe(
            self.mock_daily_df.copy(),
            run_config,
            "Daily"
        )

        # Verify output files were created
        expected_metrics_file = os.path.join(TEST_OUTPUT_DIR, "msft_ma_test_Moving_Average_Crossover_Daily_metrics.txt")
        expected_plot_file = os.path.join(TEST_OUTPUT_DIR, "msft_ma_test_Moving_Average_Crossover_Daily_plot.png")
        expected_trades_file = os.path.join(TEST_OUTPUT_DIR, "msft_ma_test_Moving_Average_Crossover_Daily_trades.csv")

        self.assertTrue(os.path.exists(expected_metrics_file))
        self.assertTrue(os.path.exists(expected_plot_file))
        self.assertTrue(os.path.exists(expected_trades_file))

    def test_calculate_required_indicators(self):
        """
        Test calculate_required_indicators function in isolation
        (though it's called by runner, this verifies its specific logic).
        """
        # Create a simpler config for this specific test
        test_indicators_spec = [
            {"indicator_id": "RSI_Period_14"},
            {"indicator_id": "EMA_Period_9", "on_indicator": "RSI_Period_14"}
        ]
        test_all_params = {
            "RSI_Period": 14,
            "EMA_Period": 9
        }
        
        df_with_indicators = calculate_required_indicators(
            self.mock_daily_df.copy(), test_indicators_spec, test_all_params
        )

        self.assertIn("RSI_14", df_with_indicators.columns)
        self.assertIn("EMA_9", df_with_indicators.columns)
        # Ensure NaNs are handled (usually some at the start of indicators)
        self.assertFalse(df_with_indicators.isnull().all().any()) 
        self.assertGreater(len(df_with_indicators), 0)

    def test_runner_empty_data(self):
        """Test runner handling empty input DataFrame."""
        run_config = load_combined_config("test_run_rsi")
        
        # Patch get_historical_data to return empty df
        with patch('src.data_manager.get_historical_data', MagicMock(return_value=pd.DataFrame())): # Changed patch target
            run_backtest_for_timeframe(pd.DataFrame(), run_config, "Daily")
            # Should print a message but not crash
            # No output files should be created for this run
            expected_metrics_file = os.path.join(TEST_OUTPUT_DIR, "intc_rsi_test_RSI_EMA_Crossover_Daily_metrics.txt")
            self.assertFalse(os.path.exists(expected_metrics_file))


if __name__ == '__main__':
    # Enable logging capture for warnings/info during tests
    import logging
    logging.basicConfig(level=logging.INFO)
    unittest.main(argv=['first-arg-is-ignored'], exit=False)