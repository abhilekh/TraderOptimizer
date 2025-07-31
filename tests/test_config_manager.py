"""
Unit tests for the config_manager module.
"""

import json
import os
import shutil
import unittest
from unittest.mock import patch # Make sure patch is imported

from src.config_manager import (
    load_combined_config,
    DEFAULT_RUN_CONFIG_ID,
    DEFAULT_STRATEGY_ID,
    DEFAULT_INDICATOR_ID,
    # Import the original constants for patching
    RUN_CONFIG_FILE as ORIGINAL_RUN_CONFIG_FILE,
    STRATEGIES_CONFIG_FILE as ORIGINAL_STRATEGIES_CONFIG_FILE,
    INDICATORS_CONFIG_FILE as ORIGINAL_INDICATORS_CONFIG_FILE
)

# Constants for *static* test config directory and file names
# These files are now expected to exist on disk at this location
TEST_CONFIG_DIR = os.path.join(os.path.dirname(__file__), 'mock_configs')
TEST_RUN_CONFIG_FILE = os.path.join(TEST_CONFIG_DIR, 'test_run_config.json')
TEST_STRATEGIES_CONFIG_FILE = os.path.join(TEST_CONFIG_DIR, 'test_strategies.json')
TEST_INDICATORS_CONFIG_FILE = os.path.join(TEST_CONFIG_DIR, 'test_indicators.json')

class TestConfigManager(unittest.TestCase):
    """Test suite for the config_manager module."""

    @classmethod
    def setUpClass(cls):
        """Set up patches once for all tests in this class."""
        # No need to create directory or files, they are static
        # The user's request implies these files are now managed externally.

        # Patch the file paths used by config_manager to point to our static test files
        cls.patcher_run_config = patch('src.config_manager.RUN_CONFIG_FILE', TEST_RUN_CONFIG_FILE)
        cls.patcher_strategies_config = patch('src.config_manager.STRATEGIES_CONFIG_FILE', TEST_STRATEGIES_CONFIG_FILE)
        cls.patcher_indicators_config = patch('src.config_manager.INDICATORS_CONFIG_FILE', TEST_INDICATORS_CONFIG_FILE)

        cls.patcher_run_config.start()
        cls.patcher_strategies_config.start()
        cls.patcher_indicators_config.start()

    @classmethod
    def tearDownClass(cls):
        """Clean up patches once after all tests in this class."""
        cls.patcher_run_config.stop()
        cls.patcher_strategies_config.stop()
        cls.patcher_indicators_config.stop()
        # No need to remove TEST_CONFIG_DIR as it's a static directory now

    # Note: No setUp/tearDown instance methods needed as setupClass/tearDownClass handles files

    def test_load_default_config(self):
        """Test loading the default run configuration."""
        config = load_combined_config() # Should load 'default'
        self.assertEqual(config["ticker"], "AAPL")
        self.assertEqual(config["start_date"], "2023-01-01")
        self.assertEqual(config["strategy_type"], "RSI_EMA_Crossover")
        self.assertEqual(config["RSI_Period"], 14)
        self.assertIn("indicators_needed_spec", config)
        self.assertTrue(config["show_plot"])
        # Check strategy params, which are flattened
        self.assertEqual(config["rsi_threshold"], 50)

    def test_load_specific_run_config_override_all(self):
        """Test loading a specific run config that overrides multiple defaults."""
        config = load_combined_config("test_run_1")
        config_direct = {}
        try :
            with open(TEST_RUN_CONFIG_FILE, 'r', encoding='utf-8') as f_obj:
                config_direct = json.load(f_obj)
        except json.JSONDecodeError as jderr:
             self.assertLogs('{TEST_RUN_CONFIG_FILE} has not invalid json', level='ERROR')
        self.assertEqual(config, config_direct)

    def test_load_specific_run_config_inherit_strategy(self):
        """Test loading a specific run config that inherits the default strategy."""
        config = load_combined_config("test_run_2")
        self.assertEqual(config["ticker"], "GOOGL")
        self.assertEqual(config["start_date"], "2023-01-01") # Inherited from default run config
        self.assertEqual(config["strategy_type"], "RSI_EMA_Crossover") # Inherited default strategy
        self.assertEqual(config["RSI_Period"], 14) # Inherited default indicator period

    def test_file_not_found(self):
        """Test FileNotFoundError when a config file is missing."""
        # Temporarily remove one file for this test
        # Need to ensure file is put back in case other tests depend on it
        original_content = ""
        try:
            with open(TEST_RUN_CONFIG_FILE, 'r', encoding='utf-8') as f:
                original_content = f.read()
            os.remove(TEST_RUN_CONFIG_FILE)
            
            with self.assertRaises(FileNotFoundError):
                load_combined_config()
        finally:
            # Always put the file back so subsequent tests don't fail
            if original_content:
                with open(TEST_RUN_CONFIG_FILE, 'w', encoding='utf-8') as f:
                    f.write(original_content)


    def test_json_decode_error(self):
        """Test JSONDecodeError for malformed JSON."""
        # Temporarily corrupt the file for this test
        original_content = ""
        try:
            with open(TEST_RUN_CONFIG_FILE, 'r', encoding='utf-8') as f:
                original_content = f.read()
            with open(TEST_RUN_CONFIG_FILE, 'w', encoding='utf-8') as f:
                f.write("{invalid json")
            
            with self.assertRaises(json.JSONDecodeError):
                load_combined_config()
        finally:
            # Restore the file
            if original_content:
                with open(TEST_RUN_CONFIG_FILE, 'w', encoding='utf-8') as f:
                    f.write(original_content)


    def test_missing_strategy_id_fallback(self):
        """Test fallback when a specified strategy_id doesn't exist."""
        # This will trigger a warning. Check for the warning message.
        # The 'non_existent_strategy' ID is in the mock_run_configs, and should trigger fallback
        with self.assertLogs('src.config_manager', level='WARNING') as cm:
            config = load_combined_config("test_run_missing_strategy")
            self.assertTrue(any("Warning: Strategy ID 'non_existent_strategy' not found." in s for s in cm.output))
        
        self.assertEqual(config["ticker"], "TEST_MISSING") # From test_run_missing_strategy
        self.assertEqual(config["strategy_type"], "RSI_EMA_Crossover") # Falls back to default strategy

    def test_missing_default_strategy(self):
        """Test behavior when the 'default' strategy is missing in strategies.json."""
        # Temporarily remove the default strategy entry
        original_strategies_content = ""
        try:
            with open(TEST_STRATEGIES_CONFIG_FILE, 'r', encoding='utf-8') as f:
                original_strategies_content = f.read()
            
            temp_strategies_data = json.loads(original_strategies_content)
            if 'default' in temp_strategies_data: # Ensure 'default' key exists in mock data for this test
                del temp_strategies_data['default']
            
            with open(TEST_STRATEGIES_CONFIG_FILE, 'w', encoding='utf-8') as f:
                json.dump(temp_strategies_data, f)
            
            with self.assertLogs('src.config_manager', level='WARNING') as cm:
                config = load_combined_config() # Should use the (now modified) strategies.json
                self.assertTrue(any("Warning: 'default' section not found in Strategy config." in s for s in cm.output))
            
            self.assertEqual(config["ticker"], "AAPL") # Inherited from run_config default
            self.assertIsNone(config.get("strategy_type")) # No strategy_type from strategies.json
        finally:
            # Restore the strategies file
            if original_strategies_content:
                with open(TEST_STRATEGIES_CONFIG_FILE, 'w', encoding='utf-8') as f:
                    f.write(original_strategies_content)

    def test_missing_indicator_id_fallback(self):
        """Test behavior when a specific indicator ID in strategies.json is missing in indicators.json."""
        # The 'NON_EXISTENT_SMA' indicator is in 'custom_strategy_missing_indicator' in test_strategies.json
        # This test ensures load_combined_config handles it gracefully without crashing.
        # It should still return a merged config, but the specific indicator params won't be in it.
        config = load_combined_config("test_missing_indicator_run")
        self.assertEqual(config["ticker"], "TEST_IND_MISSING")
        self.assertEqual(config["strategy_type"], "Moving_Average_Crossover")
        # Assert that the parameters for 'NON_EXISTENT_SMA' are NOT in the final config
        self.assertNotIn("NON_EXISTENT_SMA_Period", config) # This param should not be found

if __name__ == '__main__':
    # Enable logging capture for warnings during tests
    import logging
    logging.basicConfig(level=logging.WARNING)
    unittest.main(argv=['first-arg-is-ignored'], exit=False)