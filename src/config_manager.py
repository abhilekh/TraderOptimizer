from pathlib import Path
from typing import Any, Dict, List
# Ensure this import points to your shared utility module
from src.utils import UtilsJson


class ConfigManager:
    """
    ConfigManager is a dedicated class for managing hierarchical configuration files in a Python application,
    particularly for backtesting or trading systems. It leverages UtilsJson for efficient merging
    and inheritance resolution.

    Key Responsibilities:
    - Load configuration data from multiple JSON files (run, strategies, indicators).
    - Merge configurations hierarchically, supporting inheritance via 'base_id' references using UtilsJson.
    - Assemble a comprehensive configuration dictionary for a given run, including all relevant strategy and indicator parameters.
    """

    def __init__(
        self,
        run_config_path: Path,
        strategies_config_path: Path,
        indicators_config_path: Path,
    ):
        """
        Initializes the ConfigManager with paths to the configuration files.
        """
        self.run_config_path = run_config_path
        self.strategies_config_path = strategies_config_path
        self.indicators_config_path = indicators_config_path

        # Load all configuration files using UtilsJson
        self.run_configs = UtilsJson.read_json_file(self.run_config_path)
        self.strategies_configs = UtilsJson.read_json_file(
            self.strategies_config_path)
        self.indicators_configs = UtilsJson.read_json_file(
            self.indicators_config_path)

    def _get_merged_section(
        self, config_data: Dict[str, Any], section_id: str, default_key: str
    ) -> Dict[str, Any]:
        """
        A private helper to merge a specific section from a config file
        with its 'default' counterpart.

        This uses UtilsJson.get_merged_section to merge the specific ID (derived)
        with the 'default_key' (base), ignoring any internal 'base_id' chain for this step.
        """
        if default_key not in config_data:
            # If there's no default, just get the section
            specific_section = config_data.get(section_id)
            if not specific_section:
                raise ValueError(
                    f"ValErr:: Section ID '{section_id}' not found in the configuration.")
            return specific_section.copy()

        # Use UtilsJson to merge the specific section (derived) onto the default (base)
        # Note: 'base_key_identifier=None' is crucial here to prevent an inheritance chain traversal
        # if the section itself has a 'base_id' but we only want to merge with
        # 'default_key'.
        merged_section = UtilsJson.get_merged_section(
            config_data=config_data,
            derived_key_id=section_id,
            base_key_id=default_key,  # Explicit base ID for default merge
            base_key_identifier=None  # Do not follow inheritance chain in this step
        )
        return merged_section

    def _merge_indicator_params(
        self,
        indicator_ids: List[str],  # Changed to List[str] for clarity
        indicators_config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        A private helper to collect and merge all relevant indicator parameters
        based on the strategy's requirements.
        """
        merged_params = {}
        for indicator_id in indicator_ids:
            # Each indicator ID is merged with its default_indicator
            indicator_params = self._get_merged_section(
                indicators_config, indicator_id, "default_indicator"
            )
            merged_params[indicator_id] = indicator_params
        return {"indicators": merged_params}

    def _validate_and_setdefault(
            self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        A private helper to ensure all critical configuration parameters have values,
        providing defaults if necessary.
        """
        # Ensure 'ticker' and 'strategy_id' are present after all merging
        if "ticker" not in config:
            raise ValueError(
                "Configuration error: 'ticker' is a required field.")
        if "strategy_id" not in config:
            raise ValueError(
                "Configuration error: 'strategy_id' is a required field.")

        # Set any defaults if needed (e.g., initial_capital, timeframe)
        config.setdefault('initial_capital', 100000.0)
        config.setdefault('new_timeframe', '1d')

        return config

    # Removed recursive_add_config as its functionality is replaced by
    # UtilsJson.get_merged_section

    def load_combined_config(self, run_id: str) -> Dict[str, Any]:
        """
        Loads and merges configuration data for a specific run, assembling all required parameters.

        This method orchestrates the loading and merging of configuration files required for a backtest run.
        It relies on UtilsJson.get_merged_section to handle run configuration inheritance.

        Args:
            run_id (str): The identifier for the run whose configuration should be loaded.

        Returns:
            Dict[str, Any]: A comprehensive dictionary containing all parameters needed for the backtest.
        """

        # 1. Resolve run configuration inheritance chain using UtilsJson
        # This replaces the old recursive_add_config method.
        run_config = UtilsJson.get_merged_section(
            config_data=self.run_configs,
            derived_key_id=run_id,
            base_key_id=None,  # No explicit base ID
            base_key_identifier='base_id'  # Follow 'base_id' for inheritance
        )

        print(f"Loaded (and merged) run configuration for ID '{run_id}'.")

        # 2. Get and merge the strategy configuration
        strategy_id = run_config.get("strategy_id")
        if not strategy_id:
            raise ValueError(
                f"'strategy_id' not specified for run '{run_id}' after merging run config.")

        # Merge strategy with its default counterpart
        merged_strategy_config = self._get_merged_section(
            self.strategies_configs, strategy_id, "default_strategy"
        )

        # 3. Get and merge the required indicator parameters
        indicator_ids = merged_strategy_config.get("indicator_ids", [])
        merged_indicator_params = self._merge_indicator_params(
            indicator_ids, self.indicators_configs
        )

        # 4. Combine all configurations into a single dictionary (Run ->
        # Strategy -> Indicators)
        combined_config = {}
        # Order is important: Values from later updates overwrite earlier ones.
        # Strategy/Indicator params should usually override generic run_config
        # values

        # Start with strategy (it has core parameters)
        combined_config.update(merged_strategy_config)
        # Overlay run_config (e.g., specific ticker, date ranges)
        combined_config.update(run_config)
        # Add indicator parameters (namespaced under 'indicators')
        combined_config.update(merged_indicator_params)

        # 5. Validate the final configuration and set any defaults
        final_config = self._validate_and_setdefault(combined_config)

        print(f"Final combined configuration prepared for run ID '{run_id}'.")

        return final_config
