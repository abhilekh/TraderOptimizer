import json
from pathlib import Path
import pdb
from typing import Any, Dict
from .utils import Utils, UtilsJson

class ConfigManager:
    """
    ConfigManager is a dedicated class for managing hierarchical configuration files in a Python application, particularly for backtesting or trading systems. It is responsible for loading, merging, and validating multiple JSON configuration files, such as run configurations, strategy definitions, and indicator parameters.
    Key Responsibilities:
    - Load configuration data from multiple JSON files (run, strategies, indicators).
    - Merge configurations hierarchically, supporting inheritance via 'base_id' references.
    - Merge specific configuration sections with their 'default' counterparts for strategies and indicators.
    - Validate the presence of required configuration fields and provide defaults where necessary.
    - Assemble a comprehensive configuration dictionary for a given run, including all relevant strategy and indicator parameters.
    Typical Usage:
    Instantiate ConfigManager with paths to the configuration files, then use `load_combined_config(run_id)` to retrieve a fully merged and validated configuration for a specific run.
    Attributes:
        run_config_path (Path): Path to the run configuration JSON file.
        strategies_config_path (Path): Path to the strategies configuration JSON file.
        indicators_config_path (Path): Path to the indicators configuration JSON file.
        run_configs (Dict[str, Any]): Loaded run configurations.
        strategies_configs (Dict[str, Any]): Loaded strategies configurations.
        indicators_configs (Dict[str, Any]): Loaded indicators configurations.
    Methods:
        _get_merged_section(config_data, section_id, default_key):
            Merges a specific section from a config file with its 'default' counterpart.
        _merge_indicator_params(indicator_ids, indicators_config):
            Collects and merges all relevant indicator parameters for a strategy.
        _validate_and_setdefault(config):
            Ensures all critical configuration parameters have values, providing defaults if necessary.
        recursive_add_config(config_dict, config_id):
            Recursively merges configuration dictionaries based on a chain of 'base_id' references, preventing circular references.
        load_combined_config(run_id):
            Loads and merges configuration data for a specific run, assembling all required parameters for execution.
        ValueError: If required configuration fields are missing, configuration IDs are not found, or circular references are detected.
    """


    def __init__(
        self,
        run_config_path: Path,
        strategies_config_path: Path,
        indicators_config_path: Path,
    ):
        """
        Initializes the ConfigManager with paths to the configuration files.

        Args:
            run_config_path (Path): Path to the run_config.json file.
            strategies_config_path (Path): Path to the strategies.json file.
            indicators_config_path (Path): Path to the indicators.json file.
        """
        self.run_config_path = run_config_path
        self.strategies_config_path = strategies_config_path
        self.indicators_config_path = indicators_config_path

        # Load all configuration files
        self.run_configs = UtilsJson.read_json_file(self.run_config_path)
        self.strategies_configs = UtilsJson.read_json_file(self.strategies_config_path)
        self.indicators_configs = UtilsJson.read_json_file(self.indicators_config_path)

    def _get_merged_section(
        self, config_data: Dict[str, Any], section_id: str, default_key: str
    ) -> Dict[str, Any]:
        """
        A private helper to merge a specific section from a config file
        with its 'default' counterpart.
        """
        default_section = config_data.get(default_key, {})
        specific_section = config_data.get(section_id, {})

        if not specific_section:
            pdb.set_trace()
            raise ValueError(f"ValErr:: Section ID '{section_id}' not found in the configuration.")

        # Start with the default and update with specific settings
        merged_section = default_section.copy()
        merged_section.update(specific_section)
        return merged_section

    def _merge_indicator_params(
        self,
        indicator_ids: list,
        indicators_config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        A private helper to collect and merge all relevant indicator parameters
        based on the strategy's requirements.
        """
        merged_params = {}
        for indicator_id in indicator_ids:
            indicator_params = self._get_merged_section(
                indicators_config, indicator_id, "default_indicator"
            )
            merged_params[indicator_id] = indicator_params
        return {"indicators": merged_params}


    def _validate_and_setdefault(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        A private helper to ensure all critical configuration parameters have values,
        providing defaults if necessary.
        """
        if "ticker" not in config:
            raise ValueError("Configuration error: 'ticker' is a required field.")
        if "strategy_id" not in config:
            raise ValueError("Configuration error: 'strategy_id' is a required field.")

        return config
    
    from typing import Optional

    def recursive_add_config(self, config_dict: Dict[str, Any], config_id: Optional[str]) -> Dict[str, Any]:
        """
        Recursively merges configuration dictionaries based on a chain of 'base_id' references.
        Starting from the given `config_id`, this method traverses through the configuration hierarchy,
        merging each configuration dictionary into a single result. The merge order ensures that
        values from more specific configurations (closer to the original `config_id`) override those
        from their base configurations.
        Args:
            config_dict (Dict[str, Any]): A dictionary mapping configuration IDs to their respective configuration dictionaries.
            config_id (Optional[str]): The starting configuration ID to resolve and merge.
        Returns:
            Dict[str, Any]: The merged configuration dictionary for the given `config_id`, including all inherited base configurations.
        Raises:
            ValueError: If a referenced configuration ID is not found in `config_dict`, or if a circular reference is detected.
        Notes:
            - The method prevents infinite loops by tracking visited IDs and breaking on circular references.
            - The 'base_id' key in each configuration dictionary is used to determine the parent configuration.
        """
        visited_ids = set()
        merged_config: Dict = {}

        # Recursively add the config by id.
        while config_id is not None:
            if config_id in visited_ids:
                # Break circular loop
                break

            visited_ids.add(config_id)

            if config_id not in config_dict:
                raise ValueError(f"Config ID '{config_id}' not found in {self.run_config_path}")
            # Get the specific configuration
            current_config = config_dict.get(config_id)
            merged_config = {**current_config, **merged_config}  # type: ignore
            config_id = merged_config.get('base_id')
        return merged_config

    def load_combined_config(self, run_id: str) -> Dict[str, Any]:
        """
        Loads and merges configuration data for a specific run.

        This method orchestrates the loading and merging of configuration files required for a backtest run. It performs the following steps:
        1. Loads the run-specific configuration using the provided `run_id`.
        2. Retrieves and merges the associated strategy configuration, applying inheritance rules.
        3. Retrieves and merges the required indicator parameters based on the strategy configuration.
        4. Combines all configurations into a single dictionary.
        5. Validates the final configuration and sets any default values as needed.

        Args:
            run_id (str): The identifier for the run whose configuration should be loaded.

        Returns:
            Dict[str, Any]: A comprehensive dictionary containing all parameters needed for the backtest.

        Raises:
            ValueError: If the 'strategy_id' is not specified in the run configuration.
        """
        

        run_config = self.recursive_add_config(self.run_configs, run_id)

        print( f"Loaded configuration for run ID '{run_id}': {run_config}")

        # Get and merge the strategy configuration
        strategy_id = run_config.get("strategy_id")
        if not strategy_id:
            raise ValueError(f"'strategy_id' not specified for run '{run_id}'.")

        merged_strategy_config = self._get_merged_section(
            self.strategies_configs, strategy_id, "default_strategy"
        )

        # Get and merge the required indicator parameters
        indicator_ids = merged_strategy_config.get("indicator_ids", [])
        merged_indicator_params = self._merge_indicator_params(
            indicator_ids, self.indicators_configs
        )

        # Combine all configurations into a single dictionary
        combined_config = {}
        combined_config.update(run_config)
        combined_config.update(merged_strategy_config)
        combined_config.update(merged_indicator_params)

        # Validate the final configuration and set any defaults
        final_config = self._validate_and_setdefault(combined_config)

        print( f"Loaded configuration for run ID '{run_id}': {final_config}")

        return final_config