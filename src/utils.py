"""
Contains utility functions, primarily for plotting backtest results.
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from typing import Dict, Any, Optional

class Utils:
    
    @staticmethod
    def plot_results(
        df_results: pd.DataFrame,
        df_signals: pd.DataFrame,
        initial_capital: float,
        ticker: str,
        params: Dict[str, Any],
        timeframe: str,
        output_filepath: str
    ) -> None:
        """
        Plots the stock price with all four signal types and the equity curve.
        """
        if df_results.empty:
            print("No results to plot.")
            return
        
        if 'Signal' not in df_signals.columns:  # Ensure 'Signal' column exists
            print("No trading signals found in the results DataFrame.")
            return 
            

        print(f"Plotting results for {ticker} with parameters: {params} and timeframe: {timeframe}")

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True)

        # Plot Price and Signals
        ax1.plot(df_results.index, df_results['Close'], label='Close Price', color='blue', alpha=0.7)
        
        buy_signals = df_results[df_results['Signal'] == 1]
        buy_close_signals = df_results[df_results['Signal'] == 2]
        sell_signals = df_results[df_results['Signal'] == -1]
        sell_close_signals = df_results[df_results['Signal'] == -2]
        
        ax1.plot(buy_signals.index, df_results.loc[buy_signals.index]['Close'], '^', markersize=10, color='green', label='Buy Entry')
        ax1.plot(buy_close_signals.index, df_results.loc[buy_close_signals.index]['Close'], 'o', markersize=7, color='lime', label='Buy Close')
        ax1.plot(sell_signals.index, df_results.loc[sell_signals.index]['Close'], 'v', markersize=10, color='red', label='Sell Entry')
        ax1.plot(sell_close_signals.index, df_results.loc[sell_close_signals.index]['Close'], 'o', markersize=7, color='maroon', label='Sell Close')

        ax1.set_title(f'{ticker} Price with Trade Signals ({timeframe} Data)')
        ax1.set_ylabel('Price')
        ax1.legend()
        ax1.grid(True)

        print("Plotting equity curve...", df_results.columns)
        # Plot Equity Curve
        equity_col = 'Equity' if 'Equity' in df_results.columns else 'Equity_Curve'
        ax2.plot(df_results.index, df_results[equity_col], label='Equity Curve', color='purple')
        ax2.axhline(y=initial_capital, color='grey', linestyle='--', label='Initial Capital')

        ax2.set_title('Equity Curve')
        ax2.set_xlabel('Date')
        ax2.set_ylabel('Equity')
        ax2.legend()
        ax2.grid(True)

        plt.tight_layout()
        plt.savefig(output_filepath)
        print(f"Plot saved to: {output_filepath}")

class UtilsPath:
    """Utility class for path operations."""

    @classmethod
    def ensure_path_exists(cls, path: Path,  checkisDir=False) -> None:
        """Ensures that the given path exists, creating it if necessary."""
        if not path.exists():
            path.mkdir(parents=True, exist_ok=True)
        else:
            if checkisDir and not path.is_dir():
                raise NotADirectoryError(f"Expected a directory at {path}, but found a file.")
            

    @classmethod
    def append_file(cls, file_path: Path, data: Any, createPath: bool = True, extension: Optional[str] = None):
        """
        A helper to safely append to a file.
        Creates parent directories if they don't exist.

        Args:
            file_path (Path): The path to the output JSON file.
            data (Dict[str, Any]): The dictionary data to write.
        """
        try:
            cls.ensure_path_exists(file_path.parent, checkisDir=True)
            assert extension is not None and file_path.suffix == extension, f"Output file must be a {extension} file"

            with open(file_path, "a", encoding='utf-8') as f_ptr:
                f_ptr.write("".join("="*10)+"\n")
                if isinstance(data, list):
                    f_ptr.write("\n".join(data))
                elif isinstance(data, dict):
                    json.dump(data, f_ptr, indent=4)
                else:
                    f_ptr.write(str(data))
            print(f"Successfully wrote data to: {file_path}")

        except TypeError as e:
            raise TypeError(f"Data is not JSON serializable. Error: {e}")
        except OSError as e:
            raise OSError(f"Could not write to file at path: {file_path}. Error: {e}")
        
    @classmethod
    def write_file(cls, file_path: Path, data: Any, createPath: bool = True, extension: Optional[str] = None):
        """
        A helper to safely write to a file.
        Creates parent directories if they don't exist.

        Args:
            file_path (Path): The path to the output JSON file.
            data (Dict[str, Any]): The dictionary data to write.
        """
        try:
            cls.ensure_path_exists(file_path.parent, checkisDir=True)
            assert extension is not None and file_path.suffix == extension, f"Output file must be a {extension} file"

            with open(file_path, "w", encoding='utf-8') as f_ptr:
                if isinstance(data, list):
                    f_ptr.write("\n".join(data))
                elif isinstance(data, dict):
                    json.dump(data, f_ptr, indent=4)
                else:
                    f_ptr.write(str(data))
            print(f"Successfully wrote data to: {file_path}")

        except TypeError as e:
            raise TypeError(f"Data is not JSON serializable. Error: {e}")
        except OSError as e:
            raise OSError(f"Could not write to file at path: {file_path}. Error: {e}")
        
    @classmethod    
    def get_path(cls, config: dict, optimization_run=False):
        filename_prefix = f"backtest_{config['ticker']}_{config['strategy_id']}_{config['new_timeframe']}"
        if optimization_run:
            filename_prefix += f"_optimzer"
        


class UtilsJson:
    """Utility class for reading and writing JSON files."""

    @staticmethod
    def read_json_file(file_path: Path) -> Dict[str, Any]:
        """A helper to safely load any individual JSON configuration file.

        Args:
            file_path (Path): Path of file to read

        Raises:
            FileNotFoundError: _description_
            ValueError: _description_

        Returns:
            Dict[str, Any]: _description_
        """
        try:
            with open(file_path, "r") as fle:
                return json.load(fle)
        except FileNotFoundError:
            raise FileNotFoundError(f"Configuration file not found: {file_path}")
        except OSError:
            raise OSError(f"Not able to read the file at path: {file_path}")
        except json.JSONDecodeError:
            raise ValueError(f"Error decoding JSON from file: {file_path}")

    @staticmethod
    def write_json_file(file_path: Path, data: Dict[str, Any], createPath: bool = True) -> None:
        """
        A helper to safely write a dictionary to a JSON file.
        Creates parent directories if they don't exist.

        Args:
            file_path (Path): The path to the output JSON file.
            data (Dict[str, Any]): The dictionary data to write.
        """
        try:
            UtilsPath.ensure_path_exists(file_path.parent, checkisDir=True)

            with open(file_path, "w") as f:
                json.dump(data, f, indent=4)
            print(f"Successfully wrote JSON data to: {file_path}")

        except TypeError as e:
            raise TypeError(f"Data is not JSON serializable. Error: {e}")
        except OSError as e:
            raise OSError(f"Could not write to file at path: {file_path}. Error: {e}")




# class StrictDict:
#     """
#     A dictionary that raises an error if a key is not found.
#     This is useful for ensuring that all required keys are present.
#     """
#     def __init__(self, *args, **kwargs):
#         super().__init__(*args, **kwargs)

#     def __getitem__(self, key:str, type:typing): # type: ignore
#         raise KeyError(f"Key '{key}' access not allowed.")

#     def getCheck(self, key, key_type, default=None):
#         if key not in self and default is None:
#             raise KeyError(f"Key '{key}' not found in dictionary.")
#         return super().get(key, default)