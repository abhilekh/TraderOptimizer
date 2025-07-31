# runner.py

import os
from typing import Any, Callable, Dict, List, Union, Tuple

import matplotlib.pyplot as plt
import pandas as pd

import src.indicators as ind_calc
from src.backtester import analyze_performance, run_backtest
from src.data_manager import get_historical_data, resample_data
from src.strategy import (
    apply_moving_average_crossover_strategy,
    apply_rsi_ema_crossover_strategy
)
from utils import plot_results

# Pylint: Map strategy names from config to their functions
STRATEGY_MAP: Dict[str, Callable[[pd.DataFrame, Dict[str, Any]], pd.DataFrame]] = {
    "RSI_EMA_Crossover": apply_rsi_ema_crossover_strategy,
    "Moving_Average_Crossover": apply_moving_average_crossover_strategy
}

# Pylint: Map indicator names to their calculation functions
INDICATOR_CALC_MAP: Dict[str, Callable[..., Union[pd.Series, pd.DataFrame]]] = {
    "RSI": ind_calc.calculate_rsi,
    "EMA": ind_calc.calculate_ema,
    "SMA": ind_calc.calculate_sma
}

def _get_indicator_period(
    indicator_base_name: str,
    indicator_id: str,
    all_params: Dict[str, Any]
) -> Union[int, None]:
    """
    Determines the period for an indicator based on its ID and general config parameters.
    Handles specific cases like different SMA periods mapping to generic MA params.

    Args:
        indicator_base_name (str): The base name of the indicator (e.g., "RSI", "SMA").
        indicator_id (str): The specific ID of the indicator (e.g., "RSI_14", "SMA_50").
        all_params (Dict[str, Any]): The combined configuration dictionary.

    Returns:
        Union[int, None]: The period as an integer, or None if it cannot be determined.
    """
    if indicator_base_name == "SMA":
        # Attempt to extract period directly from ID (e.g., "SMA_50" -> 50)
        parts = indicator_id.split('_')
        if len(parts) > 1:
            try:
                return int(parts[1])
            except ValueError:
                pass # Fallback to general lookup if parsing fails

        # Fallback to specific period keys if not directly in ID
        if indicator_id == "SMA_20":
            return all_params.get("Short_MA_Period")
        if indicator_id == "SMA_50":
            return all_params.get("Long_MA_Period")
        if indicator_id == "SMA_200":
            return all_params.get("Long_MA_Period") # Assuming SMA_200 maps to Long_MA

    # For other indicators or if specific SMA lookup fails, use generic "_Period" key
    return all_params.get(f"{indicator_base_name}_Period")


def _get_indicator_input_data(
    data_frame_copy: pd.DataFrame,
    ind_spec: Dict[str, Any],
    all_params: Dict[str, Any]
) -> Tuple[Union[pd.DataFrame, pd.Series], str]:
    """
    Determines the data (DataFrame or Series) and column name to pass
    to an indicator calculation function.

    Args:
        data_frame_copy (pd.DataFrame): The current DataFrame being processed.
        ind_spec (Dict[str, Any]): The specification for the current indicator.
        all_params (Dict[str, Any]): The combined configuration dictionary.

    Returns:
        Tuple[Union[pd.DataFrame, pd.Series], str]:
            - The data (DataFrame or Series) to be used as input for the indicator.
            - The name of the column that data represents (e.g., 'Close', 'RSI_14').

    Raises:
        ValueError: If a required base indicator column for calculation is not found.
    """
    apply_on_column: str = 'Close'
    indicator_input_data: Union[pd.DataFrame, pd.Series] = data_frame_copy

    if "on_indicator" in ind_spec and ind_spec["on_indicator"] is not None:
        base_ind_id_for_input: str = ind_spec["on_indicator"]
        base_ind_name_parts: List[str] = base_ind_id_for_input.split('_')
        base_ind_base_name: str = base_ind_name_parts[0]
        
        # Reconstruct the expected column name of the *already calculated* base indicator
        base_ind_col_name: str = ""
        base_ind_period = _get_indicator_period(
            base_ind_base_name, base_ind_id_for_input, all_params
        )

        if base_ind_period is not None:
            base_ind_col_name = f"{base_ind_base_name}_{base_ind_period}"
        else:
            base_ind_col_name = base_ind_id_for_input # Fallback if period couldn't be determined

        if base_ind_col_name not in data_frame_copy.columns:
            raise ValueError(f"Base indicator '{base_ind_col_name}' not found for calculation.")

        indicator_input_data = data_frame_copy[base_ind_col_name]
        apply_on_column = base_ind_col_name # Keep track of the column name

    elif ind_spec.get("on_close", False):
        indicator_input_data = data_frame_copy['Close']
        apply_on_column = 'Close'

    return indicator_input_data, apply_on_column


def calculate_required_indicators(
    data_frame: pd.DataFrame,
    indicators_spec: List[Dict[str, Any]],
    all_params: Dict[str, Any]
) -> pd.DataFrame:
    """
    Calculates all indicators specified in the strategy config and adds them
    as new columns to the DataFrame.

    Args:
        data_frame (pd.DataFrame): The base DataFrame with OHLC prices.
        indicators_spec (List[Dict[str, Any]]): List of indicator specifications
                                                 from strategies.json.
        all_params (Dict[str, Any]): The full merged configuration to
                                    get indicator periods.

    Returns:
        pd.DataFrame: DataFrame with all specified indicators added.
    """
    data_frame_copy: pd.DataFrame = data_frame.copy()

    for ind_spec in indicators_spec:
        indicator_id: str = ind_spec.get("indicator_id", "")
        # Pylint: Avoid using single-character variable names unless well-understood
        base_indicator_name: str = indicator_id.split('_')[0]

        calc_func: Union[Callable[..., pd.Series], None] = \
            INDICATOR_CALC_MAP.get(base_indicator_name)
        if not calc_func:
            print(f"Warning: No calculation function found for indicator "
                  f"'{base_indicator_name}'. Skipping.")
            continue

        try:
            period: Union[int, None] = _get_indicator_period(
                base_indicator_name, indicator_id, all_params
            )
            if period is None:
                print(f"Error: Period for indicator '{indicator_id}' could not be determined. "
                      "Skipping.")
                continue

            indicator_input_data, apply_on_column = _get_indicator_input_data(
                data_frame_copy, ind_spec, all_params
            )
            
            final_col_name: str = f"{base_indicator_name}_{period}"

            # --- THE FIX IS HERE ---
            # Construct keyword arguments dynamically based on what the function accepts
            func_kwargs: Dict[str, Any] = {
                "data": indicator_input_data,
                "period": period
            }

            # Only add 'column' argument if the indicator function expects it
            # A simpler way: check if the indicator is RSI. RSI does not take 'column'.
            # EMA and SMA *do* take 'column' when given a DataFrame.
            if base_indicator_name in ["EMA", "SMA"] and isinstance(indicator_input_data, pd.DataFrame):
                func_kwargs["column"] = apply_on_column
            
            # Call the indicator calculation function with the determined kwargs
            data_frame_copy[final_col_name] = calc_func(**func_kwargs) # type: ignore

        except Exception as err: # Pylint: Catch specific exceptions
            print(f"Error calculating indicator {indicator_id}: {err}. Skipping.")
            continue

    data_frame_copy.dropna(inplace=True)
    return data_frame_copy


def _generate_output_filepaths(
    output_dir: str,
    output_filename_prefix: str,
    ticker: str,
    strategy_type: str,
    timeframe_name: str
) -> Tuple[str, str]:
    """
    Generates unique file paths for metrics and plots based on run parameters.

    Args:
        output_dir (str): The base directory for output files.
        output_filename_prefix (str): Prefix for generated filenames.
        ticker (str): The stock ticker.
        strategy_type (str): The name of the strategy.
        timeframe_name (str): The name of the timeframe (e.g., "Daily").

    Returns:
        Tuple[str, str]: A tuple containing (metrics_filepath, plot_filepath).
    """
    os.makedirs(output_dir, exist_ok=True)
    base_filename = f"{output_filename_prefix}_{ticker}_{strategy_type}_{timeframe_name}"
    metrics_filepath = os.path.join(output_dir, f"{base_filename}_metrics.txt")
    plot_filepath = os.path.join(output_dir, f"{base_filename}_plot.png")
    return metrics_filepath, plot_filepath

def run_backtest_for_timeframe(
    data_frame: pd.DataFrame,
    run_params: Dict[str, Any],
    timeframe_name: str
) -> None:
    """
    Runs the backtest simulation for a given DataFrame and timeframe.
    This function orchestrates the indicator calculation, strategy application,
    backtest execution, and result saving for a single timeframe.

    Args:
        data_frame (pd.DataFrame): DataFrame with OHLC data for the specific timeframe.
        run_params (Dict[str, Any]): The combined configuration parameters for the run.
        timeframe_name (str): Name of the current timeframe (e.g., "Daily", "Weekly").
    """
    ticker: str = run_params["ticker"]
    initial_capital: float = run_params["initial_capital"]
    output_dir: str = run_params["output_dir"]
    output_filename_prefix: str = run_params["output_filename_prefix"]
    show_plot: bool = run_params["show_plot"]
    strategy_type: str = run_params["strategy_type"]
    
    # Strategy parameters are now flattened directly into run_params
    # The strategy function will pick what it needs
    strategy_params: Dict[str, Any] = run_params

    print(f"\n--- Running Backtest for {ticker} ({timeframe_name}) "
          f"with {strategy_type} ---")

    if data_frame.empty:
        print(f"No data available for {timeframe_name} timeframe.")
        return

    # Calculate ALL required indicators
    data_frame_with_indicators: pd.DataFrame = calculate_required_indicators(
        data_frame, run_params["indicators_needed_spec"], run_params
    )

    if data_frame_with_indicators.empty:
        print(f"{timeframe_name} indicator calculation failed or resulted in empty data.")
        return

    # Select the strategy function
    selected_strategy_func: Callable[[pd.DataFrame, Dict[str, Any]], pd.DataFrame] = \
        STRATEGY_MAP.get(strategy_type) # type: ignore # Pylint: ignore warning for dynamic type
    if not selected_strategy_func:
        print(f"Error: Strategy type '{strategy_type}' not found in STRATEGY_MAP.")
        return

    # Apply the trading strategy
    data_frame_final: pd.DataFrame = \
        selected_strategy_func(data_frame_with_indicators, strategy_params)

    if not data_frame_final.empty:
        data_frame_equity: pd.DataFrame
        trades_completed: pd.DataFrame
        num_trades_completed: int
        data_frame_equity, trades_completed, num_trades_completed = \
            run_backtest(data_frame_final, initial_capital)

        if not data_frame_equity.empty:
            metrics_filepath, plot_filepath = _generate_output_filepaths(
                output_dir, output_filename_prefix, ticker, strategy_type, timeframe_name
            )

            analyze_performance(
                initial_capital,
                float(data_frame_equity['Equity_Curve'].iloc[-1]),
                trades_completed,
                metrics_filepath
            )

            # For plotting, extract periods if relevant to the strategy for display purposes
            rsi_period_for_plot = run_params.get("RSI_Period", 0) 
            ema_period_for_plot = run_params.get("EMA_Period", 0) 

            plot_results(
                data_frame_equity,
                initial_capital,
                ticker,
                rsi_period_for_plot,
                ema_period_for_plot,
                timeframe_name,
                plot_filepath
            )

            if show_plot:
                plt.show()
            else:
                plt.close('all') # Close plot if not showing to free memory
        else:
            print(f"{timeframe_name} backtest resulted in an empty equity curve.")
    else:
        print(f"{timeframe_name} strategy application failed or resulted in empty data.")